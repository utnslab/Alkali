
#include "mlir/IR/BuiltinDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cassert>
#include <string>
#include <climits>
#include <algorithm>

#include "Utils.h"

using namespace mlir;

namespace mlir {
namespace ep2 {

struct LowerMemcpyPattern : public OpRewritePattern<emitc::CallOp> {

  CollectInfoAnalysis& structsInfo;
  LocalAllocAnalysis& analyzer;

  LowerMemcpyPattern(MLIRContext* context, CollectInfoAnalysis& structsInfo, LocalAllocAnalysis& laa) : OpRewritePattern<emitc::CallOp>(context, /*benefit=*/ 1), structsInfo(structsInfo), analyzer(laa) {}

  LogicalResult matchAndRewrite(emitc::CallOp op, PatternRewriter& rewriter) const override {
    if (op.getCallee() != "__ep2_intrin_memcpy") {
      return failure();
    }
    mlir::Operation* opd0 = op->getOperands()[0].getDefiningOp();
    mlir::Operation* opd1 = op->getOperands()[1].getDefiningOp();

    // Struct update ops are SSA in EP2, so follow chain to source.
    auto followUpdateOps = [&](mlir::Operation* op) {
      while (op != nullptr) {
        if (!isa<emitc::CallOp>(op) || cast<emitc::CallOp>(op).getCallee() != "__ep2_intrin_struct_write") {
          break;
        }
        op = cast<emitc::CallOp>(op).getOperands()[1].getDefiningOp();
      }
      return op;
    };

    /*
    Cases where an op is a local buffer:
    1) Allocated buffer by LocalAllocAnalysis
    2) Traces back to an event queue work item (via struct access ops)
    */
    auto isLocBuf = [&](mlir::Operation* op) {
      if ((op = followUpdateOps(op)) == nullptr) {
        return false;
      }
      if (isa<emitc::VariableOp>(op) && cast<emitc::VariableOp>(op).getValue().cast<emitc::OpaqueAttr>().getValue().starts_with("&_loc_buf")) return true;

      // check for struct access ops
      while (op != nullptr) {
        if (!isa<emitc::CallOp>(op) || cast<emitc::CallOp>(op).getCallee() != "__ep2_intrin_struct_access" || !isa<emitc::PointerType>(op->getResult(0).getType())) {
          break;
        }
        op = cast<emitc::CallOp>(op).getOperands()[0].getDefiningOp();
      }
      if (isa<emitc::VariableOp>(op) && cast<emitc::VariableOp>(op).getValue().cast<emitc::OpaqueAttr>().getValue().starts_with("&work")) return true;
      return false;
    };

    // Is this a packet buffer?
    auto isRtBuf = [&](mlir::Operation* op) {
      if ((op = followUpdateOps(op)) == nullptr) {
        return false;
      }
      mlir::Type ty = op->getResult(0).getType();
      return isa<emitc::OpaqueType>(ty) && cast<emitc::OpaqueType>(ty).getValue() == "struct __buf_t";
    };

    /*
    Netronome only accepts writes at struct offsets with member variable, e.g.
    mem_write(&v9->f2), not mem_write(((char*) &v9) + 8) if f2 is at offset=8.
    Hence, translate offset (e.g. 8) to member offset. If in middle, fail.
    */
    auto translateStructOffset = [&](unsigned offs, std::string structName) {
      for (const auto& pr : structsInfo.structDefs) {
        if (pr.first.find(structName) != std::string::npos) {
          mlir::LLVM::LLVMStructType strTy = pr.second;
          unsigned partialSum = 0;
          for (int i = 0; i<strTy.getBody().size(); ++i) {
            assert(partialSum <= offs);
            if (partialSum == offs) {
              return i;
            }
            if (strTy.getBody()[i].isIntOrFloat()) {
              partialSum += strTy.getBody()[i].getIntOrFloatBitWidth() / 8;
            } else if (isa<mlir::LLVM::LLVMPointerType>(strTy.getBody()[i])) {
              partialSum += 8;
            }
          }
        }
      }
      return -1;
    };

    auto getMemberPos = [&](mlir::Operation* xfer, int offs) {
      std::string structType = cast<emitc::OpaqueType>(cast<emitc::PointerType>(xfer->getResult(0).getType()).getPointee()).getValue().str();
      structType = structType.substr(structType.find("struct ")+7);
      int memberNum = translateStructOffset(offs, structType);
      return memberNum;
    };

    /*
    Generate actual memory intrinsic (mem_write32, mem_write8, cls_write, cls_read)
    Args:
      intrinsic: intrinsic name
      isWrite: is this a read or write op
      usePktBuf: does this touch the packet buffer? (e.g. touch buffer offset?)
      xfer: which op represents the transfer register we are copying to/from?
      offs: offset in the other operand (not transfer register)
      szCopy: how many bytes to copy
      szAdvance: how many bytes to advance pointer from offs (not same sometimes), e.g.
        read 16 bytes from packet buffer into 4-byte aligned register for faster access,
        but actually only loading 14 bytes of use. Last 2 bytes are junk.
    */
    auto emitMemIntrinsic = [&](std::string intrinsic, bool isWrite, bool usePktBuf, mlir::Operation* xfer, int offs, int szCopy, int szAdvance) {
      int memberNum = getMemberPos(xfer, offs);
      int tag = op.getArgs().value().getValue()[2].cast<IntegerAttr>().getValue().getLimitedValue();

      // We set the buffer offset when using zero-copy optimization.
      const auto& opArgs = op.getArgs().value().getValue();
      int setOffset = -1;
      if (opArgs.size() >= 4) {
        setOffset = offs + opArgs[3].cast<IntegerAttr>().getValue().getLimitedValue();
      }

      auto emitOffsetCmd = [&](bool doSetOffset) {
        // emit increment
        llvm::SmallVector<Type> resTypes3 = {};
        mlir::ArrayAttr args3 = rewriter.getI32ArrayAttr({isWrite, szAdvance, doSetOffset ? setOffset : -1});
        mlir::ArrayAttr templ_args3;
        rewriter.create<emitc::CallOp>(op->getLoc(), resTypes3, rewriter.getStringAttr("__ep2_intrin_incr_offs"), args3, templ_args3, ValueRange{isWrite ? op->getOperand(0) : op->getOperand(1)});
      };

      if (usePktBuf && setOffset != -1) {
        emitOffsetCmd(true);
      }

      llvm::SmallVector<Type> resTypes = {};
      // convey to cpp translator to generate member offset here
      mlir::ArrayAttr args = rewriter.getI32ArrayAttr({~memberNum, szCopy, tag});
      mlir::ArrayAttr templ_args;

      // rewrite intrinsic, since netronome memcpy intrinsics are named funny sometimes.
      if (intrinsic == "__ep2_intrin_memcpy_cls_write32") {
        intrinsic = "__ep2_intrin_memcpy_cls_write";
      } else if (intrinsic == "__ep2_intrin_memcpy_cls_read32") {
        intrinsic = "__ep2_intrin_memcpy_cls_read";
      } else if (intrinsic == "__ep2_intrin_memcpy_cls_write8") {
        intrinsic = "__ep2_intrin_ctx_write";
        args = rewriter.getI32ArrayAttr({memberNum, false, true});
      } else if (intrinsic == "__ep2_intrin_memcpy_cls_read8") {
        assert(false && "Unsupported now");
      }
      rewriter.create<emitc::CallOp>(op->getLoc(), resTypes, rewriter.getStringAttr(intrinsic), args, templ_args, ValueRange{xfer->getResult(0), isWrite ? op->getOperand(0) : op->getOperand(1)});

      if (setOffset == -1 && usePktBuf) {
        emitOffsetCmd(false);
      }
    };

    auto decomposeMemcpy = [&](bool isWrite, bool usePktBuf, MemType mem, mlir::Operation* xfer) {
      // encodes info for TranslateToCpp file.
      std::string opPrefix = isWrite ? "_write" : "_read";
      opPrefix = std::string{"__ep2_intrin_memcpy"} + (usePktBuf ? "buf_" : "_") + toStringFunc(mem) + opPrefix;

      const auto& args = op.getArgs().value().getValue();
      int szOrig = args[1].cast<IntegerAttr>().getValue().getLimitedValue();
      int szAdvance = szOrig;
      assert(szOrig >= 0);

      if (!(usePktBuf && isWrite)) {
        /* we are copying into a register, which is size multiple of 4 bytes.
           hence, we can read junk off the boundaries, up to 4 bytes OR writing into a padded CLS field for context */
        szOrig = ((szOrig + 3) / 4) * 4;
      }

      int offOrig = args[0].cast<IntegerAttr>().getValue().getLimitedValue();

      /*
      Emit as much as possible using bigger instructions, e.g. cls_read/mem_read32
      Then do rest with slower instructions.
      */
      auto doEmission = [&](bool doEmit, bool& canUseFast, int fastMaxSize, int fastAlign, int slowMaxSize, int slowAlign) {
        int sz = szOrig;
        int off = offOrig;
        while (sz > 0) {
          if (getMemberPos(xfer, off) == -1) {
            canUseFast = false;
            break;
          }
          if (canUseFast && sz >= fastAlign) {
            int szFast = std::min(fastMaxSize, (sz/fastAlign)*fastAlign);
            if (doEmit) emitMemIntrinsic(opPrefix + std::to_string(fastAlign*8), isWrite, usePktBuf, xfer, off, szFast, std::min(szFast, szAdvance - off));
            sz -= szFast;
            off += szFast;
          } else if (sz >= slowAlign) {
            int szSlow = std::min(slowMaxSize, sz);
            if (doEmit) emitMemIntrinsic(opPrefix + std::to_string(slowAlign*8), isWrite, usePktBuf, xfer, off, szSlow, std::min(szSlow, szAdvance - off));
            sz -= szSlow;
            off += szSlow;
          }
        }
      };

      assert(mem == MemType::EMEM || mem == MemType::CLS);
      if (mem == MemType::EMEM) {
        // Netronome supports r/w aligned to 64,32,8-bit boundaries.
        // For simplicity, just use 32,8-bit. Max size is 128 for 32-bit, 32 for 8-bit.
        bool canUse32 = true;
        doEmission(false, canUse32, 128, 4, 32, 1);
        doEmission(true, canUse32, 128, 4, 32, 1);
      } else if (mem == MemType::CLS) {
        bool canUse32 = true;
        doEmission(false, canUse32, 128, 4, 32, 1);
        doEmission(true, canUse32, 128, 4, 32, 1);
      }
    };

    auto convertToXferType = [&](mlir::Type ty) {
      auto pointeeTy = cast<emitc::PointerType>(ty).getPointee();
      std::string baseStructTy = cast<emitc::OpaqueType>(pointeeTy).getValue().str();
      return rewriter.getType<emitc::PointerType>(rewriter.getType<emitc::OpaqueType>("__xrw " + baseStructTy));
    };

    bool isBuf0 = isRtBuf(opd0);
    bool isLoc0 = isLocBuf(opd0);
    bool isBuf1 = isRtBuf(opd1);
    bool isLoc1 = isLocBuf(opd1);
    
    auto getNewXferName = [&](mlir::Operation* locBuf) {
      if (isa<emitc::VariableOp>(locBuf)) {
        return cast<emitc::VariableOp>(locBuf).getValue().cast<emitc::OpaqueAttr>().getValue().str() + "_xfer";
      } else if (isa<emitc::CallOp>(locBuf) && cast<emitc::CallOp>(locBuf).getCallee() == "__ep2_intrin_struct_access") {
        return std::string{"_work_"} + std::to_string(cast<emitc::CallOp>(locBuf).getArgs().value().getValue()[0].cast<mlir::IntegerAttr>().getValue().getSExtValue()) + "_xfer";
      } else {
        assert(false && "Unreachable");
      }
    };

    /*
    3 cases for memcpy:
    1) packet buffer to packet buffer
    2) local buffer to some memory (cls, mem)
    3) some memory (cls, mem) to local buffer
    */
    if (isBuf0 && isBuf1) {
      // Bulk copy from one loc to another in MEM.
      llvm::SmallVector<Type> resTypes = {};
      mlir::ArrayAttr args = op.getArgs().value();
      mlir::ArrayAttr templ_args;
      auto bulkMemcpy = rewriter.create<emitc::CallOp>(op->getLoc(), resTypes, rewriter.getStringAttr("__ep2_intrin_memcpybuf_bulk_memcpy"), args, templ_args, op->getOperands());

      llvm::SmallVector<Type> resTypes3 = {};
      mlir::ArrayAttr args3 = rewriter.getI32ArrayAttr({2, INT32_MIN, -1});
      mlir::ArrayAttr templ_args3;
      rewriter.create<emitc::CallOp>(op->getLoc(), resTypes3, rewriter.getStringAttr("__ep2_intrin_incr_offs"), args3, templ_args3, op->getOperands());

      rewriter.replaceOp(op, bulkMemcpy);
    } else if (isLoc0) {
      // Copy from mem to transfer register (assume packet buffers stored in EMEM).
      // Assume rt_buf is aligned at alloc wide enough.
      mlir::Operation* locBuf = followUpdateOps(opd0);
      mlir::Operation* xferVar = rewriter.create<emitc::VariableOp>(op->getLoc(), convertToXferType(locBuf->getResultTypes()[0]), emitc::OpaqueAttr::get(getContext(), getNewXferName(locBuf)));

      decomposeMemcpy(false, isBuf1, isBuf1 ? MemType::EMEM : MemType::CLS, xferVar);

      llvm::SmallVector<Type> resTypes = {};
      mlir::ArrayAttr args;
      mlir::ArrayAttr templ_args;
      rewriter.replaceOpWithNewOp<emitc::CallOp>(op, resTypes, rewriter.getStringAttr("__ep2_intrin_xfer2gpr"), args, templ_args, ValueRange{xferVar->getResult(0), locBuf->getResult(0)});
    } else if (isLoc1) {
      mlir::Operation* locBuf = followUpdateOps(opd1);
      mlir::Operation* xferVar = rewriter.create<emitc::VariableOp>(op->getLoc(), convertToXferType(locBuf->getResultTypes()[0]), emitc::OpaqueAttr::get(getContext(), getNewXferName(locBuf)));

      llvm::SmallVector<Type> resTypes = {};
      mlir::ArrayAttr args;
      mlir::ArrayAttr templ_args;
      auto copy = rewriter.create<emitc::CallOp>(op->getLoc(), resTypes, rewriter.getStringAttr("__ep2_intrin_gpr2xfer"), args, templ_args, ValueRange{locBuf->getResult(0), xferVar->getResult(0)});

      decomposeMemcpy(true, isBuf0, isBuf0 ? MemType::EMEM : MemType::CLS, xferVar);
      rewriter.replaceOp(op, copy);
    } else {
      op->dump();
      llvm::errs() << "L0: " << isLoc0 << " L1: " << isLoc1 << " B0: " << isBuf0 << " B1: " << isBuf1 << '\n';
      // should be a context/table copy.
      assert(false && "Unhandled memcpy");
      return failure();
    }

    return success();
  }
};

void LowerMemcpyPass::runOnOperation() {
  auto module = getOperation();

  LocalAllocAnalysis& laa = *getCachedAnalysis<LocalAllocAnalysis>();
  CollectInfoAnalysis& cia = *getCachedAnalysis<CollectInfoAnalysis>();
  
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<LowerMemcpyPattern>(&getContext(), cia, laa);
  auto res = applyPatternsAndFoldGreedily(module, std::move(patterns));
  assert(res.succeeded());
}

}
}
