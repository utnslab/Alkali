
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

    auto followUpdateOps = [&](mlir::Operation* op) {
      while (op != nullptr) {
        if (!isa<emitc::CallOp>(op) || cast<emitc::CallOp>(op).getCallee() != "__ep2_intrin_struct_write") {
          break;
        }
        op = cast<emitc::CallOp>(op).getOperands()[1].getDefiningOp();
      }
      return op;
    };

    auto isLocBuf = [&](mlir::Operation* op) {
      if ((op = followUpdateOps(op)) == nullptr) {
        return false;
      }
      return isa<emitc::VariableOp>(op) && cast<emitc::VariableOp>(op).getValue().cast<emitc::OpaqueAttr>().getValue().starts_with("&_loc_buf");
    };

    auto isRtBuf = [&](mlir::Operation* op) {
      if ((op = followUpdateOps(op)) == nullptr) {
        return false;
      }
      mlir::Type ty = op->getResult(0).getType();
      return isa<emitc::OpaqueType>(ty) && cast<emitc::OpaqueType>(ty).getValue() == "struct __buf_t";
    };

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

    auto emitMemIntrinsic = [&, this](std::string intrinsic, bool isWrite, mlir::Operation* xfer, int offs, int sz) {
      int memberNum = getMemberPos(xfer, offs);

      int tag = op.getArgs().value().getValue()[2].cast<IntegerAttr>().getValue().getLimitedValue();
      llvm::SmallVector<Type> resTypes = {};
      // convey to cpp translator to generate member offset here
      mlir::ArrayAttr args = rewriter.getI32ArrayAttr({~memberNum, sz, tag});
      mlir::ArrayAttr templ_args;

      rewriter.create<emitc::CallOp>(op->getLoc(), resTypes, rewriter.getStringAttr(intrinsic), args, templ_args, ValueRange{xfer->getResult(0), isWrite ? op->getOperand(0) : op->getOperand(1)});

      // emit increment
      llvm::SmallVector<Type> resTypes3 = {};
      mlir::ArrayAttr args3 = rewriter.getI32ArrayAttr({isWrite, sz});
      mlir::ArrayAttr templ_args3;
      rewriter.create<emitc::CallOp>(op->getLoc(), resTypes3, rewriter.getStringAttr("__ep2_intrin_incr_offs"), args3, templ_args3, ValueRange{isWrite ? op->getOperand(0) : op->getOperand(1)});
    };

    auto decomposeMemcpy = [&, this](bool isWrite, mlir::Operation* xfer) {
      std::string opPrefix = isWrite ? "write" : "read";
      opPrefix = "__ep2_intrin_memcpy_mem_" + opPrefix;

      const auto& args = op.getArgs().value().getValue();
      int szOrig = args[1].cast<IntegerAttr>().getValue().getLimitedValue();
      assert(szOrig >= 0);
      int offOrig = args[0].cast<IntegerAttr>().getValue().getLimitedValue();

      // Netronome supports r/w aligned to 64,32,8-bit boundaries.
      // For simplicity, just use 32,8-bit. Max size is 128 for 32-bit, 32 for 8-bit.
      bool canUse32 = true;
      auto doEmission = [&](bool doEmit){
        int sz = szOrig;
        int off = offOrig;
        while (sz > 0) {
          if (getMemberPos(xfer, off) == -1) {
            canUse32 = false;
            break;
          }
          if (canUse32 && sz >= 4) {
            int sz32 = std::min(128, (sz/4)*4);
            if (doEmit) emitMemIntrinsic(opPrefix + std::string{"32"}, isWrite, xfer, off, sz32);
            sz -= sz32;
            off += sz32;
          } else if (sz >= 1) {
            int sz8 = std::min(32, sz);
            if (doEmit) emitMemIntrinsic(opPrefix + std::string{"8"}, isWrite, xfer, off, sz8);
            sz -= sz8;
            off += sz8;
          }
        }
      };

      doEmission(false);
      doEmission(true);
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

    if (isBuf0 && isBuf1) {
      // Bulk copy from one loc to another in MEM.
      llvm::SmallVector<Type> resTypes = {};
      mlir::ArrayAttr args = op.getArgs().value();
      mlir::ArrayAttr templ_args;
      auto bulkMemcpy = rewriter.create<emitc::CallOp>(op->getLoc(), resTypes, rewriter.getStringAttr("__ep2_intrin_memcpy_bulk_memcpy"), args, templ_args, op->getOperands());

      llvm::SmallVector<Type> resTypes3 = {};
      mlir::ArrayAttr args3 = rewriter.getI32ArrayAttr({2, INT32_MIN});
      mlir::ArrayAttr templ_args3;
      rewriter.create<emitc::CallOp>(op->getLoc(), resTypes3, rewriter.getStringAttr("__ep2_intrin_incr_offs"), args3, templ_args3, op->getOperands());

      rewriter.replaceOp(op, bulkMemcpy);
    } else if (isLoc0 && isBuf1) {
      // Copy from mem to transfer register (assume packet buffers stored in EMEM).
      // Assume rt_buf is aligned at alloc wide enough.
      mlir::Operation* locBuf = followUpdateOps(opd0);
      std::string newName = cast<emitc::VariableOp>(locBuf).getValue().cast<emitc::OpaqueAttr>().getValue().str() + "_xfer";
      mlir::Operation* xferVar = rewriter.create<emitc::VariableOp>(op->getLoc(), convertToXferType(locBuf->getResultTypes()[0]), emitc::OpaqueAttr::get(getContext(), newName));

      decomposeMemcpy(false, xferVar);

      llvm::SmallVector<Type> resTypes = {};
      mlir::ArrayAttr args;
      mlir::ArrayAttr templ_args;
      rewriter.replaceOpWithNewOp<emitc::CallOp>(op, resTypes, rewriter.getStringAttr("__ep2_intrin_xfer2gpr"), args, templ_args, ValueRange{xferVar->getResult(0), locBuf->getResult(0)});
    } else if (isBuf0 && isLoc1) {
      mlir::Operation* locBuf = followUpdateOps(opd1);
      mlir::Operation* xferVar = rewriter.create<emitc::VariableOp>(op->getLoc(), convertToXferType(locBuf->getResultTypes()[0]), emitc::OpaqueAttr::get(getContext(), cast<emitc::VariableOp>(locBuf).getValue().cast<emitc::OpaqueAttr>().getValue().str() + "_xfer"));

      llvm::SmallVector<Type> resTypes = {};
      mlir::ArrayAttr args;
      mlir::ArrayAttr templ_args;
      auto copy = rewriter.create<emitc::CallOp>(op->getLoc(), resTypes, rewriter.getStringAttr("__ep2_intrin_gpr2xfer"), args, templ_args, ValueRange{locBuf->getResult(0), xferVar->getResult(0)});

      decomposeMemcpy(true, xferVar);
      rewriter.replaceOp(op, copy);
    } else if (isLoc0) {
      mlir::Operation* locBuf = followUpdateOps(opd0);
      mlir::Operation* memBuf = followUpdateOps(opd1);

      llvm::SmallVector<Type> resTypes = {};
      mlir::ArrayAttr args;
      mlir::ArrayAttr templ_args;
      rewriter.replaceOpWithNewOp<emitc::CallOp>(op, resTypes, rewriter.getStringAttr("__ep2_intrin_mem2gpr"), args, templ_args, ValueRange{memBuf->getResult(0), locBuf->getResult(0)});
    } else if (isLoc1) {
      mlir::Operation* memBuf = followUpdateOps(opd0);
      mlir::Operation* locBuf = followUpdateOps(opd1);

      llvm::SmallVector<Type> resTypes = {};
      mlir::ArrayAttr args;
      mlir::ArrayAttr templ_args;
      rewriter.replaceOpWithNewOp<emitc::CallOp>(op, resTypes, rewriter.getStringAttr("__ep2_intrin_gpr2mem"), args, templ_args, ValueRange{locBuf->getResult(0), memBuf->getResult(0)});
    } else {
      op->dump();
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
