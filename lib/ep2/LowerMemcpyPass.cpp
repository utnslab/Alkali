
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

    auto isLocBuf = [&, this](mlir::Operation* op) {
      if ((op = followUpdateOps(op)) == nullptr) {
        return false;
      }
      return isa<emitc::VariableOp>(op) && cast<emitc::VariableOp>(op).getValue().cast<emitc::OpaqueAttr>().getValue().starts_with("&_loc_buf");
    };

    auto isRtBuf = [&, this](mlir::Operation* op) {
      if ((op = followUpdateOps(op)) == nullptr) {
        return false;
      }
      if (!isa<emitc::CallOp>(op)) {
        return false;
      }
      emitc::CallOp callOp = cast<emitc::CallOp>(op);
      if (callOp.getCallee() == "alloc_packet_buffer") {
        return true;
      }
      // trace back to wrapper
      while (op != nullptr) {
        if (!isa<emitc::CallOp>(op) || cast<emitc::CallOp>(op).getCallee() != "__ep2_intrin_struct_access") {
          break;
        }
        op = op->getOperands()[0].getDefiningOp();
      }
      return op == nullptr;
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
      assert(false && "Unreachable.");
      return 0;
    };

    auto emitMemIntrinsic = [&, this](std::string intrinsic, bool isWrite, mlir::Operation* xfer, int srcOffs, int dstOffs, int sz) {
      std::string structType = cast<emitc::OpaqueType>(cast<emitc::PointerType>(xfer->getResult(0).getType()).getPointee()).getValue().str();
      structType = structType.substr(structType.find("struct ")+7);
      int memberNum = translateStructOffset(isWrite ? srcOffs : dstOffs, structType);

      llvm::SmallVector<Type> resTypes = {};
      // convey to cpp translator to generate member offset here
      mlir::ArrayAttr args = rewriter.getI32ArrayAttr({isWrite ? dstOffs : srcOffs, ~memberNum, sz});
      mlir::ArrayAttr templ_args;

      rewriter.create<emitc::CallOp>(op->getLoc(), resTypes, rewriter.getStringAttr(intrinsic), args, templ_args, ValueRange{xfer->getResult(0), isWrite ? op->getOperand(0) : op->getOperand(1)});
    };

    auto decomposeMemcpy = [&, this](bool isWrite, mlir::Operation* xfer) {
      std::string opPrefix = isWrite ? "write" : "read";
      opPrefix = "__ep2_intrin_memcpy_mem_" + opPrefix;

      const auto& args = op.getArgs().value().getValue();
      int sz = args[2].cast<IntegerAttr>().getValue().getLimitedValue();
      assert(sz >= 0);
      int off0 = args[0].cast<IntegerAttr>().getValue().getLimitedValue();
      int off1 = args[1].cast<IntegerAttr>().getValue().getLimitedValue();

      // Netronome supports r/w aligned to 64,32,8-bit boundaries.
      // For simplicity, just use 32,8-bit. Max size is 128
      while (sz > 0) {
        if (sz >= 4) {
          int sz32 = std::min(128, (sz/4)*4);
          emitMemIntrinsic(opPrefix + std::string{"32"}, isWrite, xfer, off0, off1, sz32);
          sz -= sz32;
          off0 += sz32;
          off1 += sz32;
        } else if (sz >= 1) {
          emitMemIntrinsic(opPrefix + std::string{"8"}, isWrite, xfer, off0, off1, sz);
          sz = 0;
          off0 += sz;
          off1 += sz;
        }
      }
    };

    auto convertToXferType = [&](mlir::Type ty) {
      auto pointeeTy = cast<emitc::PointerType>(ty).getPointee();
      std::string baseStructTy = cast<emitc::OpaqueType>(pointeeTy).getValue().str();
      return rewriter.getType<emitc::PointerType>(rewriter.getType<emitc::OpaqueType>("__xrw " + baseStructTy));
    };

    if (isRtBuf(opd0) && isRtBuf(opd1)) {
      // Bulk copy from one loc to another in MEM.
      llvm::SmallVector<Type> resTypes = {};
      mlir::ArrayAttr args = op.getArgs().value();
      mlir::ArrayAttr templ_args;
      rewriter.replaceOpWithNewOp<emitc::CallOp>(op, resTypes, rewriter.getStringAttr("__ep2_intrin_memcpy_bulk_memcpy"), args, templ_args, op->getOperands());
    } else if (isLocBuf(opd0) && isRtBuf(opd1)) {
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
    } else if (isRtBuf(opd0) && isLocBuf(opd1)) {
      mlir::Operation* locBuf = followUpdateOps(opd1);
      mlir::Operation* xferVar = rewriter.create<emitc::VariableOp>(op->getLoc(), convertToXferType(locBuf->getResultTypes()[0]), emitc::OpaqueAttr::get(getContext(), cast<emitc::VariableOp>(locBuf).getValue().cast<emitc::OpaqueAttr>().getValue().str() + "_xfer"));

      llvm::SmallVector<Type> resTypes = {};
      mlir::ArrayAttr args;
      mlir::ArrayAttr templ_args;
      auto copy = rewriter.create<emitc::CallOp>(op->getLoc(), resTypes, rewriter.getStringAttr("__ep2_intrin_gpr2xfer"), args, templ_args, ValueRange{locBuf->getResult(0), xferVar->getResult(0)});

      decomposeMemcpy(true, xferVar);
      rewriter.replaceOp(op, copy);
    } else {
      assert(false && "Memcpy not supported");
    }

    return success();
  }
};

void LowerMemcpyPass::runOnOperation() {
  auto module = getOperation();

  LocalAllocAnalysis& laa = getAnalysis<LocalAllocAnalysis>();
  CollectInfoAnalysis& cia = *getCachedAnalysis<CollectInfoAnalysis>();
  
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<LowerMemcpyPattern>(&getContext(), cia, laa);
  auto res = applyPatternsAndFoldGreedily(module, std::move(patterns));
  assert(res.succeeded());
}

}
}
