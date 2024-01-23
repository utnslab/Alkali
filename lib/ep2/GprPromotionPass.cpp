
#include "mlir/IR/BuiltinDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

using namespace mlir;

namespace mlir {
namespace ep2 {

void GprPromotionPass::runOnOperation() {
  /*
  getOperation()->walk([&](func::FuncOp fop) {
    if (fop.getName().starts_with("__event___handler")) {
      Block& b = fop.getRegion().front();
      auto builder = mlir::OpBuilder::atBlockBegin(&b);

      fop->walk([&](emitc::VariableOp op) {
        bool foldable = true;
        for (mlir::Operation* uop : op->getUsers()) {
          mlir::Value v = op->getResult(0);
          if (isa<emitc::PointerType>(v.getType()) &&
              isa<emitc::OpaqueType>(cast<emitc::PointerType>(v.getType()).getPointee())) {
            auto ty = cast<emitc::OpaqueType>(cast<emitc::PointerType>(v.getType()).getPointee()).getValue();
            if (ty.contains("__xrw") || ty.contains("__cls") || ty.contains("__lmem")) {
              foldable = false;
              continue;
            }
            if (isa<emitc::CallOp>(uop)) {
              llvm::StringRef name = cast<emitc::CallOp>(uop).getCallee();
              if (name == "__ep2_intrin_struct_write") {
                if (isa<emitc::PointerType>(uop->getOperand(0).getType())) {
                  foldable = false;
                  continue;
                }
              } else if (name == "__ep2_intrin_struct_access") {
                if (isa<emitc::PointerType>(uop->getResult(0).getType())) {
                  foldable = false;
                  continue;
                }
              } else if (name == "__ep2_intrin_gpr2xfer") {
              } else if (name == "__ep2_intrin_xfer2gpr") {
              } else {
                foldable = false;
                continue;
              }
            } else {
              foldable = false;
              continue;
            }
          }
        }
        if (foldable) {
          op->setAttr("doFold", builder.getBoolAttr(true));
        }
      });
    }
  });
  */
}

} // namespace ep2
} // namespace mlir
