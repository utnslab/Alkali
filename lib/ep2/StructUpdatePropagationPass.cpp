
#include "mlir/IR/BuiltinDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

void StructUpdatePropagationPass::runOnOperation() {
  auto followStructUpdates = [&](mlir::Value v) {
    mlir::Operation* op = v.getDefiningOp();
    while (op != nullptr && isa<emitc::CallOp>(op) && cast<emitc::CallOp>(op).getCallee() == "__ep2_intrin_struct_write") {
      v = op->getOperand(1);
      op = op->getOperand(1).getDefiningOp();
    }
    return v;
  };

  getOperation()->walk([&](mlir::Operation* op) {
    for (int i = 0; i<op->getOperands().size(); ++i) {
      op->setOperand(i, followStructUpdates(op->getOperand(i)));
    }
  });
}

} // namespace ep2
} // namespace mlir
