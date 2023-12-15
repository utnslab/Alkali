
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "mlir/IR/BuiltinDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;

namespace mlir {
namespace ep2 {

HandlerInOutAnalysis::HandlerInOutAnalysis(Operation *module) {
  // walk through and find all function targets.
  module->walk([&](FuncOp funcOp) {
    if (funcOp->getAttr("type").cast<StringAttr>().getValue() == "handler") {
      auto args = funcOp.getArguments();
      handler_in_arg_list[funcOp] = args;

      mlir::SmallVector<mlir::Value> returnops;
      funcOp->walk([&](ep2::ReturnOp op) {
        if (op.getNumOperands() != 0) {
          assert(op.getNumOperands() == 1);
            returnops.push_back(op.getOperand(0));
        }
      });
      handler_returnop_list[funcOp] = returnops;
      printf("add return ops: %d\n", returnops.size());
    }
  });
}
} // namespace ep2
} // namespace mlir
