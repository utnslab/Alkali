#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/Transforms/Passes.h"

#include <list>


namespace mlir {
namespace ep2{

namespace {
  void insertNewBlockParameter(FuncOp funcOp) {
    OpBuilder builder(funcOp);
    for (auto &block : funcOp.getBody()) {
      builder.setInsertionPointToStart(&block);

      Value pred;
      if (block.hasNoPredecessors())
        pred = builder.create<ConstantOp>(funcOp.getLoc(), 1, 1);
      else
        pred = block.insertArgument(unsigned{0}, builder.getI1Type(), funcOp.getLoc());

      // Convert the branch ops
      auto &op = block.back();
      builder.setInsertionPoint(&op);
      if (auto brOp = dyn_cast<cf::BranchOp>(op)) {
        auto operands = llvm::to_vector(brOp.getDestOperands());
        operands.insert(operands.begin(), pred);
        builder.create<cf::BranchOp>(brOp.getLoc(), brOp.getDest(), operands);
        brOp.erase();
      } else if (auto condBrOp = dyn_cast<cf::CondBranchOp>(op)) {
        auto cond = condBrOp.getCondition();
        auto truePred = builder.create<arith::AndIOp>(condBrOp.getLoc(), builder.getI1Type(), cond, pred);
        auto notCond = builder.create<arith::SubIOp>(
            condBrOp.getLoc(), builder.getI1Type(),
            builder.create<ConstantOp>(funcOp.getLoc(), 1, 1), cond);
        auto falsePred = builder.create<arith::AndIOp>(condBrOp.getLoc(), builder.getI1Type(), notCond, pred);

        auto trueOperands = llvm::to_vector(condBrOp.getTrueDestOperands());
        trueOperands.insert(trueOperands.begin(), truePred);
        auto falseOperands = llvm::to_vector(condBrOp.getFalseDestOperands());
        falseOperands.insert(falseOperands.begin(), falsePred);

        builder.create<cf::CondBranchOp>(
            condBrOp.getLoc(), cond, condBrOp.getTrueDest(), trueOperands,
            condBrOp.getFalseDest(), falseOperands);
        condBrOp.erase();
      }

      // Insert the sink for final user
      if (!block.hasNoPredecessors() && block.hasNoSuccessors()) {
        builder.setInsertionPointToStart(&block);
        builder.create<SinkOp>(funcOp.getLoc(), mlir::ValueRange{pred});
      }
    }
  }

} // local namespace


void CFToPredPass::runOnOperation() {
  auto moduleOp = getOperation();
  moduleOp->walk(insertNewBlockParameter);
}

} // namespace ep2
} // namespace mlir