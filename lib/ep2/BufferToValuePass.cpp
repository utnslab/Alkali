#include "mlir/IR/BuiltinDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

namespace {
  void reRefBuffers(FuncOp funcOp) {
    if (funcOp.isExtern())
      return;

    // insert all buffers from function arugments
    OpBuilder builder(funcOp);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    std::vector<std::pair<Value, Value>> buffers{};
    for (auto &ba : funcOp.getArguments()) {
      if (ba.getType().isa<BufferType>()) {
        auto reRefOp = builder.create<ReRefOp>(
            funcOp.getLoc(), builder.getType<BufferType>(), ba);
        buffers.emplace_back(ba, reRefOp);
      }
    }

    // insert buffer REREF from ops returning a buffer
    funcOp.walk([&](Operation *op) {
      if (op->getNumResults() != 1)
        return;
      // no double reref
      if (isa<ReRefOp>(op))
        return;
      auto ret = op->getResult(0);
      if (!ret.getType().isa<BufferType>())
        return;

      builder.setInsertionPointAfter(op);
      auto reRefOp = builder.create<ReRefOp>(
          op->getLoc(), builder.getType<BufferType>(), ret);
      buffers.emplace_back(ret, reRefOp);
    });

    // reref all buffers
    for (auto &[buf, reRef] : buffers) {
      // collect and rewrite all uses
      std::vector<std::reference_wrapper<OpOperand>> uses;
      for (auto &use : buf.getUses())
        uses.push_back(use);
      for (auto use : uses) {
        auto op = use.get().getOwner();
        if (op == reRef.getDefiningOp())
          continue;
        if ((isa<EmitOp>(op) && use.get().getOperandNumber() == 0) ||
            (isa<ExtractOp>(op) && use.get().getOperandNumber() == 0)) {
          use.get().set(reRef);
          continue;
        }
        // if its a normal use, use the derefed result
        builder.setInsertionPoint(op);
        auto deRefOp = builder.create<DeRefOp>(
            funcOp.getLoc(), builder.getType<BufferType>(), reRef);
        use.get().set(deRefOp);
      }
    }
  }
  
} // namespace Patterns

// Conversion Pass
void BufferToValuePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // first, we find all values
  moduleOp.walk(reRefBuffers);

  // execute mem2reg on all functions
  OpPassManager pm;
  auto &funcPm = pm.nest<FuncOp>();
  funcPm.addPass(createConvertSCFToCFPass());
  funcPm.addPass(createMem2Reg());

  if (failed(runPipeline(pm, moduleOp)))
    return signalPassFailure();
}

} // namespace ep2
} // namespace mlir
