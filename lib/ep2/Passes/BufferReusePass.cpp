#include "mlir/IR/BuiltinDialect.h"

#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace ep2 {

namespace {

Value getCurrentValueAt(BufferAnalysis &analysis, Value buf, Operation *op) {
  auto block = op->getBlock();
  for (auto &map : analysis.blockInput[block]) {
    auto it = llvm::find_if(map, [&](auto &pair) {
      return pair.second.source == buf;
    });
    if (it != map.end())
      return block->getArgument(it->first);
  }

  // Its not an input value, so it must be an value declared in the scope
  return buf;
}

void toInplace(BufferAnalysis &bufferAnalysis, Value buf,
               Value targetBuf = nullptr) {
  if (!targetBuf)
    targetBuf = buf;

  std::vector<std::reference_wrapper<OpOperand>> uses;
  for (auto &use : buf.getUses())
    uses.push_back(use);
  for (auto &_use : uses) {
    auto &use = _use.get();
    TypeSwitch<Operation *, void>(use.getOwner())
    .Case<EmitOp>([&](EmitOp op) {
      OpBuilder builder(op);
      // not emitting to the buffer, skip for now.
      if (op.getBuffer() != buf)
        return;
      auto newBuffer = getCurrentValueAt(bufferAnalysis, targetBuf, op);
      // if its inplace emit, we don't need to do anything
      if (op.getValue() != newBuffer) {
        int offset = bufferAnalysis.offsetAt[op];
        builder.create<EmitOffsetOp>(op.getLoc(), newBuffer, op.getValue(), offset);
      }
      op->erase();
    }) 
    .Case<ExtractOp>([&](ExtractOp op) {
      OpBuilder builder(op);
      auto newBuffer = getCurrentValueAt(bufferAnalysis, targetBuf, op);
      int offset = bufferAnalysis.offsetAt[op];
      auto newOp = builder.create<ExtractOffsetOp>(op.getLoc(), op.getType(), newBuffer, offset);
      op.replaceAllUsesWith(newOp.getResult());
      op->erase();
    })
    .Case<InitOp>([&](InitOp op) {
      OpBuilder builder(op);
      auto newBuffer = getCurrentValueAt(bufferAnalysis, targetBuf, op);
      op.setOperand(use.getOperandNumber(), newBuffer);
      // TODO(zhiyuang): follow the inter-function use chain 
    });
  }
}

} // local namespace

// TODO(zhiyuang): currently is not for beyond example
void BufferReusePass::runOnOperation() {
  auto &bufferAnalysis = getAnalysis<BufferAnalysis>();
  auto &bufferClasses = bufferAnalysis.bufferClasses;

  // convert the analysis result to source buffer -> target buffer
  std::vector<std::pair<Operation *, BlockArgument>> bufferMapping;
  // TODO(zhiyuang): the source is not decided by the relationship, read from opt
  for (auto it = bufferClasses.begin(); it != bufferClasses.end(); it++) {
    if (!it->isLeader())
      continue; // Ignore non-leader sets.
    BlockArgument source;
    std::vector<Operation *> targets;

    for (auto memIt = bufferClasses.member_begin(it);
         memIt != bufferClasses.member_end(); ++memIt) {
      auto value = Value(*memIt);
      if (auto ba = dyn_cast<BlockArgument>(value))
        source = ba;
      else if (auto op = value.getDefiningOp())
        targets.push_back(op);
    }

    for (auto target : targets)
      bufferMapping.emplace_back(target, source);
  }

  // Do the conversion. Currently only replace INSIDE a buffer. fix this later
  for (auto [op, ba] : bufferMapping) {
    auto initOp = dyn_cast<InitOp>(op);
    toInplace(bufferAnalysis, ba);
    toInplace(bufferAnalysis, initOp.getResult(), ba);
    // we erase the op, as all users are replaced
    initOp.erase();
  }

}

} // namespace ep2
} // namespace mlir