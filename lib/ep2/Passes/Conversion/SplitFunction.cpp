#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/EquivalenceClasses.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"

#include "polygeist/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

// Analysis pass for converting the pointer value to LLVM type
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/Passes.h"

#include <algorithm>
#include <vector>

namespace mlir {
namespace ep2 {

void splitBlock(Block *srcBlock, Block *dstBlock, llvm::DenseSet<Operation*> &sourceOps, llvm::DenseSet<Value> &sourceArguments) {
  auto insertPoint = Block::iterator(dstBlock->getTerminator());
  // We need to move all in the 
  for (auto &op : *srcBlock) {
    if (!sourceOps.contains(&op))
      op.moveBefore(dstBlock, insertPoint);
  }
}

static bool inSink(Operation * op) {
  return op->hasAttr("sink");
}
static bool inSink(Value value) {
  // TODO: find a way to mark BA
  if (isa<BlockArgument>(value))
    return false;
  else if (isa<OpResult>(value))
    return inSink(cast<OpResult>(value).getOwner());
  
  llvm_unreachable("Unknown value type");
  return false;
}

static void getValuesToTransfer(ep2::FuncOp funcOp, llvm::DenseSet<Value> &values) {
  for (auto &op : funcOp.getOps()) {
    if (!inSink(&op))
      continue;
    
    for (auto operand : op.getOperands()) {
      if (!inSink(operand))
        values.insert(operand);
    }
  }
}

std::pair<ep2::InitOp, ep2::ReturnOp> createGenerate(OpBuilder &builder, mlir::Location loc, StringRef name, ArrayRef<Value> values) {
  auto initOp = builder.create<ep2::InitOp>(loc, name, values);
  return std::make_pair(initOp, builder.create<ep2::ReturnOp>(loc, ValueRange{initOp}));
}

template<typename F>
void eraseOpsIf(ep2::FuncOp funcOp, F &&pred) {
  std::vector<Operation *> opvec;
  for (auto &op : funcOp.getOps())
      if (pred(&op))
        opvec.push_back(&op);
  std::for_each(opvec.rbegin(), opvec.rend(),
                [](Operation *op) { op->erase(); });
}

void tryAddTerminator(OpBuilder &builder, ep2::FuncOp funcOp) {
  auto &lastBlock = funcOp.getBlocks().back();
  if (!lastBlock.mightHaveTerminator()) {
    builder.setInsertionPointToEnd(&lastBlock);
    builder.create<ep2::TerminateOp>(funcOp.getLoc());
  }
}

static void removeSinkAttr(ep2::FuncOp funcOp) {
  for (auto &op : funcOp.getOps()) {
    if (op.hasAttr("sink"))
      op.removeAttr("sink");
  }
}

static void removeEmptyBlocks(ep2::FuncOp funcOp) {
  auto blocks = llvm::map_to_vector(funcOp.getBlocks(), [](Block &block) { return &block; });

  // try to collapse the blocks
  for (int i = blocks.size() - 1; i > 0; i--) {
    auto block = blocks[i], prevBlock = blocks[i - 1];
    if (block->hasNoPredecessors()) {
      auto ops = llvm::map_to_vector(block->getOperations(), [](Operation &op) { return &op; });

      for (auto op : ops)
        op->moveBefore(prevBlock, prevBlock->end());
    }

    if (block->empty())
      block->erase();
  }
}

std::pair<ep2::FuncOp, ep2::FuncOp> functionSplitter(ep2::FuncOp funcOp, llvm::DenseSet<Operation *> &sinkOps, llvm::DenseSet<Value> &sinkArgs) {
  OpBuilder builder(funcOp);
  // get generation
  int generationIndex = 1;
  if (auto attr = funcOp->getAttrOfType<IntegerAttr>("generationIndex"))
    generationIndex = (int)attr.getInt();

  // mark the function
  std::vector<Operation *> sourceOpVector;
  for (auto &block : funcOp) {
    for (auto &op : block) {
      if (sinkOps.contains(&op)) {
        op.setAttr("sink", builder.getBoolAttr(true));
        sourceOpVector.push_back(&op);
      }
    }
  }

  builder.setInsertionPoint(funcOp);
  auto copyFunc = funcOp.clone();

  auto sinkFuncName = builder.getStringAttr(copyFunc.getName() + "_sink");
  auto sourceFuncName = builder.getStringAttr(copyFunc.getName() + "_source");

  // get oplists
  llvm::DenseSet<Value> values;
  getValuesToTransfer(copyFunc, values);
  // create a generate for now, before terminate
  auto &lastBlock = copyFunc.getBlocks().back();
  builder.setInsertionPoint(lastBlock.getTerminator());
  // add generate. TODO(value)
  auto valuesVector = llvm::to_vector(values);
  createGenerate(builder, copyFunc.getLoc(), sinkFuncName, valuesVector);

  // clone the function
  builder.setInsertionPoint(funcOp);
  auto sourceFunc = copyFunc.clone();
  sourceFunc.setName(sourceFuncName);
  builder.insert(sourceFunc);

  // keep the original function by clone
  auto argTypes = llvm::map_to_vector(
      valuesVector, [](Value value) { return value.getType(); });
  auto sinkFunc = builder.create<ep2::FuncOp>(copyFunc.getLoc(), sinkFuncName, builder.getFunctionType(argTypes, {}));
  sinkFunc->setAttr("type", builder.getStringAttr("handler"));
  sinkFunc.getBody().takeBody(copyFunc.getBody());
  // makesure we erase the copyfunc
  copyFunc.erase();

  // process sink func.
  {
    SmallVector<Value> &argValues = valuesVector;
    auto locs = llvm::map_to_vector(argValues,
                                    [](Value value) { return value.getLoc(); });
    auto &entryBlock = sinkFunc.getBody().front();
    auto numOldArgs = entryBlock.getNumArguments();
    auto newArgs = entryBlock.addArguments(argTypes, locs);
    for (auto [arg, value] : llvm::zip(newArgs, argValues))
      value.replaceAllUsesWith(arg);
    // remove source Ops
    eraseOpsIf(sinkFunc, [](Operation *op) { return !inSink(op); });
    // remove original arguments
    entryBlock.eraseArguments(0, numOldArgs);
    // try add terminate
    tryAddTerminator(builder, sinkFunc);
    sinkFunc->setAttr("generationIndex", builder.getI32IntegerAttr(generationIndex * 2 + 1));
  }

  // process source func. (based on the clone)
  {
    eraseOpsIf(sourceFunc, [](Operation *op) { return inSink(op); });
    tryAddTerminator(builder, sourceFunc);
    sourceFunc->setAttr("generationIndex", builder.getI32IntegerAttr(generationIndex * 2));
  }

  for (auto func : {funcOp, sinkFunc, sourceFunc}) {
    removeEmptyBlocks(func);
    removeSinkAttr(func);
  }

  

  return std::make_pair(sourceFunc, sinkFunc);
}

} // namespace mlir
} // namespace ep2
