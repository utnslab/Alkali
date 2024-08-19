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

static bool blockUsed(Block *block, llvm::DenseSet<Operation *> &sourceOps, llvm::DenseSet<Value> &sourceArguments) {
  for (auto ba : block->getArguments()) {
    if (sourceArguments.contains(ba)) {
      return true;
    }
  }
  for (auto &op : *block) {
    if (sourceOps.contains(&op)) {
      return true;
    }
  }
  return false;
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

void functionSplitter(ep2::FuncOp funcOp, llvm::DenseSet<Operation *> &sinkOps, llvm::DenseSet<Value> &sinkArgs) {
  OpBuilder builder(funcOp);

  // mark the function
  std::vector<Operation *> sourceOpVector;
  for (auto &block : funcOp) {
    // TODO: attr?
    // for (auto &arg : block.getArguments()) {
    //   if (sourceArguments.contains(arg)) {
    //     arg.getArgNumber();
    //   }
    // }
    for (auto &op : block) {
      if (sinkOps.contains(&op)) {
        op.setAttr("sink", builder.getBoolAttr(true));
        sourceOpVector.push_back(&op);
      }
    }
  }

  auto sinkFuncName = builder.getStringAttr(funcOp.getName() + "_sink");
  auto sourceFuncName = builder.getStringAttr(funcOp.getName() + "_source");

  // get oplists
  llvm::DenseSet<Value> values;
  getValuesToTransfer(funcOp, values);
  // create a generate for now, before terminate
  auto &lastBlock = funcOp.getBlocks().back();
  builder.setInsertionPoint(lastBlock.getTerminator());
  // add generate. TODO(value)
  auto valuesVector = llvm::to_vector(values);
  createGenerate(builder, funcOp.getLoc(), sinkFuncName, valuesVector);

  // clone the function
  builder.setInsertionPoint(funcOp);
  auto sourceFunc = funcOp.clone();
  sourceFunc.setName(sourceFuncName);
  builder.insert(sourceFunc);

  auto argTypes = llvm::map_to_vector(
      valuesVector, [](Value value) { return value.getType(); });
  auto sinkFunc = builder.create<ep2::FuncOp>(funcOp.getLoc(), sinkFuncName, builder.getFunctionType(argTypes, {}));
  sinkFunc->setAttr("type", builder.getStringAttr("handler"));
  sinkFunc.getBody().takeBody(funcOp.getBody());

  funcOp.erase();

  // process sink func.
  {
    auto locs = llvm::map_to_vector(valuesVector,
                                    [](Value value) { return value.getLoc(); });
    auto &entryBlock = sinkFunc.getBody().front();
    auto numOldArgs = entryBlock.getNumArguments();
    auto newArgs = entryBlock.addArguments(argTypes, locs);
    for (auto [arg, value] : llvm::zip(newArgs, valuesVector))
      value.replaceAllUsesWith(arg);
    // remove source Ops
    eraseOpsIf(sinkFunc, [](Operation *op) { return !inSink(op); });
    // remove original arguments
    entryBlock.eraseArguments(0, numOldArgs);
    // try add terminate
    tryAddTerminator(builder, sinkFunc);
  }

  // process source func. (based on the clone)
  {
    eraseOpsIf(sourceFunc, [](Operation *op) { return inSink(op); });
    tryAddTerminator(builder, sourceFunc);
  }
}

} // namespace mlir
} // namespace ep2
