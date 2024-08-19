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

void functionSplitter(ep2::FuncOp funcOp, llvm::DenseSet<Operation *> &sourceOps, llvm::DenseSet<Value> &sourceArguments) {
  OpBuilder builder(funcOp);

  auto newFuncOp = funcOp.clone();

  auto it = newFuncOp.getBlocks().rbegin();
  for (auto it = newFuncOp.getBlocks().rbegin(); it != newFuncOp.getBlocks().rend(); ++it) {
    auto &block = *it;
    if (!blockUsed(&block, sourceOps, sourceArguments)) {
      continue;
    }
  }

}

} // namespace mlir
} // namespace ep2
