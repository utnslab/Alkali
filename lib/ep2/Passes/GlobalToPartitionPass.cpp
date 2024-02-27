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

GlobalOp findGlobalOf(ModuleOp module, StringRef name) {
    for (auto global : module.getOps<GlobalOp>()) {
        if (global.getName() == name)
            return global;
    }
    return nullptr;
}

class GlobalImportPattern : public OpRewritePattern<GlobalImportOp> {
  using OpRewritePattern<GlobalImportOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GlobalImportOp refOp,
                                PatternRewriter &rewriter) const final {
    auto globalName = refOp.getName();
    auto globalAttrName = "instances_" + globalName.str();

    if (!refOp->getParentOfType<FuncOp>()->hasAttr(globalAttrName))
      return rewriter.notifyMatchFailure(refOp, "the global variable is not partitioned");
    
    auto global = findGlobalOf(refOp->getParentOfType<ModuleOp>(), globalName);
    assert(global != nullptr);

    rewriter.replaceOpWithNewOp<InitOp>(refOp, global.getType());
  }
};
} // rewrite pattern

void GlobalToPartitionPass::runOnOperation() {
  auto moduleOp = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<GlobalImportPattern>(&getContext());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  if (failed(applyPatternsAndFoldGreedily(moduleOp, frozenPatterns)))
    return signalPassFailure();

  // remove global op if possible if necessary
  for (auto global : moduleOp.getOps<GlobalOp>()) {
    bool hasUse = false;
    moduleOp->walk([&](GlobalImportOp refOp) {
      if (refOp.getName() == global.getName())
        hasUse = true;
    });

    if (!hasUse)
      global.erase();
  }
}

} // namespace ep2
} // namespace mlir