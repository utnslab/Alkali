#include "mlir/IR/BuiltinDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace ep2 {

//===----------------------------------------------------------------------===//
// Load & Store Ops
//===----------------------------------------------------------------------===//

namespace {
  std::optional<mlir::Type> unifyTypes(MLIRContext *context, std::vector<mlir::Type> types) {
    if (types.size() == 1)
      return types[0];
    
    // only works for integer types
    if (std::all_of(types.begin(), types.end(),
                    [](mlir::Type t) { return t.isIntOrFloat(); })) {
      auto max = std::max_element(types.begin(), types.end(), [](mlir::Type a, mlir::Type b) {
        return a.getIntOrFloatBitWidth() < b.getIntOrFloatBitWidth();
      });
      return mlir::IntegerType::get(context, (*max).getIntOrFloatBitWidth());
    }
    return std::nullopt;
  }

  struct LoadStoreTypeRewrite : public OpRewritePattern<ContextRefOp> {
    using OpRewritePattern<ContextRefOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ContextRefOp refOp,
                                  PatternRewriter &rewriter) const final {
      llvm::errs() << "Is refOp\n";
      if (!refOp.getType().getValueType().isa<AnyType>())
        return failure();

      // we get a undecided reference
      std::vector<mlir::Type> types;
      for (auto user : refOp->getUsers()) {
        if (LoadOp loadOp = dyn_cast<LoadOp>(user))
          types.push_back(loadOp.getType());
        else if (StoreOp storeOp = dyn_cast<StoreOp>(user))
          types.push_back(storeOp.getOperand(1).getType());
      }

      if (types.empty())
        return failure();
      
      // finding a common value
      auto newType = unifyTypes(rewriter.getContext(), types);
      if (!newType)
        return failure();
      auto newRefType = rewriter.getType<ContextRefType>(*newType);
      rewriter.replaceOpWithNewOp<ContextRefOp>(refOp, newRefType, refOp.getName(), refOp.getContext());
      return success();
    }
  };
} // namespace Patterns

// Conversion Pass
void ContextTypeInferencePass::runOnOperation() {
    auto module = getOperation();

    AnalysisManager am = getAnalysisManager();

    RewritePatternSet patterns(&getContext());
    patterns.add<LoadStoreTypeRewrite>(&getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    llvm::errs() << "Running pattern\n";

    if (failed(applyPatternsAndFoldGreedily(module, frozenPatterns)))
      signalPassFailure();
}

}
}