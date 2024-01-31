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

  struct ContextRefTypeRewrite : public OpRewritePattern<ContextRefOp> {
    AnalysisManager &am;
    ContextRefTypeRewrite(MLIRContext *context, AnalysisManager &am)
        : OpRewritePattern<ContextRefOp>(context), am(am) {}

    LogicalResult matchAndRewrite(ContextRefOp refOp,
                                  PatternRewriter &rewriter) const final {

      if (!refOp.getType().getValueType().isa<AnyType>())
        return failure();

      // First, we try to see if type is infered from other handler
      auto &contextAnalysis = am.getAnalysis<ContextBufferizationAnalysis>();
      auto type = contextAnalysis.getContextType(refOp->getParentOfType<FuncOp>(),
                                     refOp.getName()).second;

      if (type && !type.isa<AnyType>()) {
        auto newRefType = rewriter.getType<ContextRefType>(type);
        rewriter.replaceOpWithNewOp<ContextRefOp>(refOp, newRefType, refOp.getName(), refOp.getContext());
        return success();
      }

      // we try to infer from load and store ops
      std::vector<mlir::Type> types;
      for (auto user : refOp->getUsers()) {
        if (LoadOp loadOp = dyn_cast<LoadOp>(user))
          types.push_back(loadOp.getType());
        else if (StoreOp storeOp = dyn_cast<StoreOp>(user))
          types.push_back(storeOp.getValue().getType());
      }

      if (types.empty())
        return failure();

      // finding a common value
      auto newType = unifyTypes(rewriter.getContext(), types);
      if (!newType || newType->isa<AnyType>())
        return failure();

      auto newRefType = rewriter.getType<ContextRefType>(*newType);
      rewriter.replaceOpWithNewOp<ContextRefOp>(refOp, newRefType, refOp.getName(), refOp.getContext());

      return success();
    }

  };

  struct LoadTypeRewrite : public OpRewritePattern<LoadOp> {
    using OpRewritePattern<LoadOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(LoadOp loadOp,
                                  PatternRewriter &rewriter) const final {
      auto inType = loadOp.getRef().getType().getValueType();
      auto outType = loadOp.getType();
      if (inType.isa<AnyType>() || !outType.isa<AnyType>())
        return failure();

      rewriter.replaceOpWithNewOp<LoadOp>(loadOp, inType, loadOp.getRef());
      return success();
    }
  };

} // namespace Patterns



// Conversion Pass
void ContextTypeInferencePass::runOnOperation() {
    auto module = getOperation();

    AnalysisManager am = getAnalysisManager();
    auto bufferAnalysis = am.getAnalysis<ContextBufferizationAnalysis>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ContextRefTypeRewrite>(&getContext(), am);
    patterns.add<LoadTypeRewrite>(&getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    while (true) {
      bool changed = false;
      if (failed(applyPatternsAndFoldGreedily(module, frozenPatterns,
                                              GreedyRewriteConfig(), &changed)))
        signalPassFailure();
      if (!changed)
        break;
      bufferAnalysis.invalidate();
    }
}

}
}
