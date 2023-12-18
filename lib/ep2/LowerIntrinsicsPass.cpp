
#include "mlir/IR/BuiltinDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cassert>

using namespace mlir;

namespace mlir {
namespace ep2 {

struct LowerMemcpyPattern :
  public OpRewritePattern<emitc::CallOp> {

  LowerMemcpyPattern(MLIRContext* context) : OpRewritePattern<emitc::CallOp>(context, /*benefit=*/ 1) {}

  LogicalResult matchAndRewrite(emitc::CallOp op, PatternRewriter& rewriter) const override {
    if (op.getCallee() != "__ep2_intrin_memcpy") {
      return failure();
    }

    auto loc = op->getLoc();
    auto memcpySize = rewriter.create<emitc::ConstantOp>(loc, rewriter.getI32Type(), (*op.getArgs())[0]);

    llvm::SmallVector<Type> resTypes = {};
    mlir::ArrayAttr args;
    mlir::ArrayAttr templ_args;
    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, resTypes, rewriter.getStringAttr("bulk_memcpy"), args, templ_args, ValueRange{op->getOperands()[0], op->getOperands()[1], memcpySize});

    return success();
  }
};

void LowerIntrinsicsPass::runOnOperation() {
  auto module = getOperation();
  
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<LowerMemcpyPattern>(&getContext());
  auto res = applyPatternsAndFoldGreedily(module, std::move(patterns));
  assert(res.succeeded());
}

}
}
