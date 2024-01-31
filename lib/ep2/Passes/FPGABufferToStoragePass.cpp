#include "mlir/IR/BuiltinDialect.h"

#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "ep2/dialect/Analysis/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

#include <optional>

namespace mlir {
namespace ep2 {

namespace {
struct InitPattern : public OpConversionPattern<InitOp> {
  using OpConversionPattern<InitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InitOp initOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto convertedType = getTypeConverter()->convertType(initOp.getType());
    rewriter.replaceOpWithNewOp<InitOp>(initOp, convertedType);
    return success();
  }
};

struct LookupOpPattern : public OpConversionPattern<LookupOp> {
  using OpConversionPattern<LookupOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LookupOp lookupOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<BufferPoolFetchOp>(
        lookupOp, rewriter.getType<BufferType>(), adaptor.getTable(), adaptor.getKey());
    return success();
  }
};

struct UpdateOpPattern : public OpConversionPattern<UpdateOp> {
  using OpConversionPattern<UpdateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UpdateOp updateOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    
    // If we get an init op, need to alloc
    rewriter.replaceOpWithNewOp<BufferPoolCommitOp>(
        updateOp, adaptor.getTable(), adaptor.getKey(), updateOp.getValue());

    return success();
  }


};

} // local namespace

void FPGABufferToStoragePass::runOnOperation() {
  auto moduleOp = getOperation();
  OpBuilder builder(moduleOp.getContext());

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([&](TableType type) -> std::optional<Type> {
    if (type.getValueType().isa<BufferType>()) {
      return builder.getType<BufferPoolType>();
    }
    return std::nullopt;
  });

  mlir::ConversionTarget target(getContext());
  target.addDynamicallyLegalDialect<ep2::EP2Dialect>([&](mlir::Operation* op){
    return typeConverter.isLegal(op);
  });
  
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<InitPattern, LookupOpPattern, UpdateOpPattern>(typeConverter, &getContext());
  FrozenRewritePatternSet patternSet(std::move(patterns));

  if (failed(applyPartialConversion(getOperation(), target, patternSet)))
    signalPassFailure();
}

} // namespace ep2
} // namespace mlir