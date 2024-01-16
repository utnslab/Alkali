#include "mlir/IR/BuiltinDialect.h"

#include "ep2/lang/Lexer.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace mlir {
namespace ep2 {

namespace SingleValueStructPatterns {

bool isTargetStruct (Type type) {
  auto structType = type.dyn_cast<StructType>();
  return structType && structType.getNumElementTypes() == 1 && !structType.getIsEvent();
}

struct Access : public OpConversionPattern<ep2::StructAccessOp> {
  using OpConversionPattern<ep2::StructAccessOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::StructAccessOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isTargetStruct(op.getInput().getType()))
      return rewriter.notifyMatchFailure(op.getLoc(), "not a struct type");
    
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

struct Update : public OpConversionPattern<ep2::StructUpdateOp> {
  using OpConversionPattern<ep2::StructUpdateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::StructUpdateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isTargetStruct(op.getType()))
      return rewriter.notifyMatchFailure(op.getLoc(), "not a struct type");
    
    rewriter.replaceOp(op, adaptor.getNewValue());
    return success();
  }
};

struct ExtractOffset : public OpConversionPattern<ep2::ExtractOffsetOp> {
  using OpConversionPattern<ep2::ExtractOffsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::ExtractOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isTargetStruct(op.getType()))
      return rewriter.notifyMatchFailure(op.getLoc(), "not a struct type");
    
    auto newType = op.getType().cast<StructType>().getElementTypes().front();
    rewriter.replaceOpWithNewOp<ExtractOffsetOp>(op, newType, adaptor.getBuffer(), op.getOffset());
    return success();
  }
};  

struct EmitOffset : public OpConversionPattern<ep2::EmitOffsetOp> {
  using OpConversionPattern<ep2::EmitOffsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::EmitOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isTargetStruct(op.getValue().getType()))
      return rewriter.notifyMatchFailure(op.getLoc(), "not a struct type");
    
    rewriter.updateRootInPlace(op, [&]() {
      op->setOperands(adaptor.getOperands());
    });
    return success();
  }
};

struct Init : public OpConversionPattern<ep2::InitOp> {
  using OpConversionPattern<ep2::InitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::InitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // init to struct with single element
    auto event = op.getType().dyn_cast<StructType>();
    if (!event || !event.getIsEvent())
      return rewriter.notifyMatchFailure(op.getLoc(), "not a struct type");
    
    auto newValues = llvm::to_vector(adaptor.getOperands());
    auto newOp = rewriter.replaceOpWithNewOp<InitOp>(op, event.getName(), newValues);
    newOp->setAttr("context_names", op->getAttr("context_names"));

    return success();
  }
};

struct Return : public OpConversionPattern<ep2::ReturnOp> {
  using OpConversionPattern<ep2::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op, [&]() {
      op->setOperands(adaptor.getOperands());
    });
    return success();
  }
};  

struct Func : public OpConversionPattern<ep2::FuncOp> {
  using OpConversionPattern<ep2::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(op, [&]() {
      auto res = rewriter.convertRegionTypes(&op.getBody(), *getTypeConverter());
      auto &block = *res;
      auto newType = op.cloneTypeWith(block->getArgumentTypes(), op.getResultTypes());
      op.setType(newType);
    });
    return success();
  }
};



} // local namespace for patterns

void CanonicalizePass::runOnOperation() {
  OpBuilder builder(&getContext());

  // Dialect Type converter
  TypeConverter typeConverter;
  // All other types are valid
  // conversion:: 1 to 1 type mapping
  typeConverter.addConversion([](Type type) { return type; });
  // convert struct with single element to the element type
  typeConverter.addConversion([&](StructType type) -> std::optional<Type> {
    if (!type.getIsEvent() && type.getNumElementTypes() == 1) {
      return type.getElementTypes().front();
    } else {
      auto newTypes = llvm::map_to_vector(type.getElementTypes(), [&](Type t) {
        return typeConverter.convertType(t);
      });
      return builder.getType<StructType>(type.getIsEvent(), newTypes, type.getName());
    }
    return std::nullopt;
  });
  // wildcard conversion to make system work. remove this later!
  typeConverter.addSourceMaterialization(
      [&](OpBuilder &builder, Type type, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return builder
            .create<UnrealizedConversionCastOp>(loc, TypeRange{type}, inputs)
            .getOutputs()[0];
      });
  typeConverter.addSourceMaterialization(
      [&](OpBuilder &builder, ep2::AtomType type, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() != 1 || !isa<IntegerType>(inputs[0].getType()))
          return std::nullopt;
        return builder
            .create<UnrealizedConversionCastOp>(loc, TypeRange{type}, inputs)
            .getOutputs()[0];
      });
  typeConverter.addTargetMaterialization(
      [&](OpBuilder &builder, Type type, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return builder
            .create<UnrealizedConversionCastOp>(loc, TypeRange{type}, inputs)
            .getOutputs()[0];
      });

  // Dialect conversion target
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect, LLVM::LLVMDialect, emitc::EmitCDialect,
                         BuiltinDialect>();
  target.addDynamicallyLegalDialect<ep2::EP2Dialect>([&](mlir::Operation* op){
    return typeConverter.isLegal(op);
  });
  target.addDynamicallyLegalOp<ep2::FuncOp>([&](ep2::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });

  // apply rules
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<SingleValueStructPatterns::Access,
               SingleValueStructPatterns::ExtractOffset,
               SingleValueStructPatterns::EmitOffset,
               SingleValueStructPatterns::Init,
               SingleValueStructPatterns::Return,
               SingleValueStructPatterns::Func,
               SingleValueStructPatterns::Update>(typeConverter, &getContext());
  FrozenRewritePatternSet patternSet(std::move(patterns));

  if (failed(applyPartialConversion(getOperation(), target, patternSet)))
    signalPassFailure();
}

} // namespace ep2
} // namespace mlir
