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

/*
EP2 canonicalize pass demotes operations producing/consuming 1-element
structs to primitive accesses, and uses that information in dead field
optimization. We reverse this, to enable some of our optimizations.
*/

namespace mlir {
namespace ep2 {

static int uniqTypeGen = 0;

struct ExtractOffsetPattern : public OpConversionPattern<ep2::ExtractOffsetOp> {
  using OpConversionPattern<ep2::ExtractOffsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::ExtractOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (isa<ep2::StructType>(op.getOutput().getType())) {
      return failure();
    }

    auto newType = ep2::StructType::get(getContext(), false, llvm::SmallVector<mlir::Type>{op.getOutput().getType()}, "repack_type_" + std::to_string(uniqTypeGen++));
    auto extr = rewriter.create<ExtractOffsetOp>(op->getLoc(), newType, adaptor.getBuffer(), op.getOffset());

    rewriter.replaceOpWithNewOp<StructAccessOp>(op, extr->getResult(0), 0);
    return success();
  }
};  

struct EmitOffsetPattern : public OpConversionPattern<ep2::EmitOffsetOp> {
  using OpConversionPattern<ep2::EmitOffsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::EmitOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (isa<ep2::StructType>(op.getValue().getType()) ||
        isa<ep2::BufferType>(op.getValue().getType())) {
      return failure();
    }

    auto newType = ep2::StructType::get(getContext(), false, llvm::SmallVector<mlir::Type>{op.getValue().getType()}, "repack_type_" + std::to_string(uniqTypeGen++));
    auto val = rewriter.create<InitOp>(op->getLoc(), TypeRange{newType}, ValueRange{adaptor.getValue()});
    rewriter.replaceOpWithNewOp<EmitOffsetOp>(op, adaptor.getBuffer(), val->getResult(0), op.getOffset());
    return success();
  }
};  

void RepackStructTypesPass::runOnOperation() {
  OpBuilder builder(&getContext());

  // Dialect Type converter
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });

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
    if (isa<ExtractOffsetOp>(op) && 
        !isa<ep2::StructType>(cast<ep2::ExtractOffsetOp>(op).getOutput().getType())) {
      return false;
    }
    if (isa<EmitOffsetOp>(op) &&
        !isa<ep2::StructType>(cast<ep2::EmitOffsetOp>(op).getValue().getType()) &&
        !isa<ep2::BufferType>(cast<ep2::EmitOffsetOp>(op).getValue().getType())) {
      return false;
    }
    return true;
  });

  // apply rules
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ExtractOffsetPattern, EmitOffsetPattern>(typeConverter, &getContext());
  FrozenRewritePatternSet patternSet(std::move(patterns));

  if (failed(applyPartialConversion(getOperation(), target, patternSet)))
    signalPassFailure();
}

} // namespace ep2
} // namespace mlir
