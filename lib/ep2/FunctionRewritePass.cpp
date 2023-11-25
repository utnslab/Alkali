#include "mlir/IR/BuiltinDialect.h"

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

namespace mlir {
namespace ep2 {

namespace {

// convert const op
struct ConstPattern : public OpConversionPattern<ep2::ConstantOp> {
  AtomAnalysis &analyzer;
  ConstPattern(TypeConverter &converter, MLIRContext *context,
                  AtomAnalysis &analyzer)
      : OpConversionPattern<ep2::ConstantOp>(converter, context),
        analyzer(analyzer) {}

  LogicalResult
  matchAndRewrite(ep2::ConstantOp initOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // match
    auto fromType = initOp.getResult().getType();

    // rewrtie
    auto resType = typeConverter->convertType(fromType);
    auto value = adaptor.getValue();
    if (fromType.isa<ep2::AtomType>()) {
      size_t v = analyzer.atomToNum[initOp.getValue().cast<mlir::StringAttr>().getValue()];
      value = rewriter.getI32IntegerAttr({v});
    }

    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(initOp, resType, value);
    return success();
  }
};

struct CallPattern : public OpConversionPattern<ep2::CallOp> {
  using OpConversionPattern<ep2::CallOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ep2::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Type> resTypes = {typeConverter->convertType(callOp.getResult().getType())};
    mlir::ArrayAttr args;
    mlir::ArrayAttr templ_args;
    rewriter.replaceOpWithNewOp<emitc::CallOp>(callOp, resTypes, rewriter.getStringAttr(adaptor.getCallee()), args, templ_args, adaptor.getOperands());
    return success();
  }
};

struct ContextRefPattern : public OpConversionPattern<ep2::ContextRefOp> {
  using OpConversionPattern<ep2::ContextRefOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ep2::ContextRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return success();
  }
};

struct StorePattern : public OpConversionPattern<ep2::StoreOp> {

  ContextAnalysis &analyzer;
  StorePattern(TypeConverter &converter, MLIRContext *context,
                  ContextAnalysis &analyzer)
      : OpConversionPattern<ep2::StoreOp>(converter, context),
        analyzer(analyzer) {}

  LogicalResult matchAndRewrite(ep2::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto refOp = dyn_cast<ep2::ContextRefOp>(storeOp.getOperand(0).getDefiningOp());
    auto contextId = rewriter.getRemappedValue(refOp.getOperand());

    ContextAnalysis::ContextField place = analyzer.disj_contexts[analyzer.disj_groups[storeOp]][refOp.getName()];
    llvm::SmallVector<Type> resTypes = {};
    mlir::ArrayAttr args = rewriter.getI32ArrayAttr({place.offs});
    mlir::ArrayAttr templ_args;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(storeOp, resTypes, rewriter.getStringAttr("__ep2_rt_wr"), args, templ_args, ValueRange{contextId, adaptor.getValue()});

    return success();
  }
};

struct ReturnPattern : public OpConversionPattern<ep2::ReturnOp> {
  using OpConversionPattern<ep2::ReturnOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ep2::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isa<func::FuncOp>(returnOp->getParentOp()))
      return rewriter.notifyMatchFailure(returnOp,
                                         "Not a valid return in func::FuncOp");

    // rewrite when not return value
    if (returnOp->getNumOperands() == 0) {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp);
      return success();
    } else if (returnOp->getNumOperands() == 1 &&
               returnOp.getOperand(0).getType().isa<ep2::StructType>()) {
      // return an event. change to function call
      rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp);
      return success();
    }

    return rewriter.notifyMatchFailure(returnOp, "Not a valid return");
  }
};

// how will this execute?
struct GeneratePattern : public ConversionPattern {
  using ConversionPattern::ConversionPattern;
  LogicalResult match(Operation *op) const final {
    if (!isa<ep2::InitOp>(op))
      return failure();
    if (op->getNumResults() != 1 ||
        !isa<ep2::StructType>(op->getResultTypes()[0]))
      return failure();

    for (auto user : op->getUsers()) {
      if (!isa<ep2::ReturnOp>(user))
        return failure();
    }
    return success();
  }

  void rewrite(Operation *op, ArrayRef<Value> name,
               ConversionPatternRewriter &rewriter) const final {
    auto initOp = dyn_cast<ep2::InitOp>(op);
    auto loc = op->getLoc();

    // replace users
    for (auto user : op->getUsers()) {
      rewriter.replaceOpWithNewOp<func::ReturnOp>(user);
    }
  }
};

struct StructAccessPattern : public OpConversionPattern<ep2::StructAccessOp> {
  using OpConversionPattern<ep2::StructAccessOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ep2::StructAccessOp accessOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = accessOp->getLoc();

    auto resType = accessOp.getResult().getType();
    auto newType = typeConverter->convertType(resType);
    if (isa<emitc::PointerType>(newType)) {
      return rewriter.notifyMatchFailure(accessOp, "access now only support primitive type");
    }

    llvm::SmallVector<Type> resTypes = {typeConverter->convertType(accessOp.getResult().getType())};
    mlir::ArrayAttr args = rewriter.getI32ArrayAttr({(uint32_t) accessOp.getIndex()});
    mlir::ArrayAttr templ_args;
    auto load = rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_intrin_struct_access"), args, templ_args, ValueRange{adaptor.getOperands()[0]});
    rewriter.replaceOp(accessOp, load);
    return success();
  }
};

struct ExtractPattern : public OpConversionPattern<ep2::ExtractOp> {
  using OpConversionPattern<ep2::ExtractOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = extractOp->getLoc();
    auto resType = extractOp.getResult().getType();
    if (!resType.isa<ep2::StructType>())
      return rewriter.notifyMatchFailure(extractOp, "Currently only support extract op on struct");

    auto newType = typeConverter->convertType(resType);
    llvm::SmallVector<Type> resTypes = {newType};
    // TODO: Get the size to transfer. 
    mlir::ArrayAttr args;
    mlir::ArrayAttr templ_args;
    auto alloc = rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_rt_alloc_buf"), args, templ_args, ValueRange{});
    rewriter.replaceOp(extractOp, alloc);
    
    llvm::SmallVector<Type> resTypes2 = {};
    // TODO: Get the size to transfer. 
    mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({0});
    mlir::ArrayAttr templ_args2;
    rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_rt_extract"), args2, templ_args2, ValueRange{alloc.getResult(0), adaptor.getBuffer()});
    return success();
  }
};

// convert init op
struct InitPattern : public OpConversionPattern<ep2::InitOp> {
  using OpConversionPattern<ep2::InitOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ep2::InitOp initOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = initOp->getLoc();
    auto resType = initOp.getType();
    if (!resType.isa<ep2::StructType>())
      return rewriter.notifyMatchFailure(initOp, "Currently only support init op on struct");
    auto newType = typeConverter->convertType(resType);

    llvm::SmallVector<Type> resTypes = {newType};
    // TODO: Get the size to transfer. 
    // Smh lower init to separate the allocation decision from filling out the struct.
    mlir::ArrayAttr args = rewriter.getI32ArrayAttr({0});
    mlir::ArrayAttr templ_args;
    auto alloc = rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_rt_alloc_struct"), args, templ_args, ValueRange{});
    rewriter.replaceOp(initOp, alloc);

    unsigned p = 0;
    for (const auto& opd : adaptor.getOperands()) {
      llvm::SmallVector<Type> resTypes2 = {};
      mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({p});
      mlir::ArrayAttr templ_args2;
      rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_struct_write"), args2, templ_args2, ValueRange{opd, alloc.getResult(0)});
      p += 1;
    }
    return success();
  }
};

struct ControllerPattern : public OpConversionPattern<ep2::FuncOp> {
  using OpConversionPattern<ep2::FuncOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ep2::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (funcOp->getAttr("type").cast<StringAttr>().getValue() != "controller")
      return failure();

    auto newFuncOp = rewriter.create<func::FuncOp>(
        funcOp->getLoc(), funcOp.getName(), adaptor.getFunctionType());
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    rewriter.eraseOp(funcOp);
    return success();
  }
};

// convert function
struct FunctionPattern : public OpConversionPattern<ep2::FuncOp> {

  LowerStructAnalysis &analyzer;
  FunctionPattern(TypeConverter &converter, MLIRContext *context,
                  LowerStructAnalysis &analyzer)
      : OpConversionPattern<ep2::FuncOp>(converter, context),
        analyzer(analyzer) {}

  LogicalResult
  matchAndRewrite(ep2::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = funcOp->getLoc();
    if (funcOp->getAttr("type").cast<StringAttr>().getValue() != "handler")
      return rewriter.notifyMatchFailure(funcOp, "Not a handler");

    // rewrite a function call.
    auto wrapperTypes = analyzer.getWrapperTypes(funcOp);
    if (wrapperTypes.size() == 0) {
      funcOp.emitError("Cannot rewrite with valid input");
      return failure();
    }
    auto newArgTypes = getArgumentTypes(rewriter, wrapperTypes.size());
    TypeConverter::SignatureConversion signatureConversion(
        funcOp.getNumArguments());

    auto newFuncOp = rewriter.create<func::FuncOp>(
        loc, std::string{"__event_" + funcOp.getName().str()},
        rewriter.getFunctionType(TypeRange(newArgTypes), TypeRange{}));
    
    auto entryBlock = newFuncOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);

    // construct body and replace parameter
    auto inputWrapperType =
        rewriter.getType<emitc::PointerType>(rewriter.getType<emitc::OpaqueType>("struct in_t"));
    auto inputStructPtr = newFuncOp.getArgument(0);

    llvm::SmallVector<Type> resTypes = {rewriter.getI32Type()};
    mlir::ArrayAttr args = rewriter.getI32ArrayAttr({0});
    mlir::ArrayAttr templ_args;
    auto contextId = rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_intrin_struct_access"), args, templ_args, ValueRange{inputStructPtr});
    signatureConversion.remapInput(0, contextId.getResult(0));

    llvm::SmallVector<Type> resTypes3 = {inputWrapperType};
    mlir::ArrayAttr args3 = rewriter.getI32ArrayAttr({1});
    mlir::ArrayAttr templ_args3;
    auto structPtr = rewriter.create<emitc::CallOp>(loc, resTypes3, rewriter.getStringAttr("__ep2_intrin_struct_access"), args3, templ_args3, ValueRange{inputStructPtr});

    auto sourceIdx = 1;
    for (size_t i = 0; i < wrapperTypes[0].getBody().size(); i++) {
      auto convertedType =
          typeConverter->convertType(wrapperTypes[0].getBody()[i]);
      auto elementPtrType =
          rewriter.getType<emitc::PointerType>(convertedType);
      // materialize block type
      llvm::SmallVector<Type> resTypes2 = {convertedType};
      mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({i});
      mlir::ArrayAttr templ_args2;
      auto param = rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_struct_access"), args2, templ_args2, ValueRange{structPtr.getResult(0)});
      signatureConversion.remapInput(i + sourceIdx, param.getResult(0));
    }

    // change the function body
    auto res = rewriter.convertRegionTypes(&funcOp.getBody(), *typeConverter,
                                           &signatureConversion);
    if (failed(res))
      return failure();
    rewriter.mergeBlocks(*res, entryBlock);


    // remove original func
    rewriter.eraseOp(funcOp);
    return success();
  }

  llvm::SmallVector<mlir::Type> getArgumentTypes(PatternRewriter &rewriter,
                                                 int num) const {
    auto context = rewriter.getContext();

    // insert if not done
    auto argType = rewriter.getType<emitc::OpaqueType>("struct __wrapper_arg");
    llvm::SmallVector<mlir::Type> types;
    for (int i = 0; i < num; i++)
      types.push_back(rewriter.getType<emitc::PointerType>(argType));
    return types;
  }
};

} // namespace

void FunctionRewritePass::runOnOperation() {
  // install analysis
  LowerStructAnalysis &lowerStructAnalysis = getAnalysis<LowerStructAnalysis>();
  ContextAnalysis &contextAnalysis = getAnalysis<ContextAnalysis>();
  AtomAnalysis &atomAnalysis = getAnalysis<AtomAnalysis>();

  // install functions
  auto builder = OpBuilder(getOperation());

  // Dialect Type converter
  TypeConverter typeConverter;
  // All other types are valid
  // conversion:: 1 to 1 type mapping
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([&](ep2::ContextType type) {
    return IntegerType::get(type.getContext(), 32);
  });
  typeConverter.addConversion([&](ep2::BufferType type) {
    return emitc::PointerType::get(type.getContext(), builder.getType<emitc::OpaqueType>("void"));
  });
  typeConverter.addConversion([&](ep2::AtomType type) {
    return mlir::IntegerType::get(type.getContext(), 32);
  });
  typeConverter.addConversion([&](ep2::StructType type) {
    return emitc::PointerType::get(type.getContext(), builder.getType<emitc::OpaqueType>(std::string("struct ") + type.getName().str()));
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
  target.addLegalDialect<ep2::EP2Dialect, func::FuncDialect, LLVM::LLVMDialect, emitc::EmitCDialect,
                         BuiltinDialect>();

  // TODO add a pass to handle loads.
  target
      .addIllegalOp<ep2::ConstantOp, ep2::StoreOp, ep2::ContextRefOp,
                    ep2::CallOp, ep2::FuncOp, ep2::ReturnOp,
                    ep2::InitOp, ep2::StructAccessOp, ep2::ExtractOp>();

  // apply rules
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<CallPattern, ReturnPattern, ControllerPattern,
               ContextRefPattern, InitPattern, StructAccessPattern, ExtractPattern>(
      typeConverter, &getContext());
  patterns.add<ConstPattern>(typeConverter, &getContext(),
                                atomAnalysis);
  patterns.add<StorePattern>(typeConverter, &getContext(),
                                contextAnalysis);
  patterns.add<FunctionPattern>(typeConverter, &getContext(),
                                lowerStructAnalysis);

  FrozenRewritePatternSet patternSet(std::move(patterns));

  if (failed(applyPartialConversion(getOperation(), target, patternSet)))
    signalPassFailure();
}

} // namespace ep2
} // namespace mlir
