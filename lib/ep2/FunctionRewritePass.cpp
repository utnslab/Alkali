#include "mlir/IR/BuiltinDialect.h"

#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  using OpConversionPattern<ep2::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ep2::ConstantOp initOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // match
    auto fromType = initOp.getResult().getType();

    // rewrtie
    auto resType = typeConverter->convertType(fromType);
    auto value = adaptor.getValue();
    if (fromType.isa<ep2::AtomType>())
      value = rewriter.getI64IntegerAttr(0);

    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(initOp, resType, value);
    return success();
  }
};

struct CallPattern : public OpConversionPattern<ep2::CallOp> {
  using OpConversionPattern<ep2::CallOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ep2::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto resType = typeConverter->convertType(callOp.getResult().getType());
    llvm::SmallVector<Type> argTypes;
    typeConverter->convertTypes(callOp.getOperandTypes(), argTypes);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        callOp, LLVM::LLVMFunctionType::get(resType, argTypes),
        adaptor.getCallee(), adaptor.getOperands());
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
  std::string funcName;
  LLVM::LLVMFunctionType funcType;
  StorePattern(TypeConverter &converter, MLIRContext *context, std::string funcName, LLVM::LLVMFunctionType funcType)
      : OpConversionPattern<ep2::StoreOp>(converter, context),
        funcName(funcName), funcType(funcType) {}
  LogicalResult matchAndRewrite(ep2::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto refOp = dyn_cast<ep2::ContextRefOp>(storeOp.getOperand(0).getDefiningOp());
    auto contextId = rewriter.getRemappedValue(refOp.getOperand());

    // TODO name to offset
    auto offset = rewriter.create<LLVM::ConstantOp>(refOp->getLoc(),
        rewriter.getI32Type(), 0);
    auto voidPtr = rewriter.create<LLVM::BitcastOp>(refOp->getLoc(),
        rewriter.getType<LLVM::LLVMPointerType>(),
        adaptor.getValue());
    // convert the storeOp
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(storeOp,
        funcType, funcName, ValueRange{contextId, offset, voidPtr});

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
    if (isa<LLVM::LLVMPointerType>(newType))
      return rewriter.notifyMatchFailure(accessOp, "access now only support primitive type");

    auto ptr = rewriter.create<LLVM::GEPOp>(
        loc, rewriter.getType<LLVM::LLVMPointerType>(newType,0), adaptor.getInput(),
        ArrayRef<LLVM::GEPArg>{0, adaptor.getIndex()});
    auto load = rewriter.create<LLVM::LoadOp>(loc, newType, ptr);
    rewriter.replaceOp(accessOp, load);
    return success();
  }
};

struct ExtractPattern : public OpConversionPattern<ep2::ExtractOp> {
  std::string funcName;
  LLVM::LLVMFunctionType funcType;
  ExtractPattern(TypeConverter &converter, MLIRContext *context, std::string funcName, LLVM::LLVMFunctionType funcType)
      : OpConversionPattern<ep2::ExtractOp>(converter, context),
        funcName(funcName), funcType(funcType) {}

  LogicalResult
  matchAndRewrite(ep2::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = extractOp->getLoc();
    auto resType = extractOp.getResult().getType();
    if (!resType.isa<ep2::StructType>())
      return rewriter.notifyMatchFailure(extractOp, "Currently only support extract op on struct");
    
    // TODO: Get the size to transfer. 
    auto size = rewriter.create<LLVM::ConstantOp>(loc,
        rewriter.getI32Type(), 0);
    auto newType = typeConverter->convertType(resType);
    auto alloca = rewriter.create<LLVM::AllocaOp>(loc, newType,
      rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 1), 0);
    auto voidPtr = rewriter.create<LLVM::BitcastOp>(loc,
        rewriter.getType<LLVM::LLVMPointerType>(),
        alloca);
    auto callOp = rewriter.create<LLVM::CallOp>(loc,
        funcType, funcName, ValueRange{size, voidPtr, adaptor.getBuffer()});
    rewriter.replaceOp(extractOp, alloca);
    return success();
  }
};

// convert init op
struct InitPattern : public OpConversionPattern<ep2::InitOp> {
  using OpConversionPattern<ep2::InitOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ep2::InitOp initOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (initOp->getNumOperands() == 0) {
      auto resType = initOp.getType();
      if (!resType.isa<ep2::StructType>())
        return rewriter.notifyMatchFailure(initOp, "Currently only support init op on struct");
      auto newType = typeConverter->convertType(resType);
      auto zero = rewriter.create<LLVM::ConstantOp>(initOp->getLoc(),
                                                    rewriter.getI32Type(), 1);
      rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(initOp, newType, zero, 0);
      return success();
    } else {
      // TODO: remove result?
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          initOp, rewriter.getI32Type(), 0);
      return success();
    }
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
        rewriter.getType<LLVM::LLVMPointerType>(wrapperTypes[0], 0);
    auto inputStructPtr = newFuncOp.getArgument(0);

    auto contextId = rewriter.create<LLVM::LoadOp>(
        loc, rewriter.getI32Type(),
        rewriter.create<LLVM::GEPOp>(
            loc,
            rewriter.getType<LLVM::LLVMPointerType>(rewriter.getI32Type(), 0),
            inputStructPtr, ArrayRef<LLVM::GEPArg>{0, 1}));
    signatureConversion.remapInput(0, contextId);

    auto structPtr = rewriter.create<LLVM::GEPOp>(
        loc, inputWrapperType, inputStructPtr, ArrayRef<LLVM::GEPArg>{0, 2});

    auto sourceIdx = 1;
    for (size_t i = 0; i < wrapperTypes[0].getBody().size(); i++) {
      auto convertedType =
          typeConverter->convertType(wrapperTypes[0].getBody()[i]);
      auto elementPtrType =
          rewriter.getType<LLVM::LLVMPointerType>(convertedType, 0);
      // materialize block type
      auto param = rewriter.create<LLVM::LoadOp>(
          loc, rewriter.create<LLVM::GEPOp>(loc, elementPtrType, structPtr,
                                       ArrayRef<LLVM::GEPArg>{0, i}));
      signatureConversion.remapInput(i + sourceIdx, param);
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
    auto argType =
        LLVM::LLVMStructType::getIdentified(context, "__wrapper_arg");
    if (argType.getBody().size() == 0) {
      auto res = argType.setBody({rewriter.getI32Type(), rewriter.getI32Type(),
                                  rewriter.getType<LLVM::LLVMPointerType>()},
                                 false);
    }

    llvm::SmallVector<mlir::Type> types;
    for (int i = 0; i < num; i++)
      types.push_back(rewriter.getType<LLVM::LLVMPointerType>(argType, 0));
    return types;
  }
};

} // namespace

namespace { // util functiosn

void insertExternalDefination(ModuleOp moduleOp, std::vector<std::pair<std::string, LLVM::LLVMFunctionType>> &functions) {
  OpBuilder builder(moduleOp);

  // auto names = {"_ep2_rt_enqueue", "_ep2_rt_dequeue", "_ep2_rt_wait",
  // "Queue"};
  auto context = moduleOp.getContext();
  for (auto [name, functionType] : functions) {
    if (moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(name))
      continue;

    builder.setInsertionPointToStart(moduleOp.getBody());
    builder.create<LLVM::LLVMFuncOp>(moduleOp.getLoc(), name, functionType);
  }
}

} // namespace

void FunctionRewritePass::runOnOperation() {
  // install analysis
  LowerStructAnalysis &lowerStructAnalysis = getAnalysis<LowerStructAnalysis>();

  // install functions

  auto builder = OpBuilder(getOperation());
  auto i64Type = builder.getI64Type();
  auto i32Type = builder.getI32Type();
  auto voidPtrType = builder.getType<LLVM::LLVMPointerType>();
  std::vector<std::pair<std::string, LLVM::LLVMFunctionType>> globalFunctions = {
    {"Queue",
      LLVM::LLVMFunctionType::get(i64Type, {i64Type, i64Type, i64Type})},

    {"__ep2_rt_wr", 
    LLVM::LLVMFunctionType::get(builder.getType<LLVM::LLVMVoidType>(),
      {i32Type, i32Type, voidPtrType})},
    {"__ep2_rt_extract",
    LLVM::LLVMFunctionType::get(builder.getType<LLVM::LLVMVoidType>(),
      {i32Type, voidPtrType, voidPtrType})}
    };
  insertExternalDefination(getOperation(), globalFunctions);

  // Dialect Type converter
  TypeConverter typeConverter;
  // All other types are valid
  // conversion:: 1 to 1 type mapping
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([&](ep2::ContextType type) {
    return IntegerType::get(type.getContext(), 32);
  });
  typeConverter.addConversion([&](ep2::BufferType type) {
    return LLVM::LLVMPointerType::get(type.getContext(), 0);
  });
  typeConverter.addConversion([&](ep2::AtomType type) {
    return mlir::IntegerType::get(type.getContext(), 32);
  });
  typeConverter.addConversion([&](ep2::StructType type) {
    llvm::SmallVector<mlir::Type> types;
    for (auto element : type.getElementTypes())
      types.push_back(typeConverter.convertType(element));
    auto structType = LLVM::LLVMStructType::getLiteral(type.getContext(), types);
    return LLVM::LLVMPointerType::get(structType, 0);
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
  target.addLegalDialect<ep2::EP2Dialect, func::FuncDialect, LLVM::LLVMDialect,
                         BuiltinDialect>();
  target
      .addIllegalOp<ep2::ConstantOp, ep2::StoreOp, ep2::ContextRefOp,
                    ep2::CallOp, ep2::FuncOp, ep2::ReturnOp,
                    ep2::InitOp, ep2::StructAccessOp, ep2::ExtractOp>();

  // apply rules
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ConstPattern, CallPattern, ReturnPattern, ControllerPattern,
               ContextRefPattern, InitPattern, StructAccessPattern>(
      typeConverter, &getContext());
  patterns.add<FunctionPattern>(typeConverter, &getContext(),
                                lowerStructAnalysis);
  patterns.add<StorePattern>(typeConverter, &getContext(),
                                globalFunctions[1].first, globalFunctions[1].second);
  patterns.add<ExtractPattern>(typeConverter, &getContext(),
                                globalFunctions[2].first, globalFunctions[2].second);



  FrozenRewritePatternSet patternSet(std::move(patterns));

  if (failed(applyPartialConversion(getOperation(), target, patternSet)))
    signalPassFailure();
}

} // namespace ep2
} // namespace mlir
