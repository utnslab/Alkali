#include "mlir/IR/BuiltinDialect.h"

#include "ep2/lang/Lexer.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
namespace ep2 {

namespace {

// The struct in EP2 is *never* nested. It is always a flat struct
bool isStructPointer(Type type) {
  if (auto ptr = type.dyn_cast<LLVM::LLVMPointerType>()) {
    // We need to exclude void *. void * is a value rather than a pointer
    if (ptr.getElementType() == nullptr)
      return false;
    return ptr.getElementType().isa<LLVM::LLVMStructType>();
  }
  return false;
}

int getTypeSize(Type type) {
  DataLayout layout{};
  return layout.getTypeSize(type);
}

void emitStore(OpBuilder &builder, Value value, Value ptr) {
  assert(isa<LLVM::LLVMPointerType>(ptr.getType()) && "Not a pointer");
  auto ptrType = dyn_cast<LLVM::LLVMPointerType>(ptr.getType());
  assert(ptrType.getElementType() && "Pointer to void in store");
  if (value.getType() == ptr.getType()) {
    // Both are pointers. Use a memcpy
    auto len = builder.getI32IntegerAttr(getTypeSize(ptrType.getElementType()));
    builder.create<LLVM::MemcpyInlineOp>(value.getLoc(), ptr, value, len, false);
  } else // assign a value to a pointer
    builder.create<LLVM::StoreOp>(value.getLoc(), value, ptr);
}

// helper functions
// TODO: is identified?
SmallVector<Type> convertTypes(OpBuilder &builder,
                                      const TypeConverter &converter,
                                      ArrayRef<Type> ep2Types){
  auto types = llvm::map_to_vector(
      ep2Types, [&](Type t) { return converter.convertType(t); });
  return llvm::map_to_vector(
      types, [](Type t) {
        if (isStructPointer(t))
          return dyn_cast<LLVM::LLVMPointerType>(t).getElementType();
        else
          return t;
      });
}

LLVM::GEPOp emitGEP(OpBuilder &builder, Type ret, Value ptr, int index) {
  auto resType = isStructPointer(ret) ? ret : LLVM::LLVMPointerType::get(ret);
  return builder.create<LLVM::GEPOp>(ptr.getLoc(), resType, ptr, ArrayRef<LLVM::GEPArg>{0, index});
}

LLVM::BitcastOp castToVoid(OpBuilder &builder, Value val) {
  auto voidPtr = builder.getType<LLVM::LLVMPointerType>();
  return builder.create<LLVM::BitcastOp>(val.getLoc(), voidPtr, val);
}

// Traits
template<typename T>
struct ConversionToCallPattern : public OpConversionPattern<T> {
  LowerLLVMPass::TableT &apiFunctions;
  ConversionToCallPattern(TypeConverter &converter, MLIRContext *context, LowerLLVMPass::TableT &apiFunctions)
      : OpConversionPattern<T>(converter, context), apiFunctions(apiFunctions) {}
};

template<typename T>
struct ConversionWithAnalysis : public OpConversionPattern<T> {
  AnalysisManager &am;
  ConversionWithAnalysis(TypeConverter &converter, MLIRContext *context, AnalysisManager &am)
      : OpConversionPattern<T>(converter, context), am(am) {}
};

// Rule sets
struct StructAccessOpPattern : public OpConversionPattern<ep2::StructAccessOp> {
  using OpConversionPattern<ep2::StructAccessOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::StructAccessOp structAccessOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto resType = getTypeConverter()->convertType(structAccessOp.getType());
    auto gepOp = emitGEP(rewriter, resType, adaptor.getInput(),
                         static_cast<int>(structAccessOp.getIndex()));
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(structAccessOp, gepOp);

    return success();
  }
};

struct StructUpdateOpPattern : public OpConversionPattern<ep2::StructUpdateOp> {
  using OpConversionPattern<ep2::StructUpdateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::StructUpdateOp updateOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto resType = adaptor.getNewValue().getType();
    auto gepOp = emitGEP(rewriter, resType, adaptor.getInput(),
                         static_cast<int>(updateOp.getIndex()));
    // TODO(zhiyuang): if its a pointer, we only need GEP (but without load).
    // Do we have any nested struct?
    rewriter.create<LLVM::StoreOp>(updateOp.getLoc(), adaptor.getNewValue(), gepOp);
    rewriter.replaceOp(updateOp, ValueRange{adaptor.getInput()});

    return success();
  }
};

LLVM::ConstantOp emitGetSize(OpBuilder &builder, Type type) {
  DataLayout layout{};
  if (auto ptr = type.dyn_cast<LLVM::LLVMPointerType>()) {
    // Could be a pointer to a struct
    assert(ptr.getElementType() && "Pointer to void used in emit/extract?");
    auto size = builder.create<LLVM::ConstantOp>(
        builder.getUnknownLoc(), builder.getIntegerType(32),
        layout.getTypeSize(ptr.getElementType()));
    return size;
  } else
    return nullptr;
}

struct ExtractOpPattern : public ConversionToCallPattern<ep2::ExtractOp> {
  using ConversionToCallPattern<ep2::ExtractOp>::ConversionToCallPattern;

  LogicalResult
  matchAndRewrite(ep2::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto retType = getTypeConverter()->convertType(extractOp.getType());
    auto constOp = emitGetSize(rewriter, retType);
    if (!constOp)
      return rewriter.notifyMatchFailure(extractOp, "extract to a non-struct type");
    auto callOp = rewriter.create<LLVM::CallOp>(
        extractOp.getLoc(), apiFunctions["__rt_buf_extract"], ValueRange{adaptor.getBuffer(), constOp});
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(extractOp, retType, callOp.getResult());
    return success();
  }
};

struct EmitOpPattern : public ConversionToCallPattern<ep2::EmitOp> {
  using ConversionToCallPattern<ep2::EmitOp>::ConversionToCallPattern;

  LogicalResult
  matchAndRewrite(ep2::EmitOp emitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (emitOp.getValue().getType().isa<ep2::BufferType>())
      return rewriter.notifyMatchFailure(emitOp, "emit to a buffer type in non-buffer pattern");

    auto retType = adaptor.getValue().getType();
    auto constOp = emitGetSize(rewriter, retType);
    if (!constOp)
      return rewriter.notifyMatchFailure(emitOp, "emit to a non-struct type");

    auto ptr = castToVoid(rewriter, adaptor.getValue());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(emitOp,
        apiFunctions["__rt_buf_emit"],
        ValueRange{adaptor.getBuffer(), constOp, ptr});
    return success();
  }
};

struct EmitBufferPattern : public ConversionToCallPattern<ep2::EmitOp> {
  using ConversionToCallPattern<ep2::EmitOp>::ConversionToCallPattern;

  LogicalResult
  matchAndRewrite(ep2::EmitOp emitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!emitOp.getValue().getType().isa<ep2::BufferType>())
      return rewriter.notifyMatchFailure(emitOp, "emit to a non-buffer type in buffer pattern");
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(emitOp,
        apiFunctions["__rt_buf_concat"],
        ValueRange{adaptor.getBuffer(), adaptor.getValue()});
    return success();
  }
};

struct InitOpToBufPattern : public ConversionToCallPattern<ep2::InitOp> {
  using ConversionToCallPattern<ep2::InitOp>::ConversionToCallPattern;

  LogicalResult
  matchAndRewrite(ep2::InitOp initOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!initOp.getType().isa<ep2::BufferType>())
      return rewriter.notifyMatchFailure(initOp, "Not a buffer type");

    auto retType = getTypeConverter()->convertType(initOp.getType());
    auto const1 = rewriter.create<LLVM::ConstantOp>(
        initOp.getLoc(), rewriter.getIntegerType(32), 1);
    auto bufp = rewriter.create<LLVM::AllocaOp>(initOp.getLoc(), retType, const1, 0);
    rewriter.create<LLVM::CallOp>(initOp.getLoc(), apiFunctions["__rt_buf_init"], ValueRange{bufp});
    rewriter.replaceOp(initOp, bufp);
    return success();
  }
};

struct InitOpToEventPattern : public OpConversionPattern<ep2::InitOp> {
  using OpConversionPattern<ep2::InitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::InitOp initOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!initOp.getType().isa<ep2::StructType>() ||
        !initOp.getType().dyn_cast<ep2::StructType>().getIsEvent())
      return rewriter.notifyMatchFailure(initOp, "Not a event type");

    // create a struct
    auto retType = getTypeConverter()->convertType(initOp.getType());
    auto value1 = rewriter.create<LLVM::ConstantOp>(
        initOp.getLoc(), rewriter.getIntegerType(32), 1);
    auto ptr = rewriter.create<LLVM::AllocaOp>(initOp.getLoc(), retType, value1, 0);

    // assignments
    auto mapped = llvm::to_vector(adaptor.getOperands());
    // TODO(zhiyuang): now hardcoded to skip atom
    for (size_t i = 1; i < mapped.size(); i++) {
      auto gepOp = emitGEP(rewriter, mapped[i].getType(), ptr, i-1);
      emitStore(rewriter, mapped[i], gepOp);
    }

    // return
    rewriter.replaceOp(initOp, ptr);
    return success();
  }
};

struct AtomConstantPattern : public ConversionWithAnalysis<ep2::ConstantOp> {
  using ConversionWithAnalysis<ep2::ConstantOp>::ConversionWithAnalysis;

  LogicalResult
  matchAndRewrite(ep2::ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!constOp.getType().isa<ep2::AtomType>())
      return rewriter.notifyMatchFailure(constOp, "Not an atom type");
    auto &analysis = am.getAnalysis<AtomAnalysis>();
    auto [_, id] = analysis.atomToNum[constOp.getValue().dyn_cast<StringAttr>().getValue()];
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(constOp, rewriter.getIntegerType(32), id);
    return success();
  }
};

struct HandlerPattern : public OpConversionPattern<ep2::FuncOp> {
  using OpConversionPattern<ep2::FuncOp>::OpConversionPattern;

  struct HandlerRemapper {
    SmallVector<Type> types, convertedTypes;
    const TypeConverter &converter;
    OpBuilder &builder;

    HandlerRemapper(ArrayRef<Type> types, const TypeConverter &converter,
                    OpBuilder &builder)
        : types(types), converter(converter), builder(builder) {
        convertedTypes = convertTypes(builder, converter, types);
    }
    Value getMapped(Value eventStructPtr, int i) {
      // TODO: do not deref if we want a pointer (e.g on-stack struct) and its a pointer
      auto gepOp = emitGEP(builder, convertedTypes[i], eventStructPtr, i);
      if (isStructPointer(gepOp.getType()))
        return gepOp;
      else
        return builder.create<LLVM::LoadOp>(eventStructPtr.getLoc(), gepOp);
    }
    SmallVector<Type> getConvertedType() {
      auto inputStruct = LLVM::LLVMStructType::getLiteral(builder.getContext(),
                                                          convertedTypes);
      auto inputStructPtr = LLVM::LLVMPointerType::get(inputStruct);
      return {inputStructPtr};
    }
  };

  LogicalResult
  matchAndRewrite(ep2::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = funcOp->getLoc();
    std::string eventName = funcOp->getAttr("event").cast<StringAttr>().getValue().str();

    if (!funcOp.isHandler())
      return rewriter.notifyMatchFailure(funcOp, "Not a handler");

    if (funcOp.isExtern()) { // remove all externs in generated function calls
      rewriter.eraseOp(funcOp);
      return success();
    }

    // create a remapper
    HandlerRemapper remapper{funcOp.getArgumentTypes(), *getTypeConverter(), rewriter};

    // empty
    TypeConverter::SignatureConversion signatureConversion(
        funcOp.getNumArguments());

    auto functionType = rewriter.getFunctionType(remapper.getConvertedType(), TypeRange{});
    functionType.dump();
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        loc, funcOp.getName(),
        LLVM::LLVMFunctionType::get(rewriter.getType<LLVM::LLVMVoidType>(),
                                    {remapper.getConvertedType()}));

    auto entryBlock = newFuncOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);

    // create instructions convert from context to event
    for (size_t i = 0; i < funcOp.getNumArguments(); i++) {
      signatureConversion.remapInput(i, remapper.getMapped(newFuncOp.getArgument(0), i));
    }

    // change the function body
    auto newBlock = rewriter.convertRegionTypes(&funcOp.getBody(), *getTypeConverter(),
                                           &signatureConversion);
    if (failed(newBlock))
      return failure();
    rewriter.mergeBlocks(*newBlock, entryBlock);

    // remove original func
    rewriter.replaceOp(funcOp, newFuncOp);

    return success();
  }
};

struct ReturnOpPattern : public ConversionToCallPattern<ep2::ReturnOp> {
  using ConversionToCallPattern<ep2::ReturnOp>::ConversionToCallPattern;

  LogicalResult
  matchAndRewrite(ep2::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto event = adaptor.getInput().front();
    auto ptr = castToVoid(rewriter, event);
    auto size = emitGetSize(rewriter, event.getType());
    auto targetQid = rewriter.create<LLVM::ConstantOp>(
        returnOp.getLoc(), rewriter.getIntegerType(32), 1);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(returnOp,
                                            apiFunctions["__rt_generate"],
                                            ValueRange{targetQid, size, ptr});
    return success();
  }
};

struct TerminatePattern : public OpConversionPattern<ep2::TerminateOp> {
  using OpConversionPattern<ep2::TerminateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::TerminateOp termOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(termOp, ValueRange{});
    return success();
  }
};


// Aggregated functions
void populateTypeConversion(TypeConverter &typeConverter, OpBuilder &builder) {
  // missing context conversion
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([&](ep2::ContextType type) {
    // This is a void *
    return builder.getType<LLVM::LLVMPointerType>();
  });
  typeConverter.addConversion([&](ep2::BufferType type) {
    auto bufferImpl = LLVM::LLVMStructType::getLiteral(
        builder.getContext(),
        {builder.getType<LLVM::LLVMPointerType>(), // void *
         builder.getIntegerType(32), builder.getIntegerType(32)}
        // int size, offset
    );
    return LLVM::LLVMPointerType::get(bufferImpl);
  });
  typeConverter.addConversion([&](ep2::TableType type) {
    return builder.getType<LLVM::LLVMPointerType>();
  });
  typeConverter.addConversion([&](ep2::AtomType type) {
    return mlir::IntegerType::get(type.getContext(), 32);
  });
  typeConverter.addConversion([&](ep2::StructType type) {
    auto structType = LLVM::LLVMStructType::getLiteral(
      builder.getContext(),
      convertTypes(builder, typeConverter, type.getElementTypes()),
      !type.getIsEvent() // All normal structs are packed
    );
    return LLVM::LLVMPointerType::get(structType);
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
  typeConverter.addArgumentMaterialization(
      [&](OpBuilder &builder, Type type, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        llvm::errs() << "wildcard conversion: argument\n";
        if (inputs.size() != 1)
          return std::nullopt;
        return builder
            .create<UnrealizedConversionCastOp>(loc, TypeRange{type}, inputs)
            .getOutputs()[0];
      });
  typeConverter.addTargetMaterialization(
      [&](OpBuilder &builder, Type type, ValueRange inputs,
          Location loc) -> std::optional<Value> {
        llvm::errs() << "wildcard conversion: target\n";
        if (inputs.size() != 1)
          return std::nullopt;
        return builder
            .create<UnrealizedConversionCastOp>(loc, TypeRange{type}, inputs)
            .getOutputs()[0];
      });
}

} // localnamespace

void LowerLLVMPass::populateAPIFunctions(TypeConverter &converter) {
  ModuleOp module = getOperation();
  OpBuilder builder(module);
  builder.setInsertionPointToStart(module.getBody());

  // Types
  auto I32 = builder.getIntegerType(32);
  auto Void = builder.getType<LLVM::LLVMVoidType>();
  auto VoidPtr = builder.getType<LLVM::LLVMPointerType>();
  auto Buf = converter.convertType(builder.getType<ep2::BufferType>());

  apiFunctions["__rt_buf_extract"] = builder.create<LLVM::LLVMFuncOp>(
      module.getLoc(), "__rt_buf_extract",
      LLVM::LLVMFunctionType::get(VoidPtr, {Buf, I32}));
  apiFunctions["__rt_buf_emit"] = builder.create<LLVM::LLVMFuncOp>(
      module.getLoc(), "__rt_buf_emit",
      LLVM::LLVMFunctionType::get(Void, {Buf, I32, VoidPtr}));
  apiFunctions["__rt_buf_concat"] = builder.create<LLVM::LLVMFuncOp>(
      module.getLoc(), "__rt_buf_concat",
      LLVM::LLVMFunctionType::get(Void, {Buf, Buf}));
  apiFunctions["__rt_buf_init"] = builder.create<LLVM::LLVMFuncOp>(
      module.getLoc(), "__rt_buf_init",
      LLVM::LLVMFunctionType::get(Void, {Buf}));
  apiFunctions["__rt_generate"] = builder.create<LLVM::LLVMFuncOp>(
      module.getLoc(), "__rt_generate",
      LLVM::LLVMFunctionType::get(Void, {I32, I32, VoidPtr}));
}


void LowerLLVMPass::runOnOperation() {
  // install analysis
  auto am = getAnalysisManager();
  am.getAnalysis<AtomAnalysis>();

  auto builder = OpBuilder(getOperation());

  // Dialect Type converter
  TypeConverter typeConverter;
  populateTypeConversion(typeConverter, builder);

  // prepare api functions
  populateAPIFunctions(typeConverter);

  // Dialect conversion target
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<ep2::EP2Dialect, func::FuncDialect, LLVM::LLVMDialect,
                         BuiltinDialect>();

  target.addIllegalOp<ep2::FuncOp, ep2::TerminateOp,
  ep2::StructAccessOp, ep2::StructUpdateOp, ep2::ExtractOp, ep2::EmitOp,
  ep2::ReturnOp>();
  target.addDynamicallyLegalOp<ep2::InitOp>([&](ep2::InitOp op) {
    return !op.getType().isa<BufferType,StructType>();
  });
  target.addDynamicallyLegalOp<ep2::ConstantOp>([&](ep2::ConstantOp op) {
    return !op.getType().isa<AtomType>();
  });
  // target.addDynamicallyLegalDialect<scf::SCFDialect>([&](mlir::Operation* op){
  //   return typeConverter.isLegal(op);
  // });

  // apply rules
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<HandlerPattern, TerminatePattern, StructAccessOpPattern,
               StructUpdateOpPattern>(typeConverter, &getContext());
  patterns.add<ExtractOpPattern, EmitOpPattern, EmitBufferPattern,
               InitOpToBufPattern, ReturnOpPattern>(
      typeConverter, &getContext(), apiFunctions);
  patterns.add<AtomConstantPattern>(typeConverter, &getContext(), am);

  patterns.add<InitOpToEventPattern>(typeConverter, &getContext());

  FrozenRewritePatternSet patternSet(std::move(patterns));

  if (failed(applyPartialConversion(getOperation(), target, patternSet)))
    signalPassFailure();
}

} // namespace ep2
} // namespace mlir