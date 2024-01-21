#include "mlir/IR/BuiltinDialect.h"

#include "ep2/lang/Lexer.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

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

Type getI8PtrType(OpBuilder &builder) {
  return LLVM::LLVMPointerType::get(builder.getI8Type());
}

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
int getTypeSize(Value value) {
  return getTypeSize(value.getType());
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
  // TODO(zhiyuang): for now we remove all atom types
  auto filtered = llvm::to_vector(ep2Types);
  if (filtered[0].isa<ep2::AtomType>())
    filtered.erase(filtered.begin());

  auto types = llvm::map_to_vector(
      filtered, [&](Type t) { return converter.convertType(t); });
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

// Buf Operations
Value getBufPointer(OpBuilder &builder, Value buf) {
  auto bufp = emitGEP(builder, getI8PtrType(builder), buf, 0);
  auto ptr = builder.create<LLVM::LoadOp>(buf.getLoc(), getI8PtrType(builder), bufp);
  return ptr;
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

struct InitOpToStructPattern : public OpConversionPattern<ep2::InitOp> {
  using OpConversionPattern<ep2::InitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::InitOp initOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!initOp.getType().isa<ep2::StructType>() ||
        initOp.getType().dyn_cast<ep2::StructType>().getIsEvent())
      return rewriter.notifyMatchFailure(initOp, "init a struct but not event");

    if (initOp.getNumOperands() != 0)
      return rewriter.notifyMatchFailure(initOp, "init normal struct only support empty init");

    // create a struct
    auto retType = getTypeConverter()->convertType(initOp.getType());
    auto value1 = rewriter.create<LLVM::ConstantOp>(
        initOp.getLoc(), rewriter.getIntegerType(32), 1);

    // return
    rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(initOp, retType, value1, 0);
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
    // TODO(zhiyuang): hard code jmp over atom
    auto mapped = llvm::to_vector(adaptor.getOperands());
    for (size_t i = 1; i < mapped.size(); i++) {
      auto gepOp = emitGEP(rewriter, mapped[i].getType(), ptr, i-1);
      emitStore(rewriter, mapped[i], gepOp);
    }

    // return
    rewriter.replaceOp(initOp, ptr);
    return success();
  }
};

struct InitOpToEventRawPattern : public OpConversionPattern<ep2::InitOp> {
  using OpConversionPattern<ep2::InitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::InitOp initOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!initOp.getType().isa<ep2::StructType>() ||
        !initOp.getType().dyn_cast<ep2::StructType>().getIsEvent())
      return rewriter.notifyMatchFailure(initOp, "Not a event type");

    // TODO(zhiyuang): remove the atom first
    auto mapped = llvm::to_vector(adaptor.getOperands());
    if (initOp.getNumOperands() > 0 && initOp.getOperand(0).getType().isa<ep2::AtomType>())
      mapped.erase(mapped.begin());

    HandlerDependencyAnalysis::HandlerFullName next{initOp};
    rewriter.create<LLVM::CallOp>(initOp.getLoc(), TypeRange{}, next.mangle(), mapped);
    rewriter.replaceOp(initOp, mapped[0]);

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

struct ConstantPattern : public OpConversionPattern<ep2::ConstantOp> {
  using OpConversionPattern<ep2::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::ConstantOp constOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>();
    if (!intAttr)
      return rewriter.notifyMatchFailure(constOp, "only support integer type");

    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(constOp, constOp.getType(), intAttr.getValue());
    return success();
  }
};

struct ControllerPattern : public OpConversionPattern<ep2::FuncOp> {
  using OpConversionPattern<ep2::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!funcOp.isController())
      return rewriter.notifyMatchFailure(funcOp, "Not a controller");

    // remove original func
    rewriter.eraseOp(funcOp);
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
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());

    // remove original func
    rewriter.replaceOp(funcOp, newFuncOp);

    return success();
  }
};

struct HandlerRawPattern : public OpConversionPattern<ep2::FuncOp> {
  using OpConversionPattern<ep2::FuncOp>::OpConversionPattern;

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
    auto convertedTypes = llvm::map_to_vector(
      convertTypes(rewriter, *getTypeConverter(), funcOp.getArgumentTypes()),
      [](Type t) -> Type {
        if (isa<LLVM::LLVMStructType>(t))
          return LLVM::LLVMPointerType::get(t);
        else
          return t;
      });


    // empty
    TypeConverter::SignatureConversion signatureConversion(
        funcOp.getNumArguments());

    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        loc, funcOp.getName(),
        LLVM::LLVMFunctionType::get(rewriter.getType<LLVM::LLVMVoidType>(),
                                    convertedTypes));

    auto entryBlock = newFuncOp.addEntryBlock();
    // create instructions convert from context to event
    for (size_t i = 0; i < funcOp.getNumArguments(); i++) {
      signatureConversion.remapInput(i, newFuncOp.getArgument(i));
    }

    // change the function body
    auto newBlock = rewriter.convertRegionTypes(&funcOp.getBody(), *getTypeConverter(),
                                           &signatureConversion);
    if (failed(newBlock))
      return failure();
    rewriter.mergeBlocks(*newBlock, entryBlock);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());

    // remove original func
    rewriter.replaceOp(funcOp, newFuncOp);

    return success();
  }
};

// try lower to native..
struct EmitOffsetPattern : public OpConversionPattern<ep2::EmitOffsetOp> {
  using OpConversionPattern<ep2::EmitOffsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::EmitOffsetOp emitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (emitOp.getOffset() % 8 != 0)
      return rewriter.notifyMatchFailure(emitOp, "offset not aligned to 8");

    auto valueType = adaptor.getValue().getType();
    auto byteOffset = static_cast<int>(emitOp.getOffset() / 8);

    auto bufBase = getBufPointer(rewriter, adaptor.getBuffer());
    auto bufp = rewriter.create<LLVM::GEPOp>(
        bufBase.getLoc(), getI8PtrType(rewriter), bufBase,
        ArrayRef<LLVM::GEPArg>{byteOffset});

    if (isa<LLVM::LLVMPointerType>(valueType)) {
      auto size = getTypeSize(adaptor.getValue());
      rewriter.create<LLVM::MemcpyInlineOp>(emitOp.getLoc(), bufp, adaptor.getValue(), rewriter.getI32IntegerAttr(size), false);
    } else { // its a value type
      auto casted = rewriter.create<LLVM::BitcastOp>(
          emitOp.getLoc(), LLVM::LLVMPointerType::get(valueType), bufp);
      rewriter.create<LLVM::StoreOp>(
          emitOp.getLoc(), adaptor.getValue(),
          casted);
    }
    rewriter.eraseOp(emitOp);
    return success();
  }
};

struct ExtractOffsetPattern : public OpConversionPattern<ep2::ExtractOffsetOp> {
  using OpConversionPattern<ep2::ExtractOffsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::ExtractOffsetOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (extractOp.getOffset() % 8 != 0)
      return rewriter.notifyMatchFailure(extractOp, "offset not aligned to 8");

    auto valueType = getTypeConverter()->convertType(extractOp.getType());
    auto byteOffset = static_cast<int>(extractOp.getOffset() / 8);

    auto bufBase = getBufPointer(rewriter, adaptor.getBuffer());
    auto bufp = rewriter.create<LLVM::GEPOp>(
        bufBase.getLoc(), getI8PtrType(rewriter), bufBase,
        ArrayRef<LLVM::GEPArg>{byteOffset});

    if (isa<LLVM::LLVMPointerType>(valueType)) {
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(extractOp, valueType, bufp);
    } else { // its a value type
      auto casted = rewriter.create<LLVM::BitcastOp>(
          extractOp.getLoc(), LLVM::LLVMPointerType::get(valueType), bufp);
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(extractOp, casted);
    }
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

struct ReturnOpCallPattern : public ConversionWithAnalysis<ep2::ReturnOp> {
  using ConversionWithAnalysis<ep2::ReturnOp>::ConversionWithAnalysis;

  LogicalResult
  matchAndRewrite(ep2::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto event = adaptor.getInput().front();
    HandlerDependencyAnalysis::HandlerFullName next{returnOp};
    auto &analysis = am.getAnalysis<HandlerDependencyAnalysis>();

    // TODO(zhiyuang): this is a hack. fix me!
    if (analysis.lookupHandler(next).isExtern()) 
      event = castToVoid(rewriter, event);
  
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(returnOp, TypeRange{}, next.mangle(), ValueRange{event});
    return success();
  }
};

struct ReturnOpRawPattern : public OpConversionPattern<ep2::ReturnOp> {
  using OpConversionPattern<ep2::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(returnOp);
    return success();
  }
};

struct GlobalPattern : public OpConversionPattern<ep2::GlobalOp> {
  using OpConversionPattern<ep2::GlobalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::GlobalOp globalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto resType = getTypeConverter()->convertType(globalOp.getType());
    rewriter.create<LLVM::GlobalOp>(globalOp.getLoc(), resType, false, LLVM::Linkage::External, globalOp.getName(), nullptr);
    rewriter.eraseOp(globalOp);
    return success();
  }
};  

struct GlobalImportPattern : public OpConversionPattern<ep2::GlobalImportOp> {
  using OpConversionPattern<ep2::GlobalImportOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::GlobalImportOp importOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto resType = getTypeConverter()->convertType(importOp.getType());
    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(importOp, resType, importOp.getName());
    return success();
  }
};

struct LookupPattern : public ConversionToCallPattern<ep2::LookupOp> {
  using ConversionToCallPattern<ep2::LookupOp>::ConversionToCallPattern;

  LogicalResult
  matchAndRewrite(ep2::LookupOp lookupOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO: check the table type. only support int as a key for now
    auto resType = getTypeConverter()->convertType(lookupOp.getType());
    if (!adaptor.getKey().getType().isa<IntegerType>())
      return rewriter.notifyMatchFailure(lookupOp, "lookup with non-integer key");

    // key type conversion
    Value key = adaptor.getKey();
    if (adaptor.getKey().getType() != rewriter.getIntegerType(32))
      key = rewriter.create<BitCastOp>(lookupOp.getLoc(), rewriter.getIntegerType(32), adaptor.getKey());

    auto callOp = rewriter.create<LLVM::CallOp>(
        lookupOp.getLoc(), apiFunctions["__rt_table_lookup"], ValueRange{adaptor.getTable(), key});
    if (isa<LLVM::LLVMPointerType>(resType)) {
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(lookupOp, resType, callOp.getResult());
    } else {
      llvm_unreachable("not implemented");
    }
    return success();
  }
};

struct UpdatePattern : public ConversionToCallPattern<ep2::UpdateOp> {
  using ConversionToCallPattern<ep2::UpdateOp>::ConversionToCallPattern;

  LogicalResult
  matchAndRewrite(ep2::UpdateOp updateOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!adaptor.getKey().getType().isa<IntegerType>())
      return rewriter.notifyMatchFailure(updateOp,
                                         "lookup with non-integer key");
    // key type conversion
    Value key = adaptor.getKey();
    if (adaptor.getKey().getType() != rewriter.getIntegerType(32))
      key = rewriter.create<BitCastOp>(updateOp.getLoc(), rewriter.getIntegerType(32), adaptor.getKey());

    Value value;
    if (isa<LLVM::LLVMPointerType>(adaptor.getValue().getType())) {
      value = rewriter.create<LLVM::BitcastOp>(
          updateOp.getLoc(), rewriter.getType<LLVM::LLVMPointerType>(),
          adaptor.getValue());
    } else {
      llvm_unreachable("not implemented value type as a value in table update");
    }
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        updateOp, apiFunctions["__rt_table_update"], ValueRange{adaptor.getTable(), key, value});
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

// control flow
struct CFCondBranchPattern : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern<cf::CondBranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp branchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(branchOp, [&](){
      branchOp->setOperands(adaptor.getOperands());
    });
    return success();
  }
};

struct CFBranchPattern : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern<cf::BranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp branchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(branchOp, [&](){
      branchOp->setOperands(adaptor.getOperands());
    });
    return success();
  }
};

// patterns to arith ops
struct AddPattern : public OpConversionPattern<ep2::AddOp> {
  using OpConversionPattern<ep2::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::AddOp addOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<LLVM::AddOp>(addOp, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct SubPattern : public OpConversionPattern<ep2::SubOp> {
  using OpConversionPattern<ep2::SubOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::SubOp subOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<LLVM::AddOp>(subOp, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct CmpPattern : public OpConversionPattern<ep2::CmpOp> {
  using OpConversionPattern<ep2::CmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    LLVM::ICmpPredicate predicate;
    switch (op.getPredicate()) {
      case -::ep2::tok_cmp_eq:
        predicate = LLVM::ICmpPredicate::eq;
        break;
      case -::ep2::tok_cmp_le:
        predicate = LLVM::ICmpPredicate::sle;
        break;
      case -::ep2::tok_cmp_ge:
        predicate = LLVM::ICmpPredicate::sge;
        break;
      case '<':
        predicate = LLVM::ICmpPredicate::slt;
        break;
      case '>':
        predicate = LLVM::ICmpPredicate::sgt;
        break;
      default: {
        assert(false && "Unsupported comparison");
        break;
      }
    }
    auto attr = LLVM::ICmpPredicateAttr::get(rewriter.getContext(), predicate);
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, attr, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct BitCastPattern : public OpConversionPattern<ep2::BitCastOp> {
  using OpConversionPattern<ep2::BitCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::BitCastOp bitcastOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto srcType = bitcastOp.getInput().getType().dyn_cast<IntegerType>();
    auto dstType = bitcastOp.getType().dyn_cast<IntegerType>();
    if (!srcType || !dstType)
      return rewriter.notifyMatchFailure(bitcastOp, "bitcast from non-integer type");
    
    if (srcType.getWidth() < dstType.getWidth())
      rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(bitcastOp, dstType, adaptor.getInput());
    else if (srcType.getWidth() > dstType.getWidth())
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(bitcastOp, dstType, adaptor.getInput());
    else // eq
      rewriter.replaceOp(bitcastOp, adaptor.getInput());
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
  // table related
  apiFunctions["__rt_table_lookup"] = builder.create<LLVM::LLVMFuncOp>(
      module.getLoc(), "__rt_table_lookup",
      LLVM::LLVMFunctionType::get(VoidPtr, {VoidPtr, I32}));
  apiFunctions["__rt_table_update"] = builder.create<LLVM::LLVMFuncOp>(
      module.getLoc(), "__rt_table_update",
      LLVM::LLVMFunctionType::get(Void, {VoidPtr, I32, VoidPtr}));

  // TODO(zhiyuang): change to enum
  // external handlers. Only for call and raw mode
  if (generateMode == "call") {
    apiFunctions["__handler_NET_SEND_net_send"] = builder.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "__handler_NET_SEND_net_send",
        LLVM::LLVMFunctionType::get(Void, {VoidPtr}));
    apiFunctions["__handler_DMA_WRITE_REQ_dma_write"] = builder.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "__handler_DMA_WRITE_REQ_dma_write",
        LLVM::LLVMFunctionType::get(Void, {VoidPtr}));
  } else if (generateMode == "raw") {
    apiFunctions["__handler_NET_SEND_net_send"] = builder.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "__handler_NET_SEND_net_send",
        LLVM::LLVMFunctionType::get(Void, {Buf}));
    auto DmaType = LLVM::LLVMPointerType::get( LLVM::LLVMStructType::getLiteral(builder.getContext(), {I32, I32}, true));
    apiFunctions["__handler_DMA_WRITE_REQ_dma_write"] =
        builder.create<LLVM::LLVMFuncOp>(
            module.getLoc(), "__handler_DMA_WRITE_REQ_dma_write",
            LLVM::LLVMFunctionType::get( Void, {Buf, DmaType}));
  }
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
  target.addIllegalDialect<ep2::EP2Dialect>();
  target.addLegalDialect<LLVM::LLVMDialect, BuiltinDialect>();
  target.addDynamicallyLegalDialect<cf::ControlFlowDialect>([&](mlir::Operation* op){
    return typeConverter.isLegal(op);
  });
  target.addDynamicallyLegalOp<ep2::InitOp>([&](ep2::InitOp op) {
    return !op.getType().isa<BufferType,StructType>();
  });

  // apply rules
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ControllerPattern, TerminatePattern, StructAccessOpPattern,
               StructUpdateOpPattern, CFCondBranchPattern, CFBranchPattern, ConstantPattern>(typeConverter, &getContext());
  patterns.add<ExtractOpPattern, EmitOpPattern, EmitBufferPattern,
               InitOpToBufPattern>(
      typeConverter, &getContext(), apiFunctions);
  patterns.add<LookupPattern, UpdatePattern>(typeConverter, &getContext(), apiFunctions);
  patterns.add<AtomConstantPattern>(typeConverter, &getContext(), am);

  patterns.add<InitOpToStructPattern>(typeConverter, &getContext());
  // TODO: use runtime?
  patterns.add<EmitOffsetPattern, ExtractOffsetPattern>(typeConverter, &getContext());
  patterns.add<AddPattern, SubPattern, CmpPattern, BitCastPattern, GlobalImportPattern, GlobalPattern>(typeConverter, &getContext());

  // function related ones
  if (generateMode == "rt") {
    patterns.add<InitOpToEventPattern, HandlerPattern>(typeConverter, &getContext());
    patterns.add<ReturnOpPattern>(typeConverter, &getContext(), apiFunctions);
  } else if (generateMode == "call") {
    patterns.add<InitOpToEventPattern, HandlerPattern>(typeConverter, &getContext());
    patterns.add<ReturnOpCallPattern>(typeConverter, &getContext(), am);
  } else if (generateMode == "raw") {
    patterns
        .add<HandlerRawPattern, ReturnOpRawPattern, InitOpToEventRawPattern>(
            typeConverter, &getContext());
  } else
    llvm_unreachable("unknown generate mode");

  FrozenRewritePatternSet patternSet(std::move(patterns));

  if (failed(applyPartialConversion(getOperation(), target, patternSet)))
    signalPassFailure();
}

} // namespace ep2
} // namespace mlir