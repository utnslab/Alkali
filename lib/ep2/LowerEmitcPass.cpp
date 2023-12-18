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

#include "llvm/ADT/SmallSet.h"

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

  ContextBufferizationAnalysis &analyzer;
  StorePattern(TypeConverter &converter, MLIRContext *context,
                  ContextBufferizationAnalysis &analyzer)
      : OpConversionPattern<ep2::StoreOp>(converter, context),
        analyzer(analyzer) {}

  LogicalResult matchAndRewrite(ep2::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto refOp = dyn_cast<ep2::ContextRefOp>(storeOp.getOperand(0).getDefiningOp());
    auto contextId = rewriter.getRemappedValue(refOp.getOperand());

    llvm::SmallVector<Type> resTypes = {};
    int pos = analyzer.getContextType(cast<func::FuncOp>(storeOp->getParentOp()), refOp.getName()).first;
    mlir::ArrayAttr args = rewriter.getI32ArrayAttr({pos});
    mlir::ArrayAttr templ_args;

    rewriter.replaceOpWithNewOp<emitc::CallOp>(storeOp, resTypes, rewriter.getStringAttr("__ep2_intrin_ctx_write"), args, templ_args, ValueRange{adaptor.getValue(), contextId});

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

    rewriter.eraseOp(returnOp);
    return success();
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

static std::unordered_map<mlir::Operation*, unsigned> offsetMap;

struct ExtractPattern : public OpConversionPattern<ep2::ExtractOp> {
  using OpConversionPattern<ep2::ExtractOp>::OpConversionPattern;

  LocalAllocAnalysis &analyzer;
  ExtractPattern(TypeConverter &converter, MLIRContext *context,
                  LocalAllocAnalysis &analyzer)
      : OpConversionPattern<ep2::ExtractOp>(converter, context),
        analyzer(analyzer) {}

  LogicalResult
  matchAndRewrite(ep2::ExtractOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = extractOp->getLoc();
    auto resType = extractOp.getResult().getType();
    if (!resType.isa<ep2::StructType>())
      return rewriter.notifyMatchFailure(extractOp, "Currently only support extract op on struct");

    // TODO support dynamically allocated structs too
    assert(analyzer.localAllocs.find(extractOp) != analyzer.localAllocs.end());

    auto varOp = rewriter.create<emitc::VariableOp>(loc, typeConverter->convertType(resType), emitc::OpaqueAttr::get(getContext(), std::string{"&"} + analyzer.localAllocs[extractOp]));
    rewriter.replaceOp(extractOp, varOp);

    // calculate size
    int structSize = 0;
    for (mlir::Type ty : cast<ep2::StructType>(resType).getElementTypes()) {
      if (ty.isIntOrFloat()) {
        structSize += ty.getIntOrFloatBitWidth() / 8;
      }
    }

    if (offsetMap.find(extractOp->getParentOp()) == offsetMap.end()) {
      offsetMap[extractOp->getParentOp()] = 0;
    }
    offsetMap[extractOp->getParentOp()] += structSize;

    llvm::SmallVector<Type> resTypes2 = {};
    mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({structSize});
    mlir::ArrayAttr templ_args2;
    rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_memcpy"), args2, templ_args2, ValueRange{varOp, adaptor.getBuffer()});

    llvm::SmallVector<Type> resTypes = {};
    mlir::ArrayAttr args = rewriter.getI32ArrayAttr({structSize});
    mlir::ArrayAttr templ_args;
    rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_intrin_incr_ptr"), args, templ_args, ValueRange{adaptor.getBuffer()});

    // TODO a copy elision- move such optimization to a separate pass.
    // rewriter.replaceOpWithNewOp<emitc::CastOp>(extractOp, typeConverter->convertType(resType), adaptor.getBuffer());
    return success();
  }
};

struct EmitPattern : public OpConversionPattern<ep2::EmitOp> {
  using OpConversionPattern<ep2::EmitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::EmitOp emitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = emitOp->getLoc();
    auto resType = emitOp.getValue().getType();

    int size = 0;
    if (isa<ep2::BufferType>(resType)) {
      // assume MTU-size alloc of buffers
      size = 1500 - offsetMap[emitOp->getParentOp()];
    } else if (isa<ep2::StructType>(resType)) {
      for (mlir::Type ty : cast<ep2::StructType>(resType).getElementTypes()) {
        if (ty.isIntOrFloat()) {
          size += ty.getIntOrFloatBitWidth() / 8;
        }
      }
    } else {
      assert(false && "Unsupported emit type.");
    }

    llvm::SmallVector<Type> resTypes2 = {};
    mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({size});
    mlir::ArrayAttr templ_args2;
    auto emit = rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_memcpy"), args2, templ_args2, ValueRange{adaptor.getBuffer(), adaptor.getValue()});
    rewriter.replaceOp(emitOp, emit);

    llvm::SmallVector<Type> resTypes = {};
    mlir::ArrayAttr args = rewriter.getI32ArrayAttr({size});
    mlir::ArrayAttr templ_args;
    rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_intrin_incr_ptr"), args, templ_args, ValueRange{adaptor.getBuffer()});

    return success();
  }
};

struct StructUpdatePattern : public OpConversionPattern<ep2::StructUpdateOp> {
  using OpConversionPattern<ep2::StructUpdateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::StructUpdateOp updateOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = updateOp->getLoc();
    llvm::SmallVector<Type> resTypes2 = {typeConverter->convertType(updateOp.getOperand(0).getType())};
    // todo
    mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({updateOp.getIndex()});
    mlir::ArrayAttr templ_args2;
    auto callOp = rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_struct_write"), args2, templ_args2, ValueRange{adaptor.getNewValue(), adaptor.getInput()});

    rewriter.replaceOp(updateOp, callOp);
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

    bool isGenerate = true;
    if (initOp->getNumResults() == 1) {
      for (auto user : initOp->getUsers()) {
        if (!isa<ep2::ReturnOp>(user)) {
          isGenerate = false;
        }
      }
    }

    if (resType.isa<ep2::StructType>()) {
      auto newType = typeConverter->convertType(resType);

      auto alloc = rewriter.create<emitc::VariableOp>(loc, newType, emitc::OpaqueAttr::get(getContext(), "&next_work"));
      rewriter.replaceOp(initOp, alloc);

      if (isGenerate) {
        auto outputStructPtr = cast<func::FuncOp>(alloc->getParentOp()).getArgument(1);
        llvm::SmallVector<Type> resTypes4 = {};
        mlir::ArrayAttr args40 = rewriter.getI32ArrayAttr({0});
        mlir::ArrayAttr args41 = rewriter.getI32ArrayAttr({1});
        mlir::ArrayAttr templ_args4;
        rewriter.create<emitc::CallOp>(loc, resTypes4, rewriter.getStringAttr("__ep2_intrin_struct_write"), args40, templ_args4, ValueRange{adaptor.getOperands()[0], outputStructPtr});
        rewriter.create<emitc::CallOp>(loc, resTypes4, rewriter.getStringAttr("__ep2_intrin_struct_write"), args41, templ_args4, ValueRange{alloc, outputStructPtr});
      }
      
      unsigned p = 0;
      for (const auto& opd : adaptor.getOperands()) {
        if (p == 0) {
          p += 1;
          continue;
        }
        llvm::SmallVector<Type> resTypes2 = {};
        mlir::ArrayAttr args2;
        if (p == 1) {
          args2 = rewriter.getStrArrayAttr({"ctx"});
        } else {
          args2 = rewriter.getI32ArrayAttr({p-2});
        }
        mlir::ArrayAttr templ_args2;
        rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_struct_write"), args2, templ_args2, ValueRange{opd, alloc});
        p += 1;
      }
    } else if (resType.isa<ep2::BufferType>()) {
      auto newType = typeConverter->convertType(resType);

      llvm::SmallVector<Type> resTypes = {newType};
      // TODO: Get the size to transfer. 
      // Smh lower init to separate the allocation decision from filling out the struct.
      mlir::ArrayAttr args;
      mlir::ArrayAttr templ_args;
      auto alloc = rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_rt_alloc_buf"), args, templ_args, ValueRange{});
      rewriter.replaceOp(initOp, alloc);
    } else {
      return rewriter.notifyMatchFailure(initOp, "Currently only support init op on struct");
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
    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct TerminatePattern : public OpConversionPattern<ep2::TerminateOp> {
  using OpConversionPattern<ep2::TerminateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::TerminateOp termOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(termOp);
    return success();
  }
};

// convert function
struct FunctionPattern : public OpConversionPattern<ep2::FuncOp> {

  LowerStructAnalysis &analyzer;
  HandlerDependencyAnalysis &handAnalyzer;
  LocalAllocAnalysis &allocAnalyzer;

  FunctionPattern(TypeConverter &converter, MLIRContext *context,
                  LowerStructAnalysis &analyzer, HandlerDependencyAnalysis &handAnalyzer, LocalAllocAnalysis &allocAnalyzer)
      : OpConversionPattern<ep2::FuncOp>(converter, context),
        analyzer(analyzer), handAnalyzer(handAnalyzer), allocAnalyzer(allocAnalyzer) {}

  LogicalResult
  matchAndRewrite(ep2::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = funcOp->getLoc();
    if (funcOp->getAttr("type").cast<StringAttr>().getValue() != "handler")
      return rewriter.notifyMatchFailure(funcOp, "Not a handler");

    // rewrite a function call.
    ArrayRef<LLVM::LLVMStructType> wrapperTypes = analyzer.getWrapperTypes(funcOp);
    if (wrapperTypes.size() == 0) {
      funcOp.emitError("Cannot rewrite with valid input");
      return failure();
    }

    auto newArgTypes = getArgumentTypes(rewriter, funcOp.getName(), wrapperTypes.size());
    TypeConverter::SignatureConversion signatureConversion(
        funcOp.getNumArguments());

    auto newFuncOp = rewriter.create<func::FuncOp>(
        loc, std::string{"__event_" + funcOp.getName().str()},
        rewriter.getFunctionType(TypeRange(newArgTypes), TypeRange{}));
    
    auto entryBlock = newFuncOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);

    std::string eventName = funcOp->getAttr("event").cast<StringAttr>().getValue().str();

    // construct body and replace parameter
    auto inputWrapperType =
        rewriter.getType<emitc::PointerType>(rewriter.getType<emitc::OpaqueType>(std::string{"struct event_param_" + eventName}));
    auto contextType =
        rewriter.getType<emitc::PointerType>(rewriter.getType<emitc::OpaqueType>(std::string{"struct context_chain_1_t"}));
    auto inputStructPtr = newFuncOp.getArgument(0);

    llvm::SmallVector<Type> resTypes = {inputWrapperType};
    mlir::ArrayAttr args = rewriter.getI32ArrayAttr({1});
    mlir::ArrayAttr templ_args;
    auto eventPtr = rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_intrin_struct_access"), args, templ_args, ValueRange{inputStructPtr});

    llvm::SmallVector<Type> resTypes3 = {contextType};
    mlir::ArrayAttr args3 = rewriter.getStrArrayAttr({"ctx"});
    mlir::ArrayAttr templ_args3;
    auto structPtr = rewriter.create<emitc::CallOp>(loc, resTypes3, rewriter.getStringAttr("__ep2_intrin_struct_access"), args3, templ_args3, ValueRange{eventPtr.getResult(0)});
    signatureConversion.remapInput(0, structPtr.getResult(0));

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
      auto param = rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_struct_access"), args2, templ_args2, ValueRange{eventPtr.getResult(0)});
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
                                                 llvm::StringRef funcName,
                                                 int num) const {
    auto context = rewriter.getContext();

    // insert if not done
    llvm::SmallVector<mlir::Type> types;
    for (int i = 0; i < num; i++) {
      auto argType = rewriter.getType<emitc::OpaqueType>("struct __wrapper_arg_t");
      types.push_back(rewriter.getType<emitc::PointerType>(argType));
    }
    return types;
  }
};

} // namespace

void LowerEmitcPass::runOnOperation() {
  // install analysis
  LowerStructAnalysis &lowerStructAnalysis = getAnalysis<LowerStructAnalysis>();
  ContextBufferizationAnalysis &contextAnalysis = getAnalysis<ContextBufferizationAnalysis>();
  HandlerDependencyAnalysis &handlerDepAnalysis = getAnalysis<HandlerDependencyAnalysis>();
  AtomAnalysis &atomAnalysis = getAnalysis<AtomAnalysis>();
  LocalAllocAnalysis &allocAnalysis = getAnalysis<LocalAllocAnalysis>();

  // install functions
  auto builder = OpBuilder(getOperation());

  // Dialect Type converter
  TypeConverter typeConverter;
  // All other types are valid
  // conversion:: 1 to 1 type mapping
  typeConverter.addConversion([](Type type) { return type; });
  // TODO add an attribute to ep2::ContextType to know which context_chain_t to convert to.
  typeConverter.addConversion([&](ep2::ContextType type) {
    return emitc::PointerType::get(type.getContext(), builder.getType<emitc::OpaqueType>("struct context_chain_1_t"));
  });
  typeConverter.addConversion([&](ep2::BufferType type) {
    return emitc::PointerType::get(type.getContext(), builder.getType<emitc::OpaqueType>("char"));
  });
  typeConverter.addConversion([&](ep2::AtomType type) {
    return mlir::IntegerType::get(type.getContext(), 32);
  });
  typeConverter.addConversion([&](ep2::StructType type) {
    return emitc::PointerType::get(type.getContext(), builder.getType<emitc::OpaqueType>(std::string{"struct "} + (type.getIsEvent() ? "event_param_" : "") + type.getName().str()));
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
      .addIllegalOp<ep2::ConstantOp, ep2::StoreOp, ep2::ContextRefOp, ep2::TerminateOp,
                    ep2::CallOp, ep2::FuncOp, ep2::ReturnOp, ep2::StructUpdateOp,
                    ep2::InitOp, ep2::StructAccessOp, ep2::ExtractOp, ep2::EmitOp>();

  // apply rules
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<CallPattern, ReturnPattern, ControllerPattern, EmitPattern, StructUpdatePattern,
               ContextRefPattern, InitPattern, StructAccessPattern, TerminatePattern>(
      typeConverter, &getContext());
  patterns.add<ExtractPattern>(typeConverter, &getContext(),
                                allocAnalysis);
  patterns.add<ConstPattern>(typeConverter, &getContext(),
                                atomAnalysis);
  patterns.add<StorePattern>(typeConverter, &getContext(),
                                contextAnalysis);
  patterns.add<FunctionPattern>(typeConverter, &getContext(),
                                lowerStructAnalysis, handlerDepAnalysis, allocAnalysis);

  FrozenRewritePatternSet patternSet(std::move(patterns));

  if (failed(applyPartialConversion(getOperation(), target, patternSet)))
    signalPassFailure();
}

} // namespace ep2
} // namespace mlir
