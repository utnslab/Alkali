#include "mlir/IR/BuiltinDialect.h"

#include "ep2/lang/Lexer.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

#include "Utils.h"

// TODO support nested struct/context read/write, e.g. ctx.s.v1 = 1;

/*
In this pass, we emit lots of intrinsics. We document some of the arguments
to intrinsics for clarity here, most of the intrinsics' names are
self-explanatory. More documentation is in TranslateToCpp.cpp file.
*/

namespace mlir {
namespace ep2 {

namespace {

static mlir::Operation* getParentFunction(mlir::Operation* op) {
  while (!isa<FunctionOpInterface>(op)) {
    op = op->getParentOp();
    assert(op != nullptr);
  }
  return op;
}

static unsigned calcSize(mlir::Type ty) {
  if (isa<mlir::IntegerType>(ty)) {
    assert(ty.isIntOrFloat());
    return ty.getIntOrFloatBitWidth()/8;
  } else if (isa<ep2::StructType>(ty)) {
    unsigned size = 0;
    for (auto ety : cast<ep2::StructType>(ty).getElementTypes()) {
      assert(ety.isIntOrFloat());
      size += ety.getIntOrFloatBitWidth()/8;
    }
    return size;
  }
}

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
      size_t v = analyzer.atomToNum[initOp.getValue().cast<mlir::StringAttr>().getValue()].second;
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

static mlir::Type addClsToEmitcPtrTy(mlir::MLIRContext* ctx, mlir::Type ty) {
  if (isa<emitc::PointerType>(ty)) {
    std::string newTy = std::string{"__shared __cls "} + cast<emitc::OpaqueType>(cast<emitc::PointerType>(ty).getPointee()).getValue().str();
    return emitc::PointerType::get(ctx, emitc::OpaqueType::get(ctx, newTy));
  } else {
    return ty;
  }
}

/*
Pattern: get a reference to allocated struct, if a reference type, and generate a
memcpy intrinsic from the context to that local struct. Else just read out of the
context, and let the Netronome compiler generate code.
*/
struct LoadPattern : public OpConversionPattern<ep2::LoadOp> {
  ContextBufferizationAnalysis &analyzer;
  LocalAllocAnalysis &allocAnalyzer;

  LoadPattern(TypeConverter &converter, MLIRContext *context,
                  ContextBufferizationAnalysis &analyzer, LocalAllocAnalysis &allocAnalysis)
      : OpConversionPattern<ep2::LoadOp>(converter, context),
        analyzer(analyzer), allocAnalyzer(allocAnalysis) {}

  LogicalResult matchAndRewrite(ep2::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = loadOp->getLoc();
    auto refOp = dyn_cast<ep2::ContextRefOp>(loadOp->getOperand(0).getDefiningOp());
    auto contextId = rewriter.getRemappedValue(refOp.getOperand());

    llvm::SmallVector<Type> resTypes = {addClsToEmitcPtrTy(getContext(), typeConverter->convertType(loadOp->getResult(0).getType()))};
    int pos = analyzer.getContextType(cast<func::FuncOp>(getParentFunction(loadOp)), refOp.getName()).first;

    /*
    Arg0: context field position, ctx->f{pos}
    Arg1: are we reading into a reference type? If so, return type
          of ctx_read should be a POINTER to context. Actual memcpy will
          do copy from context.
    */

    mlir::ArrayAttr args = rewriter.getI32ArrayAttr({pos, isa<ep2::StructType>(loadOp->getResult(0).getType())});
    mlir::ArrayAttr templ_args;

    auto ctxV = rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_intrin_ctx_read"), args, templ_args, ValueRange{contextId});

    if (isa<ep2::StructType>(loadOp->getResult(0).getType())) {
      auto resType = loadOp->getResult(0).getType();
      assert(allocAnalyzer.localAllocs.find(loadOp) != allocAnalyzer.localAllocs.end());

      int memcpySize = calcSize(resType);
      auto varOp = rewriter.create<emitc::VariableOp>(loc, typeConverter->convertType(resType), emitc::OpaqueAttr::get(getContext(), std::string{"&"} + allocAnalyzer.localAllocs[loadOp]));
      rewriter.replaceOp(loadOp, varOp);

      /*
      Arg0: offset in transfer register to start copying to
      Arg1: size of copy
      Arg2: tag.
        0 means we are copying FROM memory, TO local buffer.
        1 means we are copying TO memory, FROM local buffer.
        2 means we are copying FROM memory, TO memory.
      */

      llvm::SmallVector<Type> resTypes2 = {};
      mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({0, memcpySize, 0});
      mlir::ArrayAttr templ_args2;
      rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_memcpy"), args2, templ_args2, ValueRange{varOp, ctxV.getResult(0)});
    } else {
      rewriter.replaceOp(loadOp, ctxV);
    }
    return success();
  }
};

struct StorePattern : public OpConversionPattern<ep2::StoreOp> {
  ContextBufferizationAnalysis &analyzer;
  LocalAllocAnalysis &allocAnalyzer;

  StorePattern(TypeConverter &converter, MLIRContext *context,
                  ContextBufferizationAnalysis &analyzer, LocalAllocAnalysis &allocAnalysis)
      : OpConversionPattern<ep2::StoreOp>(converter, context),
        analyzer(analyzer), allocAnalyzer(allocAnalysis) {}

  LogicalResult matchAndRewrite(ep2::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = storeOp->getLoc();
    auto refOp = dyn_cast<ep2::ContextRefOp>(storeOp->getOperand(0).getDefiningOp());
    auto contextId = rewriter.getRemappedValue(refOp.getOperand());

    mlir::Type vTy = storeOp->getOperand(1).getType();
    int pos = analyzer.getContextType(cast<func::FuncOp>(getParentFunction(storeOp)), refOp.getName()).first;

    if (isa<ep2::StructType>(vTy)) {
      llvm::SmallVector<Type> resTypes = {addClsToEmitcPtrTy(getContext(), typeConverter->convertType(vTy))};
      mlir::ArrayAttr args = rewriter.getI32ArrayAttr({pos, isa<ep2::StructType>(vTy)});
      mlir::ArrayAttr templ_args;

      auto ctxV = rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_intrin_ctx_read"), args, templ_args, ValueRange{contextId});

      // get reference to allocated buffer
      // generate memcpy into it.
      auto resType = vTy;
      assert(allocAnalyzer.localAllocs.find(storeOp) != allocAnalyzer.localAllocs.end());
      int memcpySize = calcSize(resType);
      auto varOp = rewriter.create<emitc::VariableOp>(loc, typeConverter->convertType(resType), emitc::OpaqueAttr::get(getContext(), std::string{"&"} + allocAnalyzer.localAllocs[storeOp]));

      llvm::SmallVector<Type> resTypes2 = {};

      // Same encoding as memcpy.
      mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({0, memcpySize, 1});
      mlir::ArrayAttr templ_args2;
      auto memcpyOp = rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_memcpy"), args2, templ_args2, ValueRange{ctxV.getResult(0), varOp});
      rewriter.replaceOp(storeOp, memcpyOp);
    } else {
      llvm::SmallVector<Type> resTypes = {};

      /*
      Same encoding as ctx_read- position, whether a reference type or not.
      Instead of ctx_read, where we take address of result if a reference type,
        here we dereference the incoming operand if a reference type, to store it.
      */
      mlir::ArrayAttr args = rewriter.getI32ArrayAttr({pos, isa<ep2::StructType>(vTy)});
      mlir::ArrayAttr templ_args;
      auto wr = rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_intrin_ctx_write"), args, templ_args, ValueRange{adaptor.getValue(), contextId});
      rewriter.replaceOp(storeOp, wr);
    }
    return success();
  }
};


struct ReturnPattern : public OpConversionPattern<ep2::ReturnOp> {
  using OpConversionPattern<ep2::ReturnOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ep2::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isa<func::FuncOp>(getParentFunction(returnOp)))
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

    // Same encoding as ctx_read intrinsic.
    mlir::ArrayAttr args = rewriter.getI32ArrayAttr({(uint32_t) accessOp.getIndex(), isa<ep2::StructType>(accessOp->getResult(0).getType())});
    mlir::ArrayAttr templ_args;
    auto load = rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_intrin_struct_access"), args, templ_args, ValueRange{adaptor.getOperands()[0]});
    rewriter.replaceOp(accessOp, load);
    return success();
  }
};

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

    assert(analyzer.localAllocs.find(extractOp) != analyzer.localAllocs.end());

    int memcpySize = calcSize(resType);
    auto varOp = rewriter.create<emitc::VariableOp>(loc, typeConverter->convertType(resType), emitc::OpaqueAttr::get(getContext(), std::string{"&"} + analyzer.localAllocs[extractOp]));
    rewriter.replaceOp(extractOp, varOp);

    llvm::SmallVector<Type> resTypes2 = {};
    // Same encoding as LoadOp.
    mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({0, memcpySize, 0});
    mlir::ArrayAttr templ_args2;
    rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_memcpy"), args2, templ_args2, ValueRange{varOp, adaptor.getBuffer()});
    return success();
  }
};

struct ExtractOffsetPattern : public OpConversionPattern<ep2::ExtractOffsetOp> {
  using OpConversionPattern<ep2::ExtractOffsetOp>::OpConversionPattern;

  LocalAllocAnalysis &analyzer;

  ExtractOffsetPattern(TypeConverter &converter, MLIRContext *context,
                  LocalAllocAnalysis &analyzer)
      : OpConversionPattern<ep2::ExtractOffsetOp>(converter, context),
        analyzer(analyzer) {}

  LogicalResult
  matchAndRewrite(ep2::ExtractOffsetOp extractOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = extractOp->getLoc();
    auto resType = extractOp.getResult().getType();
    if (!resType.isa<ep2::StructType>())
      return rewriter.notifyMatchFailure(extractOp, "Currently only support extract op on struct");

    assert(analyzer.localAllocs.find(extractOp) != analyzer.localAllocs.end());

    int memcpySize = calcSize(resType);
    auto varOp = rewriter.create<emitc::VariableOp>(loc, typeConverter->convertType(resType), emitc::OpaqueAttr::get(getContext(), std::string{"&"} + analyzer.localAllocs[extractOp]));
    rewriter.replaceOp(extractOp, varOp);

    llvm::SmallVector<Type> resTypes2 = {};

    /*
    Same encoding as regular ExtractOp, except we add an additional arg (arg3)
    specifying an offset. Regular ExtractOp keeps a dynamic offset, ExtractOffsetOp
    statically encodes it.
    */
    mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({0, memcpySize, 0, extractOp.getOffset() / 8});
    mlir::ArrayAttr templ_args2;
    rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_memcpy"), args2, templ_args2, ValueRange{varOp, adaptor.getBuffer()});
    return success();
  }
};

struct EmitPattern : public OpConversionPattern<ep2::EmitOp> {
  using OpConversionPattern<ep2::EmitOp>::OpConversionPattern;

  EmitPattern(TypeConverter &converter, MLIRContext *context)
      : OpConversionPattern<ep2::EmitOp>(converter, context) {}

  LogicalResult
  matchAndRewrite(ep2::EmitOp emitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = emitOp->getLoc();
    auto resType = emitOp.getValue().getType();
    int memcpySize = isa<ep2::BufferType>(resType) ? 0 : calcSize(resType);
    unsigned srcOffs = isa<ep2::BufferType>(resType) ? (-memcpySize) : 0;

    llvm::SmallVector<Type> resTypes2 = {};

    /*
    Arg0: offset in target buffer. Still 0 usually, but in buf-to-buf copy, we use
          buffer size minus srcOffs to get the starting point.
    Arg1: size of copy
    Arg2: tag, like usual.
    */
    mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({srcOffs, memcpySize, isa<ep2::BufferType>(resType) ? 2 : 1});
    mlir::ArrayAttr templ_args2;
    auto emit = rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_memcpy"), args2, templ_args2, ValueRange{adaptor.getBuffer(), adaptor.getValue()});
    rewriter.replaceOp(emitOp, emit);
    return success();
  }
};

struct EmitOffsetPattern : public OpConversionPattern<ep2::EmitOffsetOp> {
  using OpConversionPattern<ep2::EmitOffsetOp>::OpConversionPattern;

  EmitOffsetPattern(TypeConverter &converter, MLIRContext *context)
      : OpConversionPattern<ep2::EmitOffsetOp>(converter, context) {}

  LogicalResult
  matchAndRewrite(ep2::EmitOffsetOp emitOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = emitOp->getLoc();
    auto resType = emitOp.getValue().getType();
    int memcpySize = isa<ep2::BufferType>(resType) ? 0 : calcSize(resType);
    unsigned srcOffs = isa<ep2::BufferType>(resType) ? (-memcpySize) : 0;

    llvm::SmallVector<Type> resTypes2 = {};
    // Same change from EmitOp, as ExtractOp -> ExtractOffsetOp.
    mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({srcOffs, memcpySize, isa<ep2::BufferType>(resType) ? 2 : 1, emitOp.getOffset() / 8});
    mlir::ArrayAttr templ_args2;
    auto emit = rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_memcpy"), args2, templ_args2, ValueRange{adaptor.getBuffer(), adaptor.getValue()});
    rewriter.replaceOp(emitOp, emit);
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

    // Same encoding as ctx_write, ctx_read, struct_access_op
    mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({updateOp.getIndex(), isa<ep2::StructType>(updateOp->getOperand(1).getType())});
    mlir::ArrayAttr templ_args2;
    auto callOp = rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_struct_write"), args2, templ_args2, ValueRange{adaptor.getNewValue(), adaptor.getInput()});

    rewriter.replaceOp(updateOp, callOp);
    return success();
  }
};

struct InitPattern : public OpConversionPattern<ep2::InitOp> {
  ContextBufferizationAnalysis &analyzer;
  LocalAllocAnalysis &allocAnalyzer;

  InitPattern(TypeConverter &converter, MLIRContext *context, ContextBufferizationAnalysis& cba, LocalAllocAnalysis& analysis)
      : OpConversionPattern<ep2::InitOp>(converter, context), analyzer(cba), allocAnalyzer(analysis) {}

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
      std::string eventName = resType.cast<ep2::StructType>().getName().str();

      if (!cast<ep2::StructType>(resType).getIsEvent()) {
        // Regular struct initialization.
        auto alloc = rewriter.create<emitc::VariableOp>(loc, newType, emitc::OpaqueAttr::get(getContext(), std::string{"&"} + allocAnalyzer.localAllocs[initOp]));

        unsigned p = 0;
        for (const auto& opd : adaptor.getOperands()) {
          llvm::SmallVector<Type> resTypes2 = {};
          mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({p, isa<ep2::StructType>(initOp->getOperand(p).getType())});
          mlir::ArrayAttr templ_args2;
          rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_struct_write"), args2, templ_args2, ValueRange{opd, alloc});
          p += 1;
        }
        rewriter.replaceOp(initOp, alloc);
        return success();
      }

      // Alignment required for inlined_net_recv/send handlers.
      auto newTypeAligned =
          rewriter.getType<emitc::PointerType>(rewriter.getType<emitc::OpaqueType>(std::string{"__declspec(aligned(4)) struct event_param_" + eventName}));
      auto alloc = rewriter.create<emitc::VariableOp>(loc, newTypeAligned, emitc::OpaqueAttr::get(getContext(), "&next_work_" + eventName));

      /*
      Initialize like usual- first arg is atom, which we discard- TODO fix this
      Context is 2nd arg in EP2 init -> map it to named ctx variable.
      Then the actual args.
      */
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
          args2 = rewriter.getI32ArrayAttr({p-2, isa<ep2::StructType>(initOp->getOperand(p).getType())});
        }
        mlir::ArrayAttr templ_args2;
        rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_struct_write"), args2, templ_args2, ValueRange{opd, alloc});
        p += 1;
      }

      /*
      Implement Netronome-specific logic to push next work item to event queue.
      (OR, for NET_SEND event, just call inlined_net_send
      1. move work to work_ref (xrw allocation)
      2. cls_workq_add_work(&next_work_ref_EV_NAME);
      */
      mlir::Type retType = initOp->getResult(0).getType();
      if (isa<ep2::StructType>(retType) && cast<ep2::StructType>(retType).getIsEvent()) {
        ep2::StructType structTy = cast<ep2::StructType>(retType);
        // an event generation.
        if (structTy.getName() == "NET_SEND") {
          llvm::SmallVector<Type> resTypes3 = {};
          mlir::ArrayAttr args3;
          mlir::ArrayAttr templ_args3;

          /*
          Set size of packet. TODO, right now hardcoded as incoming packet's size
          for zero-copy optimization. Should adapt, if using zero-copy, use .sz,
          else use .offs. This choice happens in TranslateToCpp code.
          */
          rewriter.create<emitc::CallOp>(loc, resTypes3, rewriter.getStringAttr("__ep2_intrin_pkt_size_set"), args3, templ_args3, ValueRange{});
          rewriter.create<emitc::CallOp>(loc, resTypes3, rewriter.getStringAttr("inlined_net_send"), args3, templ_args3, ValueRange{alloc});
        } else {
          auto xferType =
              rewriter.getType<emitc::PointerType>(rewriter.getType<emitc::OpaqueType>(std::string{"__xrw struct event_param_" + eventName}));

          auto eventXfer = rewriter.create<emitc::VariableOp>(loc, xferType, emitc::OpaqueAttr::get(getContext(), std::string{"&next_work_ref_"} + eventName));

          llvm::SmallVector<Type> resTypes = {};
          mlir::ArrayAttr args;
          mlir::ArrayAttr templ_args;
          auto copy = rewriter.create<emitc::CallOp>(loc, resTypes, rewriter.getStringAttr("__ep2_intrin_gpr2xfer"), args, templ_args, ValueRange{alloc, eventXfer});

          /*
          Because of handler replication, not as simple as 1 cls_workq_add_work
          command. If we partition on hash of some key among different queues,
          or round-robin, need to emit a dispatch() call. We fill out dispatch() call
          in EmitNetronomePass.cpp.
          */
          {
            std::string outputQStr;
            int ctxFieldPos = -1;

            if (initOp->hasAttr("enqInfo")) {
              std::string sprayInfo = initOp->getAttr("sprayInfo").cast<mlir::StringAttr>().getValue().str();
              std::string sprayMethod = sprayInfo.substr(0, sprayInfo.find(" "));
              if (sprayMethod == "PARTITION") {
                std::string partitionKey = sprayInfo.substr(sprayInfo.find(" ")+1);
                ctxFieldPos = analyzer.getContextType(cast<func::FuncOp>(getParentFunction(initOp)), partitionKey).first;
              }

              llvm::ArrayRef<mlir::Attribute> outputQueues = initOp->getAttr("enqInfo").cast<mlir::ArrayAttr>().getValue();
              for (auto q : outputQueues) {
                outputQStr += std::to_string(cast<mlir::IntegerAttr>(q).getValue().getSExtValue());
                outputQStr += ' ';
              }
            }

            llvm::SmallVector<Type> resTypes3 = {};
            mlir::ArrayAttr args3 = rewriter.getStrArrayAttr({eventName, outputQStr, std::to_string(ctxFieldPos)});
            mlir::ArrayAttr templ_args3;
            rewriter.create<emitc::CallOp>(loc, resTypes3, rewriter.getStringAttr("__ep2_intrin_enq_work"), args3, templ_args3, ValueRange{eventXfer.getResult()});
          }
        }
      }

      rewriter.replaceOp(initOp, alloc);
    } else if (resType.isa<ep2::BufferType>()) {
      /*
      TODO note buffer allocation works only when buf is its own field.
      If it is a struct member, will not work.
      */
      auto newType = typeConverter->convertType(resType);
      auto varOp = rewriter.create<emitc::VariableOp>(loc, newType, emitc::OpaqueAttr::get(getContext(), std::string{"alloc_packet_buf()"}));
      rewriter.replaceOp(initOp, varOp);
    } else if (resType.isa<ep2::TableType>()) {
      auto newType = typeConverter->convertType(resType);
      auto varOp = rewriter.create<emitc::VariableOp>(loc, newType, emitc::OpaqueAttr::get(getContext(), std::string{"&"} + allocAnalyzer.localAllocs[initOp]));
      rewriter.replaceOp(initOp, varOp);
    } else {
      /*
      TODO fix this, just declare a dummy variable for convenience.
      This code is completely wrong, but shouldn't matter.
      */
      auto newType = typeConverter->convertType(resType);
      auto varOp = rewriter.create<emitc::VariableOp>(loc, newType, emitc::OpaqueAttr::get(getContext(), std::string{"rr_ctr"}));
      rewriter.replaceOp(initOp, varOp);
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

struct NopPattern : public OpConversionPattern<ep2::NopOp> {
  using OpConversionPattern<ep2::NopOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ep2::NopOp nopOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(nopOp);
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

struct BitCastPattern : public OpConversionPattern<ep2::BitCastOp> {
  using OpConversionPattern<ep2::BitCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::BitCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, typeConverter->convertType(op->getResult(0).getType()), adaptor.getOperands());
    return success();
  }
};

struct SubPattern : public OpConversionPattern<ep2::SubOp> {
  using OpConversionPattern<ep2::SubOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::SubOp>(op, typeConverter->convertType(op->getResult(0).getType()), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct AddPattern : public OpConversionPattern<ep2::AddOp> {
  using OpConversionPattern<ep2::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::AddOp>(op, typeConverter->convertType(op->getResult(0).getType()), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct MulPattern : public OpConversionPattern<ep2::MulOp> {
  using OpConversionPattern<ep2::MulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::MulOp>(op, typeConverter->convertType(op->getResult(0).getType()), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct CmpPattern : public OpConversionPattern<ep2::CmpOp> {
  using OpConversionPattern<ep2::CmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    emitc::CmpPredicate cmpType;
    switch (op.getPredicate()) {
      case -::ep2::tok_cmp_eq:
        cmpType = emitc::CmpPredicate::eq;
        break;
      case -::ep2::tok_cmp_le:
        cmpType = emitc::CmpPredicate::le;
        break;
      case -::ep2::tok_cmp_ge:
        cmpType = emitc::CmpPredicate::ge;
        break;
      case '<':
        cmpType = emitc::CmpPredicate::lt;
        break;
      case '>':
        cmpType = emitc::CmpPredicate::gt;
        break;
      default: {
        assert(false && "Unsupported comparison");
        break;
      }
    }
    auto attr = emitc::CmpPredicateAttr::get(getContext(), cmpType);
    rewriter.replaceOpWithNewOp<emitc::CmpOp>(op, typeConverter->convertType(op->getResult(0).getType()), attr, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct SelectPattern : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Type> resTypes = {typeConverter->convertType(op->getResult(0).getType())};
    mlir::ArrayAttr args;
    mlir::ArrayAttr templ_args;
    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, resTypes, rewriter.getStringAttr("__ep2_intrin_ternary"), args, templ_args, adaptor.getOperands());
    return success();
  }
};

struct BranchPattern : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern<cf::BranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, op.getDest(), adaptor.getDestOperands());
    return success();
  }
};

struct CondBranchPattern : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern<cf::CondBranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(op, adaptor.getCondition(), op.getTrueDest(), adaptor.getTrueDestOperands(), op.getFalseDest(), adaptor.getFalseDestOperands());
    return success();
  }
};

struct GlobalImportPattern : public OpConversionPattern<ep2::GlobalImportOp> {
  using OpConversionPattern<ep2::GlobalImportOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ep2::GlobalImportOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto newType = typeConverter->convertType(op.getOutput().getType());
    rewriter.replaceOpWithNewOp<emitc::VariableOp>(op, newType, emitc::OpaqueAttr::get(getContext(), std::string{"&"} + op.getName().str()));
    return success();
  }
};

struct TableLookupPattern : public OpConversionPattern<ep2::LookupOp> {
  using OpConversionPattern<ep2::LookupOp>::OpConversionPattern;

  LocalAllocAnalysis &allocAnalyzer;

  TableLookupPattern(TypeConverter &converter, MLIRContext *context,
                  LocalAllocAnalysis &allocAnalysis)
      : OpConversionPattern<ep2::LookupOp>(converter, context), allocAnalyzer(allocAnalysis) {}

  // T[me_cam_lookup(k)]
  LogicalResult
  matchAndRewrite(ep2::LookupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    if (isa<ep2::StructType>(op.getValue().getType())) {
      auto buf = rewriter.create<emitc::VariableOp>(op->getLoc(), typeConverter->convertType(op.getValue().getType()), emitc::OpaqueAttr::get(getContext(), std::string{"&"} + allocAnalyzer.localAllocs[op]));

      llvm::SmallVector<Type> resTypes = {};
      // Arg0 is whether the table element we are reading out is a reference type.
      mlir::ArrayAttr args = rewriter.getI32ArrayAttr({true, op.getTable().getType().getSize()});
      mlir::ArrayAttr templ_args;
      rewriter.create<emitc::CallOp>(op->getLoc(), resTypes, rewriter.getStringAttr("__ep2_intrin_table_lookup"), args, templ_args, ValueRange{adaptor.getTable(), adaptor.getKey(), buf->getResult(0)});

      rewriter.replaceOp(op, buf);
    } else {
      llvm::SmallVector<Type> resTypes = {typeConverter->convertType(op.getValue().getType())};
      mlir::ArrayAttr args = rewriter.getI32ArrayAttr({false, op.getTable().getType().getSize()});
      mlir::ArrayAttr templ_args;
      rewriter.replaceOpWithNewOp<emitc::CallOp>(op, resTypes, rewriter.getStringAttr("__ep2_intrin_table_lookup"), args, templ_args, ValueRange{adaptor.getOperands()});
    }

    return success();
  }
};

struct TableUpdatePattern : public OpConversionPattern<ep2::UpdateOp> {
  using OpConversionPattern<ep2::UpdateOp>::OpConversionPattern;
  // T[me_cam_update(k)] = v

  LogicalResult
  matchAndRewrite(ep2::UpdateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::SmallVector<Type> resTypes = {};
    // Same arg pattern as table_lookup op
    mlir::ArrayAttr args = rewriter.getI32ArrayAttr({isa<ep2::StructType>(op.getValue().getType()), op.getTable().getType().getSize()});
    mlir::ArrayAttr templ_args;
    rewriter.replaceOpWithNewOp<emitc::CallOp>(op, resTypes, rewriter.getStringAttr("__ep2_intrin_table_update"), args, templ_args, adaptor.getOperands());
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
    std::string eventName = funcOp->getAttr("event").cast<StringAttr>().getValue().str();

    if (funcOp->getAttr("type").cast<StringAttr>().getValue() != "handler")
      return rewriter.notifyMatchFailure(funcOp, "Not a handler");

    // rewrite a function call.
    ArrayRef<LLVM::LLVMStructType> wrapperTypes = analyzer.getWrapperTypes(funcOp);
    if (wrapperTypes.size() == 0) {
      funcOp.emitError("Cannot rewrite with valid input");
      return failure();
    }

    // no args
    llvm::SmallVector<mlir::Type> newArgTypes;
    TypeConverter::SignatureConversion signatureConversion(
        funcOp.getNumArguments());

    auto newFuncOp = rewriter.create<func::FuncOp>(
        loc, std::string{"__event_" + funcOp.getName().str()},
        rewriter.getFunctionType(TypeRange(newArgTypes), TypeRange{}));

    if (funcOp->hasAttr("location")) {
      newFuncOp->setAttr("location", funcOp->getAttr("location"));
    }
    if (funcOp->hasAttr("atom")) {
      newFuncOp->setAttr("atom", funcOp->getAttr("atom"));
    }
    
    auto entryBlock = newFuncOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);

    // construct body and replace parameter
    auto inputWrapperXferType =
        rewriter.getType<emitc::PointerType>(rewriter.getType<emitc::OpaqueType>(std::string{"__xrw struct event_param_" + eventName}));
    auto inputWrapperType =
        rewriter.getType<emitc::PointerType>(rewriter.getType<emitc::OpaqueType>(std::string{"__declspec(aligned(4)) struct event_param_" + eventName}));
    auto contextType =
        rewriter.getType<emitc::PointerType>(rewriter.getType<emitc::OpaqueType>(std::string{"__shared __cls struct context_chain_1_t"}));

    auto eventPtr = rewriter.create<emitc::VariableOp>(loc, inputWrapperType, emitc::OpaqueAttr::get(getContext(), std::string{"&work"}));
    if (eventName != "NET_RECV") {
      auto eventXfer = rewriter.create<emitc::VariableOp>(loc, inputWrapperXferType, emitc::OpaqueAttr::get(getContext(), std::string{"&work_ref"}));

      {
        /*
        Every ME has 1 dedicated work queue. Hence, just call deq_work intrinsic for the
        queue corresponding to our replica id.
        */
        auto getReplicaId = [](std::string name) {
          std::string id = name.substr(1 + name.rfind("_"));
          for (int i = 0; i<id.size(); ++i) {
            if (!isdigit(id[i])) {
              return std::string{""};
            }
          }
          return id;
        };

        llvm::SmallVector<Type> resTypes3 = {};
        /*
        Arg0: event class to spray across
        Arg1: list of queues to spray over, encoded in a string.
        */
        mlir::ArrayAttr args3 = rewriter.getStrArrayAttr({eventName, getReplicaId(funcOp.getName().str())});
        mlir::ArrayAttr templ_args3;
        rewriter.create<emitc::CallOp>(loc, resTypes3, rewriter.getStringAttr("__ep2_intrin_deq_work"), args3, templ_args3, ValueRange{eventXfer.getResult()});
      }
      {
        llvm::SmallVector<Type> resTypes3 = {};
        mlir::ArrayAttr args3 = rewriter.getStrArrayAttr({});
        mlir::ArrayAttr templ_args3;
        rewriter.create<emitc::CallOp>(loc, resTypes3, rewriter.getStringAttr("__ep2_intrin_xfer2gpr"), args3, templ_args3, ValueRange{eventXfer.getResult(), eventPtr.getResult()});
      }
    } else {
      llvm::SmallVector<Type> resTypes3 = {};
      mlir::ArrayAttr args3;
      mlir::ArrayAttr templ_args3;
      rewriter.create<emitc::CallOp>(loc, resTypes3, rewriter.getStringAttr("inlined_net_recv"), args3, templ_args3, ValueRange{eventPtr.getResult()});
    }

    bool isFirstStage = true;
    for (const auto& pr : handAnalyzer.eventDeps) {
      for (const auto& s : pr.second) {
        if (s == eventName) {
          isFirstStage = false;
        }
      }
    }
    if (isFirstStage) {
      // initialize context field at start of pipeline.
      llvm::SmallVector<Type> resTypes3 = {contextType};
      mlir::ArrayAttr args3;
      mlir::ArrayAttr templ_args3;
      auto ctxAlloc = rewriter.create<emitc::CallOp>(loc, resTypes3, rewriter.getStringAttr("alloc_context_chain_ring_entry"), args3, templ_args3, ValueRange{});

      llvm::SmallVector<Type> resTypes4 = {};
      mlir::ArrayAttr args4 = rewriter.getStrArrayAttr({"ctx"});
      mlir::ArrayAttr templ_args4;
      rewriter.create<emitc::CallOp>(loc, resTypes4, rewriter.getStringAttr("__ep2_intrin_struct_write"), args4, templ_args4, ValueRange{ctxAlloc->getResult(0), eventPtr.getResult()});
    }

    int sourceIdx = 0;
    if (isa<ep2::ContextType>(funcOp.getFunctionType().getInputs()[0])) {
      llvm::SmallVector<Type> resTypes3 = {contextType};
      mlir::ArrayAttr args3 = rewriter.getStrArrayAttr({"ctx"});
      mlir::ArrayAttr templ_args3;
      auto structPtr = rewriter.create<emitc::CallOp>(loc, resTypes3, rewriter.getStringAttr("__ep2_intrin_struct_access"), args3, templ_args3, ValueRange{eventPtr.getResult()});
      signatureConversion.remapInput(0, structPtr.getResult(0));
      sourceIdx += 1;
    }

    for (size_t i = 0; i < wrapperTypes[0].getBody().size(); i++) {
      auto convertedType =
          typeConverter->convertType(wrapperTypes[0].getBody()[i]);
      auto elementPtrType =
          rewriter.getType<emitc::PointerType>(convertedType);
      // materialize block type
      llvm::SmallVector<Type> resTypes2 = {convertedType};
      mlir::ArrayAttr args2 = rewriter.getI32ArrayAttr({i, isa<ep2::StructType>(wrapperTypes[0].getBody()[i])});
      mlir::ArrayAttr templ_args2;
      auto param = rewriter.create<emitc::CallOp>(loc, resTypes2, rewriter.getStringAttr("__ep2_intrin_struct_access"), args2, templ_args2, ValueRange{eventPtr.getResult()});
      signatureConversion.remapInput(i + sourceIdx, param.getResult(0));
    }

    // change the function body
    auto res = rewriter.convertRegionTypes(&funcOp.getBody(), *typeConverter,
                                           &signatureConversion);
    if (failed(res))
      return failure();
    rewriter.mergeBlocks(*res, entryBlock);

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());

    // remove original func
    rewriter.eraseOp(funcOp);

    return success();
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
  TableAnalysis &tableAnalysis = getAnalysis<TableAnalysis>();
  const CollectInfoAnalysis& info = getCachedAnalysis<CollectInfoAnalysis>().value();

  markAnalysesPreserved<LocalAllocAnalysis>();

  // install functions
  auto builder = OpBuilder(getOperation());

  getOperation()->walk([&](mlir::Operation* op) {
    if (isa<ep2::InitOp>(op) && op->hasOneUse()) {
      for (mlir::Operation* user : op->getUsers()) {
        if (isa<ep2::ReturnOp>(user)) {
          op->moveBefore(user);
        }
      }
    }
  });

  // Type conversion must generate types including memory hierarchy placement

  // Dialect Type converter
  TypeConverter typeConverter;
  // All other types are valid
  // conversion:: 1 to 1 type mapping
  typeConverter.addConversion([](Type type) { return type; });
  // TODO add an attribute to ep2::ContextType to know which context_chain_t to convert to.
  typeConverter.addConversion([&](ep2::ContextType type) {
    return emitc::PointerType::get(type.getContext(), builder.getType<emitc::OpaqueType>("__shared __cls struct context_chain_1_t"));
  });
  typeConverter.addConversion([&](ep2::BufferType type) {
    return builder.getType<emitc::OpaqueType>("struct __buf_t");
  });

  typeConverter.addConversion([&](ep2::TableType type) {
    TableInfo tInfo = getTableStr(type);
    std::string qualifier;
    if (info.tableInfos.find(tInfo.tableType)->second.first.isLocal) {
      qualifier = "__shared __lmem ";
    } else {
      qualifier = "__export __shared __cls ";
    }
    return builder.getType<emitc::PointerType>(builder.getType<emitc::OpaqueType>(qualifier + tInfo.tableType));
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

  target.addIllegalOp<ep2::ConstantOp, ep2::StoreOp, ep2::ContextRefOp, ep2::TerminateOp, 
    ep2::CallOp, ep2::FuncOp, ep2::ReturnOp, ep2::StructUpdateOp, ep2::NopOp, ep2::InitOp,
    ep2::StructAccessOp, ep2::ExtractOp, ep2::EmitOp, ep2::BitCastOp, ep2::SubOp, ep2::AddOp,
    ep2::CmpOp, ep2::MulOp, arith::SelectOp, ep2::LookupOp, ep2::UpdateOp, ep2::LoadOp,
    ep2::ExtractOffsetOp, ep2::EmitOffsetOp, ep2::GlobalImportOp>();
  target.addDynamicallyLegalDialect<cf::ControlFlowDialect>([&](mlir::Operation* op){
    return typeConverter.isLegal(op);
  });

  // apply rules
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<CallPattern, ReturnPattern, ControllerPattern, StructUpdatePattern, EmitPattern,
               ContextRefPattern, StructAccessPattern, TerminatePattern, NopPattern, BitCastPattern,
               SubPattern, AddPattern, MulPattern, CmpPattern, SelectPattern, EmitOffsetPattern,
               TableUpdatePattern, BranchPattern, CondBranchPattern, GlobalImportPattern>(typeConverter, &getContext());
  patterns.add<ExtractPattern>(typeConverter, &getContext(),
                                allocAnalysis);
  patterns.add<ExtractOffsetPattern>(typeConverter, &getContext(),
                                allocAnalysis);
  patterns.add<ConstPattern>(typeConverter, &getContext(),
                                atomAnalysis);
  patterns.add<InitPattern>(typeConverter, &getContext(),
                                contextAnalysis, allocAnalysis);
  patterns.add<LoadPattern>(typeConverter, &getContext(),
                                contextAnalysis, allocAnalysis);
  patterns.add<TableLookupPattern>(typeConverter, &getContext(),
                                allocAnalysis);
  patterns.add<StorePattern>(typeConverter, &getContext(),
                                contextAnalysis, allocAnalysis);
  patterns.add<FunctionPattern>(typeConverter, &getContext(),
                                lowerStructAnalysis, handlerDepAnalysis, allocAnalysis);

  FrozenRewritePatternSet patternSet(std::move(patterns));

  if (failed(applyPartialConversion(getOperation(), target, patternSet)))
    signalPassFailure();
}

} // namespace ep2
} // namespace mlir
