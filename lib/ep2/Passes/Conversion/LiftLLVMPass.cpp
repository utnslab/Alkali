#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/EquivalenceClasses.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"

#include "polygeist/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

// Analysis pass for converting the pointer value to LLVM type
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/Passes.h"


namespace mlir {
namespace ep2 {

namespace {

using namespace dataflow;

struct StackSlotValue : public AbstractDenseLattice {
  llvm::DenseMap<Value, Value> updateTable{};
  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    auto rvalue = static_cast<const StackSlotValue &>(rhs);
    auto changed = ChangeResult::NoChange;

    for (auto &pair : rvalue.updateTable) {
        auto [it, inserted] = updateTable.try_emplace(pair.first, pair.second);
        if (inserted)
            changed = ChangeResult::Change;
    }

    return changed;
  }

  // print and lookup

  Value lookup(Value value) const {
    llvm::DenseMap<Value, Value>::const_iterator it;
    while ((it = updateTable.find(value)) != updateTable.end()) {
      if (it->first == it->second)
        break;
      value = it->second;
    }
    return value;
  }


  void print(llvm::raw_ostream &os) const override {
    os << "Dumping Dict...\n";
    for (auto &pair : updateTable) {
      os << "  ";
      pair.first.getLoc().print(os);
      os << " -> ";
      auto v = lookup(pair.second);
      if (v.getDefiningOp())
        v.getDefiningOp()->print(os);
      os << "\n";
    }
    os << "\n";
  }

  void printValue(llvm::raw_ostream &os, Value value) const {
    auto it = updateTable.find(value);
    if (it == updateTable.end())
      os << "Not a stack variable.";
    else {
      os << "definition = ";
      it->second.getDefiningOp()->print(os);
    }
  }
  
  // interfaces
  ChangeResult reset() {
    if (updateTable.empty())
      return ChangeResult::NoChange;
    updateTable.clear();
    return ChangeResult::Change;
  }
  void init(Value value, Value source) {
    auto [it, inserted] = updateTable.try_emplace(value, source);
    assert(inserted && "Value already initialized");
  }
  void init(Value value) { init(value, value); }

  ChangeResult update(Value value, Value newValue) {
    llvm::DenseMap<Value, Value>::iterator it;
    auto changed = ChangeResult::NoChange;
    while ((it = updateTable.find(value)) != updateTable.end()) {
      if (it->first == it->second) {
        it->second = newValue;
        changed = ChangeResult::Change;
        break;
      }
      value = it->second;
    }
    if (changed == ChangeResult::Change)
      updateTable.try_emplace(newValue, newValue);
    return changed;
  }
};

class StackVariableAnalysis : public DenseForwardDataFlowAnalysis<StackSlotValue> {
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;

  void visitOperation(Operation *op, const StackSlotValue &before,
                              StackSlotValue *after) override final {
    auto changed = after->join(before);

    auto opChanged =
        TypeSwitch<Operation *, ChangeResult>(op)
            .Case([&](memref::AllocaOp op) {
              after->init(op.getResult());
              return ChangeResult::Change;
            })
            .Case<polygeist::Memref2PointerOp, polygeist::Pointer2MemrefOp,
                  UnrealizedConversionCastOp>([&](Operation *op) {
              after->init(op->getResult(0), op->getOperand(0));
              return ChangeResult::Change;
            })
            .Case([&](ep2::AssignOp op) {
              after->update(op.getLhs(), op.getRhs());
              return ChangeResult::Change;
            })
            .Default([&](Operation *op) { return ChangeResult::NoChange; });

    propagateIfChanged(after, changed | opChanged);
  }

  // hook for external function
  void visitCallControlFlowTransfer(CallOpInterface call,
                                            CallControlFlowAction action,
                                            const StackSlotValue &before,
                                            StackSlotValue *after) override {
    AbstractDenseForwardDataFlowAnalysis::visitCallControlFlowTransfer(
        call, action, before, after);
  }

  void setToEntryState(StackSlotValue *state) override {
    propagateIfChanged(state, state->reset());
  }
};

struct Memref2PointerEliminate : public OpConversionPattern<polygeist::Memref2PointerOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(polygeist::Memref2PointerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {adaptor.getSource()});
    return success();
  }
};

struct Pointer2MemrefEliminate : public OpConversionPattern<polygeist::Pointer2MemrefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(polygeist::Pointer2MemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {adaptor.getSource()});
    return success();
  }
};

struct CallBufferPattern : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcName = op.getCallee();
    auto func = op->getParentOfType<ModuleOp>()
      .lookupSymbol<func::FuncOp>(funcName);
    if (func && !func.isDeclaration())
      return rewriter.notifyMatchFailure(op, "Not an external function on decl");

    auto loc = op.getLoc();
    if (funcName == "bufextract") {
      auto args = adaptor.getOperands();
      auto buf = args[0];
      auto header = args[1];

      rewriter.create<ep2::ExtractOp>(loc, header.getType(), buf);
      rewriter.replaceOpWithNewOp<ep2::NextOp>(op);
      return success();
    } else if (funcName == "bufemit") {
      auto args = adaptor.getOperands();
      auto buf = args[0];
      auto header = args[1];

      rewriter.replaceOpWithNewOp<ep2::EmitOp>(op, buf, header);
      return success();
    } else {
      // TODO: normal event calls. delete them for now.
      rewriter.eraseOp(op);
      return success();
    }

  }
};

Value castEP2Value(OpBuilder &builder, Value source, Type target) {
  auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(
      source.getLoc(), TypeRange{target}, ValueRange{source});
  return castOp.getResult(0);
}
Value castEP2Value(OpBuilder &builder, Value source) {
  return castEP2Value(builder, source, builder.getType<ep2::AnyType>());
}
Value castEP2Value(OpBuilder &builder, Value source, TypeConverter &converter) {
  auto newType = converter.convertType(source.getType());
  assert(newType && "Conversion Failure");
  return castEP2Value(builder, source, newType);
}

std::optional<int64_t> affineToIndex(AffineMap map) {
  if (!(map.isConstant() && map.getConstantResults().size() == 2))
    return std::nullopt;
  return map.getConstantResults()[1];
}

struct CallRewrite : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const final {
    auto funcName = op.getCallee();
    auto func = op->getParentOfType<ModuleOp>()
      .lookupSymbol<func::FuncOp>(funcName);
    if (func && !func.isDeclaration())
      return rewriter.notifyMatchFailure(op, "Not an external function on decl");

    auto loc = op.getLoc();
    auto args = op.getArgOperands();
    if (funcName == "bufextract") {
      auto buf =  castEP2Value(rewriter, args[0], rewriter.getType<ep2::BufferType>());
      auto header = castEP2Value(rewriter, args[1]);

      auto extractOp = rewriter.create<ep2::ExtractOp>(loc, header.getType(), buf);
      rewriter.replaceOpWithNewOp<ep2::AssignOp>(op, header, extractOp);
      return success();
    } else if (funcName == "bufemit") {
      auto buf =  castEP2Value(rewriter, args[0], rewriter.getType<ep2::BufferType>());
      auto header = castEP2Value(rewriter, args[1]);

      rewriter.replaceOpWithNewOp<ep2::EmitOp>(op, buf, header);
      return success();
    } else {
      // TODO: normal event calls. delete them for now.
      rewriter.eraseOp(op);
      return success();
    }

  }
};

struct StoreRewrite : public OpRewritePattern<affine::AffineStoreOp> {
  TypeConverter &converter;
  StoreRewrite(MLIRContext *context, TypeConverter &converter) :
    OpRewritePattern(context), converter(converter) {};

  LogicalResult matchAndRewrite(affine::AffineStoreOp op,
                                PatternRewriter &rewriter) const final {
    auto map = op.getMap();
    if (!(map.isConstant() && map.getConstantResults().size() == 2))
      return rewriter.notifyMatchFailure(op, "Not a 2d affine index for struct");

    auto memory = castEP2Value(rewriter, op.getMemRef(), converter);
    auto res = map.getConstantResults();

    auto updateOp = rewriter.create<ep2::StructUpdateOp>(op.getLoc(), memory.getType(), memory, res[1], op.getValueToStore());
    // avoid crush, use Operation * methods. see https://github.com/llvm/llvm-project/issues/39319
    rewriter.create<ep2::AssignOp>(op.getLoc(), memory, updateOp->getOpResults().front());
    rewriter.eraseOp(op);
    return success();
  }
};

struct FoldPolygeistCast : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return failure();
    Value source = op.getOperand(0);
    auto changed = failure();
    while (source.getDefiningOp() &&
           isa<polygeist::Memref2PointerOp, polygeist::Pointer2MemrefOp>(
               source.getDefiningOp())) {
      source = source.getDefiningOp()->getOperand(0);
      changed = success();
    }

    // TODO: zhiyuang: source could be null
    if (changed.succeeded())
      rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
          op, op.getResultTypes(), ValueRange{source});
    return changed;
  }
};

} // local namespace

void populateTypeConversion(TypeConverter &typeConverter, OpBuilder &builder) {
  // for struct
  auto bufType = LLVM::LLVMStructType::getLiteral(builder.getContext(), {
    MemRefType::get(ArrayRef<int64_t>{0,1}, builder.getI8Type()),
    builder.getI16Type()
  });

  // conversion of type
  typeConverter.addConversion([&](Type type)  -> std::optional<Type> {
    if (type == bufType)
      return builder.getType<ep2::BufferType>();
    return type;
  });
  // memref to code
  typeConverter.addConversion([&](MemRefType memref) -> std::optional<Type> {
    auto dims = memref.getShape();
    // handle 1d array
    if (dims.size() != 2 || dims[0] != 1)
      return std::nullopt;

    llvm::SmallVector<Type> flatArr{dims[1], memref.getElementType()};
    auto structType = builder.getType<ep2::StructType>(false, flatArr, "_anno_struct_");
    return structType;
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

LogicalResult runCallEliminate(Operation * op) {
  auto builder = OpBuilder(op);

  TypeConverter converter;
  populateTypeConversion(converter, builder);

  // apply rules
  mlir::RewritePatternSet patterns(builder.getContext());
  patterns
      .add<CallRewrite, FoldPolygeistCast>(builder.getContext());
  patterns.add<StoreRewrite>(builder.getContext(), converter);

  FrozenRewritePatternSet patternSet(std::move(patterns));
  return applyPatternsAndFoldGreedily(op, patternSet);
}

struct AssignEliminate : OpConversionPattern<ep2::AssignOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ep2::AssignOp assignOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // If the table is installed correctly, we should eliminate all assign
    if (assignOp.getLhs() != assignOp.getRhs())
      return failure();
    rewriter.eraseOp(assignOp);
    return success();
  }
};

struct MemRefAllocConversion : OpConversionPattern<memref::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::AllocaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto retType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<ep2::InitOp>(op, retType);
    return success();
  }
};

struct AffineLoadConversion : OpConversionPattern<affine::AffineLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO: check this is an 1D index
    // auto index = adaptor.getIndices().back();
    auto index = affineToIndex(op.getMap());
    if (!index)
      return rewriter.notifyMatchFailure(op, "Not a 2d constant affine");

    rewriter.replaceOpWithNewOp<ep2::StructAccessOp>(op, adaptor.getMemref(), *index);
    return success();
  }
};

struct CastConversion : OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    SmallVector<Type> newTypes;
    auto retType = getTypeConverter()->convertTypes(op.getResultTypes(), newTypes);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, newTypes, adaptor.getInputs());
    return success();
  }
};

LogicalResult runAssignEliminate(Operation * op) {
  auto builder = OpBuilder(op);

  TypeConverter converter;
  populateTypeConversion(converter, builder);

  ConversionTarget target(*op->getContext());
  target.addIllegalOp<ep2::AssignOp>();
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return converter.isLegal(op);
  });

  // apply rules
  mlir::RewritePatternSet patterns(builder.getContext());
  patterns.add<AssignEliminate>(converter, builder.getContext());
  patterns.add<MemRefAllocConversion,AffineLoadConversion,CastConversion>(converter, builder.getContext());

  FrozenRewritePatternSet patternSet(std::move(patterns));

  return applyPartialConversion(op, target, patternSet);
}

void LiftLLVMPasses::runOnOperation() {
  OpBuilder builder(getOperation());
  // Set all functions to public
  getOperation()->walk([&](func::FuncOp func) {
    // make sure everything is public and cannot be ignored
    if (!func.isDeclaration())
        func.setPublic();
  });

  // transform function call first
  if (failed(runCallEliminate(getOperation()))) {
    llvm::errs() << "Eliminate Failure\n";
    return signalPassFailure();
  }

  OpPassManager pm;
  auto &funcPm = pm.nest<FuncOp>();
  funcPm.addPass(createCSEPass());

  if (failed(runPipeline(pm, getOperation())))
    return signalPassFailure();

  getOperation()->dump();

  DataFlowSolver solver;
  // must have this two lines, to help with liveness
  solver.load<SparseConstantPropagation>();
  solver.load<DeadCodeAnalysis>();
  solver.load<StackVariableAnalysis>();

  if (solver.initializeAndRun(getOperation()).failed())
    return signalPassFailure();

  getOperation()->walk([&](func::FuncOp func) {
    auto frozenOpList = llvm::map_to_vector(func.getOps(), [](auto &op){return &op;});
    for (auto op : frozenOpList) {
      // find the correct slot to use
      builder.setInsertionPoint(op);
      auto state = solver.lookupState<StackSlotValue>(op);
      if (state) {
        // replace the OPs and optimize
        auto newOperands = llvm::map_to_vector(op->getOperands(), [&](Value operand) {
          auto mappedValue = state->lookup(operand);
          if (mappedValue.getType() != operand.getType()) {
            auto castOp = builder.create<UnrealizedConversionCastOp>(
                operand.getLoc(), TypeRange{operand.getType()},
                ValueRange{mappedValue});
            mappedValue = castOp.getResult(0);
          }
          return mappedValue;
        });
        op->setOperands(newOperands);
      }
    }
  });

  // eliminate the assign op.
  if (runAssignEliminate(getOperation()).failed())
    signalPassFailure();
}


} // namespace ep2
} // namespace mlir
