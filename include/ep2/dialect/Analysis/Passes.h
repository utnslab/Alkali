#ifndef _ANALYSIS_PASSES_H_
#define _ANALYSIS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "ep2/dialect/Analysis/ForwardDataflowAnalysis.h"

namespace mlir {
namespace ep2 {

/////////////////////////////
// Extracted field usage analysis
/////////////////////////////

struct FieldUsage : ValueStateBase {

  Value source;
  StructType type;
  llvm::BitVector used;

  FieldUsage() = delete;
  FieldUsage(Value source)
      : source{source}, type{source.getType().dyn_cast<StructType>()},
        used{static_cast<unsigned>(type.getNumElementTypes())} {}

  // interface
  void merge(FieldUsage &rhs) {
    // TODO(zhiyuang): check source?
    used |= rhs.used;
  }
};

struct FieldExtractInfo {
  // TODO(zhiyuang): collect final result
  DenseMap<Value, FieldUsage> usage{};
  DenseMap<Value, SmallVector<Operation*>> users{};
  void update(FieldUsage &use, Operation *op) {
    auto [it, isNew] = usage.try_emplace(use.source, use);
    if (!isNew)
      it->second.merge(use);

    auto [it2, _] = users.try_emplace(use.source);
    it2->second.push_back(op);
  }
};

struct FieldExtractVisitor : public ep2::Visitor<FieldUsage, FieldExtractInfo> {
  void visit(Operation *op, DataflowContext<FieldUsage> &context,
             FieldExtractInfo &global) {
    if (auto extractOp = dyn_cast<ExtractOffsetOp>(op)) {
      FieldUsage state{extractOp};
      context.decalre(extractOp, state);
      global.update(state, op);
    }
    if (auto accessOp = dyn_cast<StructAccessOp>(op)) {
      if (auto state = context.query(accessOp.getInput())) {
        state->used.set(accessOp.getIndex());
        global.update(*state, op);
      }
    }
    if (auto updateOp = dyn_cast<StructUpdateOp>(op)) {
      if (auto state = context.query(updateOp.getInput())) {
        FieldUsage newState = *state;
        newState.used.set(updateOp.getIndex());
        context.decalre(updateOp, newState);
        global.update(newState, op);
      }
    }
  }
  // no need to provide new handler
  void processInitialHandler(FuncOp func, InputMapT &inputs) {}
};

using ExtractFiledAnalysis =
    ::mlir::ep2::ForwardDataflowAnalyis<FieldUsage, FieldExtractInfo, FieldExtractVisitor>;

/////////////////////////////
// lifetime of parameter
/////////////////////////////

struct ContextVariableInfo : ValueStateBase {
  std::string name;
  bool used;

  ContextVariableInfo(std::string name)
      : name{name}, used{false} {}

  // interface
  void merge(ContextVariableInfo &rhs) {
    used |= rhs.used;
  }
};

struct ContextVariableResult {
  // TODO(zhiyuang): collect final result
  std::map<std::string, ContextVariableInfo> finalInfo{};
  void update(std::string name, ContextVariableInfo &info) {
    auto [it, isNew] = finalInfo.try_emplace(name, info);
    if (!isNew)
      it->second.merge(info);
  }
};

struct ContextVariableAnalysisVisitor : public ep2::Visitor<ContextVariableInfo, ContextVariableResult> {
  void visit(Operation *op, DataflowContext<ContextVariableInfo> &context,
             ContextVariableResult &global) {
    // for any other op, just check if its used
    // TODO(zhiyuang): finish isRealUse
    for (auto operand : op->getOperands()) {
      bool isRealUse = !isa<InitOp>(op);
      if (auto state = context.query(operand)) {
        state->used |= isRealUse;
        global.update(state->name, *state);
      }
    }
    
    if (auto initOp = dyn_cast<InitOp>(op)) {
      auto event = dyn_cast<StructType>(initOp.getType());
      if (event && event.getIsEvent() && initOp->hasAttr("context_names")) {
        auto names = initOp->getAttrOfType<ArrayAttr>("context_names")
                         .getAsValueRange<StringAttr>();
        for (auto [i, name, value] :
             llvm::enumerate(names, initOp->getOperands())) {
          ContextVariableInfo info{name.str()};
          context.update(value, info);
          global.update(name.str(), info);
        }
      }
    }
  }
  // no need to provide new handler
  void processInitialHandler(FuncOp func, InputMapT &inputs) {}
};

using ContextVariableAnalysis =
    ::mlir::ep2::ForwardDataflowAnalyis<ContextVariableInfo, ContextVariableResult, ContextVariableAnalysisVisitor>;

} // namespace ep2
} // namespace mlir

#endif // _ANALYSIS_PASSES_H_