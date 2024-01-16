#ifndef _FORWARD_DATAFLOW_ANALYSIS_H_
#define _FORWARD_DATAFLOW_ANALYSIS_H_

#include "mlir/IR/BuiltinDialect.h"

#include <map>
#include <string>
#include <type_traits>

#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace mlir {
namespace ep2 {

struct ValueStateBase {
  void merge(ValueStateBase &rhs) {};
};

template<typename ValueState>
struct DataflowContext {
  using InputMapT = std::vector<std::map<int, ValueState>>;
  llvm::DenseMap<Value, ValueState> table;

  void decalre(Value value, ValueState state) {
    table.try_emplace(value, std::move(state));
  }

  void update(Value value, ValueState state) {
    auto [it, isNew] = table.try_emplace(value, std::move(state));
    if (!isNew)
      it->second.merge(state);
  }

  ValueState *query(Value value) {
    auto it = table.find(value);
    if (it == table.end())
      return nullptr;
    return &it->second;
  }

  DataflowContext() : table{} {}
  DataflowContext(InputMapT &inputs, Block &block) {
    std::map<int, ValueState> merged;

    // try to merge variable from different source
    for (auto &mapping : inputs)
      for (auto &[argIndex, data] : mapping) {
        auto [it, isNew] = merged.try_emplace(argIndex, data);
        if (!isNew)
          it->second.merge(data);
      }

    // find the mapping to value
    for (auto &[key, value] : merged) {
      auto ba = block.getArgument(key);
      table.try_emplace(ba, value);
    }
  }

  void merge(DataflowContext &rhs) {
    for (auto &[key, value] : rhs.table) {
      auto [it, isNew] = table.try_emplace(key, value);
      if (!isNew)
        it->second.merge(value);
    }
  }
};

template<typename ValueState, typename GlobalState>
struct Visitor {
  using InputMapT = std::vector<std::map<int, ValueState>>;
  void visit(Operation *op, DataflowContext<ValueState> &context,
             GlobalState &global){};
  void processInitialHandler(FuncOp func, InputMapT &inputs){};
};


/// Class for general forward dataflow analysis on EP2
template <typename ValueState, typename GlobalState, typename VisitorT>
struct ForwardDataflowAnalyis {
  using InputMapT = std::vector<std::map<int, ValueState>>;

  void analysisFuncion(FuncOp funcOp) {
    // as context cannot be shared beyond function, its local
    for (auto &block : funcOp.getBody()) {
      auto [it, _] = blockContext.try_emplace(&block, inputTable[&block], block);
      auto &context = it->second;

      // merge all context. this will not messup with inputs, as block arguments
      // are separated values
      for (auto pre : block.getPredecessors())
        context.merge(blockContext[pre]);

      // do per-op operation
      for (auto &op : block) {
        visitor.visit(&op, context, globalState);
        // init op
        if (auto initOp = dyn_cast<InitOp>(&op)) {
          StructType event;
          if (!(event = initOp.getType().dyn_cast<StructType>()))
            continue;
          if (!event.getIsEvent())
            continue;

          auto targetFunc = dependency.lookupHandler(initOp);
          if (targetFunc.isExtern())
            continue;
          auto [it, _] = inputTable.try_emplace(&targetFunc.getBody().front());
          auto &inputs = it->second.emplace_back();

          for (unsigned i = 0; i < initOp.getNumOperands(); ++i) {
            auto arg = initOp.getOperand(i);
            if (auto state = context.query(arg))
              inputs.emplace(i - 1, *state);
          }
        }

        // TODO(zhiyuang): branch op interface?
        if (auto branchOp = dyn_cast<cf::CondBranchOp>(&op)) {
          auto insertInputs = [&](Block *dest,
                                  Operation::operand_range &&operands) {
            auto [it, _] = inputTable.try_emplace(dest);
            auto &inputs = it->second.emplace_back();
            for (auto [i, arg] : llvm::enumerate(operands))
              if (auto state = context.query(arg))
                inputs.emplace(i, *state);
          };

          insertInputs(branchOp.getTrueDest(), branchOp.getTrueOperands());
          insertInputs(branchOp.getFalseDest(),
                       branchOp.getFalseDestOperands());
        }
      } // for ops in block
    }
  }

  // data
  mlir::AnalysisManager &am;
  HandlerDependencyAnalysis &dependency;

  // must be default cosntructable
  VisitorT visitor{};
  GlobalState globalState{};

  std::map<Block *, InputMapT> inputTable{};
  std::map<Block *, DataflowContext<ValueState>> blockContext{};

  ForwardDataflowAnalyis(Operation *op, mlir::AnalysisManager &am)
      : am(am), dependency(am.getAnalysis<HandlerDependencyAnalysis>()) {
    // analysis in the main constructor
    dependency.forEachComponent([&](size_t i,
                                    HandlerDependencyAnalysis::GraphType &graph,
                                    std::vector<FuncOp> &order) {
      for (auto funcOp : order) {
        // construct input for function with no predecessor
        if (!dependency.hasPredecessor(funcOp)) {
          auto [it, _] = inputTable.try_emplace(&funcOp.getBody().front());
          visitor.processInitialHandler(funcOp, it->second);
        }

        analysisFuncion(funcOp);
      }
    });
  }
};

} // namespace ep2
} // namespace mlir

#endif // _FORWARD_DATAFLOW_ANALYSIS_H_