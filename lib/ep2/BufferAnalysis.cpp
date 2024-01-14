#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/EquivalenceClasses.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace mlir {
namespace ep2 {

namespace {

using ArgMap = BufferAnalysis::ArgMap;
using BufferHistory = BufferAnalysis::BufferHistory;

int getEP2TypeSize(Type type) {
  return llvm::TypeSwitch<Type, int>(type)
      .Case<StructType>([](StructType type) {
        int size = 0;
        for (auto field : type.getElementTypes())
          size += getEP2TypeSize(field);
        return size;
      })
      .Case<IntegerType>([](IntegerType type) {
        return type.getWidth(); 
      })
      .Default([](auto type) {
        llvm_unreachable("unknown type");
        return 0;
      });
}
  
// Our context structure. mappping from history to name
struct HistoryTable {
  llvm::DenseMap<Value, BufferHistory> table;

  void declare(Value value) {
    table.try_emplace(value, value, true, 0);
  }

  int emit(Value value, Type type) {
    auto it = table.find(value);
    if (it == table.end())
      return 0;

    auto &history = it->second;
    auto curOffset = history.offset;
    history.offset += getEP2TypeSize(type);
    return curOffset;
  }

  // merge input
  void installMapping(Block &block, std::vector<ArgMap> &mappings) {
    std::map<int, BufferHistory> merged;

    // try to merge variable from different source
    for (auto &mapping : mappings)
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

  void merge(HistoryTable &rhs) {
    for (auto &[key, value] : rhs.table) {
      auto [it, isNew] = table.try_emplace(key, value);
      if (!isNew)
        it->second.merge(value);
    }
  }
};

void analysisFuncion(FuncOp funcOp,
                     HandlerDependencyAnalysis &dependency,
                     std::map<Block *, std::vector<ArgMap>> &blockInput,
                     std::map<Operation *, int> &offsetAt,
                     llvm::EquivalenceClasses<mlir::detail::ValueImpl *> &equivalence) {

  std::map<Block *, HistoryTable> blockContext;
  for (auto &block : funcOp.getBody()) {
    auto [it, _] = blockContext.try_emplace(&block);
    auto &table = it->second;

    // merge all context
    for (auto pre : block.getPredecessors())
      table.merge(blockContext[pre]);

    // and overwrite inputs
    table.installMapping(block, blockInput[&block]);

    // do per-op operation
    for (auto &op : block) {
      // declare new inited buffers
      if (auto initOp = dyn_cast<InitOp>(&op)) {
        if (initOp.getType().isa<BufferType>())
          table.declare(initOp);
      }
      // emit buffers
      if (auto emitOp = dyn_cast<EmitOp>(&op)) {
        auto buffer = emitOp.getBuffer();
        auto value = emitOp.getValue();
        if (value.getType().isa<BufferType>()) {
          // We check if the hisotry is valid for a buffer.
          auto &valueHist = table.table[value];
          auto &bufferHist = table.table[buffer];
          if (valueHist.known && bufferHist.known)
            // TODO(zhiyuang): this only works for single emit (not emit to one buffer multiple times) fix later!
            equivalence.unionSets(valueHist.source.getImpl(), bufferHist.source.getImpl());
        } else {
          auto offset = table.emit(emitOp.getBuffer(), emitOp.getValue().getType());
          offsetAt.try_emplace(&op, offset);
        }
      }
      if (auto extractOp = dyn_cast<ExtractOp>(&op)) {
        auto offset = table.emit(extractOp.getBuffer(), extractOp.getType());
        offsetAt.try_emplace(&op, offset);
      }

      // ops for generating new inputs
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
        auto [it, _] = blockInput.try_emplace(&targetFunc.getBody().front());
        auto &inputs = it->second.emplace_back();
        for (unsigned i = 0; i < initOp.getNumOperands(); ++i) {
          auto arg = initOp.getOperand(i);
          if (arg.getType().isa<BufferType>()) {
            auto &history = table.table[arg];
            // TODO(zhiyuang): i - 1?
            inputs.emplace(i - 1, history);
          }
        }
      }
      
      // branch op
      if (auto branchOp = dyn_cast<cf::CondBranchOp>(&op)) {
        auto [it, _] = blockInput.try_emplace(branchOp.getTrueDest());
        auto &inputs = it->second.emplace_back();
        for (unsigned i = 0; i < branchOp.getNumTrueOperands(); i++) {
          auto arg = branchOp.getTrueOperand(i);
          if (arg.getType().isa<BufferType>()) {
            auto &history = table.table[arg];
            inputs.emplace(i, history);
          }
        }

        auto [it2, _b2] = blockInput.try_emplace(branchOp.getTrueDest());
        auto &inputs2 = it2->second.emplace_back();
        for (unsigned i = 0; i < branchOp.getNumTrueOperands(); i++) {
          auto arg = branchOp.getTrueOperand(i);
          if (arg.getType().isa<BufferType>()) {
            auto &history = table.table[arg];
            inputs2.emplace(i, history);
          }
        }
      }
    }
  }
}

} // local namespace

BufferAnalysis::BufferAnalysis(Operation* module, AnalysisManager &am) {
  auto &dependency = am.getAnalysis<HandlerDependencyAnalysis>();

  auto moduleOp = dyn_cast<ModuleOp>(module);
  if (!moduleOp) {
    module->emitError("Buffer analysis on a non-module op");
    return;
  }

  dependency.forEachComponent([&](size_t i,
                                  HandlerDependencyAnalysis::GraphType &graph,
                                  std::vector<FuncOp> &order) {
    for (auto funcOp : order) {
      if (!dependency.hasPredecessor(funcOp)) {
        // install the source from extern
        auto [it, _] = blockInput.try_emplace(&funcOp.getBody().front());
        auto &inputs = it->second.emplace_back();
        for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
          auto arg = funcOp.getArgument(i);
          if (arg.getType().isa<BufferType>()) {
            // source is ba
            inputs.emplace(i, BufferHistory{arg, true, 0});
          }
        }
      }

      analysisFuncion(funcOp, dependency, blockInput, offsetAt, bufferClasses);
    }
  });
}

} // namespace ep2
} // namespace mlir