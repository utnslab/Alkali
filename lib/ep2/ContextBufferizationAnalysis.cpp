#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/EquivalenceClasses.h"

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace ep2 {

ContextBufferizationAnalysis::ContextBufferizationAnalysis(Operation* op, AnalysisManager& am) {
  HandlerDependencyAnalysis& handlerAnalysis = am.getAnalysis<HandlerDependencyAnalysis>();
  auto moduleOp = dyn_cast<ModuleOp>(op);

  handlerAnalysis.forEachComponent([&,this](size_t i, auto& graph, auto& order){
    auto &table = contextTables.emplace_back();
    for (auto funcOp : order) {
      contextMap.try_emplace(funcOp, table);
      for (auto refOp : moduleOp.getOps<ContextRefOp>()) {
          llvm::StringRef field = refOp.getName();
          auto type = refOp.getType().getValueType();

          auto [it, isNew] = table.try_emplace(field, type);
          if (!isNew) {
            if (it->second.isa<AnyType>())
              it->second = type;
            else if (it->second != type && !type.isa<AnyType>())
              refOp->emitError("Context field type mismatch");
          }
      }
    }
  });
}

mlir::Type ContextBufferizationAnalysis::getContextType(Operation *op,
                                                        StringRef name) {
  auto opIt = contextMap.find(op);
  if (opIt == contextMap.end()) {
    op->emitError("Operation not found");
    return AnyType::get(op->getContext());
  }

  auto it = opIt->second.find(name);
  if (it == opIt->second.end()) {
    op->emitError("Context field not found");
    return AnyType::get(op->getContext());
  }
  return it->second;
}

} // namespace ep2
} // namespace mlir
