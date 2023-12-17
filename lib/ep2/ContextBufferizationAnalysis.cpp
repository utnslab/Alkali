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

ContextBufferizationAnalysis::ContextBufferizationAnalysis(Operation *op,
                                                           AnalysisManager &am)
    : am(am) {
  HandlerDependencyAnalysis& handlerAnalysis = am.getAnalysis<HandlerDependencyAnalysis>();
  handlerAnalysis.forEachComponent(
      [&, this](size_t i, HandlerDependencyAnalysis::GraphType &graph,
                HandlerDependencyAnalysis::OrderType &order) {
        // create a context table for all connected nodes
        auto &table = contextTables.emplace_back();
        for (auto funcOp : order) {
          contextMap.try_emplace(funcOp, table);
          funcOp->walk([&](ContextRefOp refOp) {
            llvm::StringRef field = refOp.getName();
            auto type = refOp.getType().getValueType();

            auto [it, isNew] = table.try_emplace(field, type);
            if (!isNew) {
              if (it->second.isa<AnyType>())
                it->second = type;
              else if (it->second != type && !type.isa<AnyType>())
                refOp->emitError("Context field type mismatch");
            }
          });
        }
      });
}

mlir::Type ContextBufferizationAnalysis::getContextType(FuncOp funcOp,
                                                        StringRef name) {
  auto opIt = contextMap.find(funcOp);
  if (opIt == contextMap.end()) {
    funcOp->emitError("Operation not found");
    return AnyType::get(funcOp->getContext());
  }

  auto it = opIt->second.find(name);
  if (it == opIt->second.end()) {
    funcOp->emitError("Context field not found");
    return AnyType::get(funcOp->getContext());
  }
  return it->second;
}

} // namespace ep2
} // namespace mlir
