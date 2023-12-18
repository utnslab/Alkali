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
        int ctr = 0;
        for (auto funcOp : order) {
          contextMap.try_emplace(funcOp.getSymName().str(), table);
          funcOp->walk([&](ContextRefOp refOp) {
            llvm::StringRef field = refOp.getName();
            auto type = refOp.getType().getValueType();

            const std::pair<int, mlir::Type> pr = {ctr, type};
            auto [it, isNew] = table.try_emplace(field, pr);
            if (!isNew) {
              if (it->second.second.isa<AnyType>())
                it->second.second = type;
              else if (it->second.second != type && !type.isa<AnyType>())
                refOp->emitError("Context field type mismatch");
            } else {
              ctr += 1;
            }
          });
        }
      });
}

std::pair<int, mlir::Type> ContextBufferizationAnalysis::getContextType(FunctionOpInterface funcOp,
                                                        StringRef name) {
  std::string funcName = funcOp.getName().str().find("__event_") == std::string::npos ?
    funcOp.getName().str() : funcOp.getName().str().substr(8);

  auto opIt = contextMap.find(funcName);
  if (opIt == contextMap.end()) {
    funcOp->emitError("Operation not found");
    return {0, mlir::Type{}};
  }

  auto it = opIt->second.find(name);
  if (it == opIt->second.end()) {
    funcOp->emitError("Context field not found");
    return {0, mlir::Type{}};
  }
  return it->second;
}

} // namespace ep2
} // namespace mlir
