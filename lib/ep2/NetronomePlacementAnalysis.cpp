
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace ep2 {

NetronomePlacementAnalysis::NetronomePlacementAnalysis(Operation* module, AnalysisManager& am) {
  const HandlerDependencyAnalysis& hda = am.getAnalysis<HandlerDependencyAnalysis>();

  // TODO build a communication cost graph between handlers.
  // three kinds of communication: queues, context, and tables.
  // for now, just place everything on island 0.
  // once we need to, deal with multi-island / use appropriate memory placement.

  unsigned meNum = 0;
  std::map<ep2::FuncOp, std::pair<std::string, std::string>> funcEventAtomMap;

  for (const auto& pr : hda.handlersMap) {
    funcEventAtomMap.emplace(pr.second, std::pair<std::string, std::string>(pr.first.event.str(), pr.first.atom.str()));
  }
  for (const std::vector<ep2::FuncOp>& subGraph : hda.subGraphsOrder) {
    for (const ep2::FuncOp fop : subGraph) {
      std::string event = funcEventAtomMap[fop].first;
      if (event == "NET_SEND") {
        continue;
      }
      this->placementMap[funcEventAtomMap[fop].second] = std::pair<unsigned, unsigned>{0, meNum++};
    }
  }

  // max number of micro-engines on an island
  assert(meNum <= 12);
}

}
}
