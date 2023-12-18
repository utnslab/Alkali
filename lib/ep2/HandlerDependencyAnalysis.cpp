
#include "mlir/IR/BuiltinDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"

#include <queue>

using namespace mlir;

namespace mlir {
namespace ep2 {

void HandlerDependencyAnalysis::getConnectedComponents() {
  // Transform this to an undirected graph, and get degree information
  GraphType undirectedGraph(graph);
  std::map<KeyTy, int> inDegree;
  for (auto &[handler, edges] : graph) {
    for (auto &target : edges) {
      auto [it, _] = inDegree.try_emplace(target, 0);
      it->second++;

      auto &targetEdges = undirectedGraph[target];
      if (!std::any_of(targetEdges.begin(), targetEdges.end(),
                       [&](auto &edge) { return edge == handler; }))
        targetEdges.emplace_back(handler);
    }
  }

  // find all connected components
  std::map<KeyTy, int> color;
  std::vector<std::queue<KeyTy>> initialHandlers;
  for (auto &[handler, _]: graph)
    color[handler] = 0;

  int numColors = 1;
  for (auto &[handler, _]: undirectedGraph) {
    if (color[handler] == 0) {
      int curColor = numColors++;
      auto &subGraph = subGraphs.emplace_back();
      auto &nodes = initialHandlers.emplace_back();

      // BFS the graph
      std::queue<KeyTy> worklist;
      worklist.push(handler);
      while(!worklist.empty()) {
        auto cur = worklist.front();
        worklist.pop();

        subGraph[cur] = graph[cur];
        if (inDegree[cur] == 0)
          nodes.push(cur);

        color[cur] = curColor;
        for (auto &target: undirectedGraph[cur])
          if (color[target] == 0)
            worklist.push(target);
      }
    }
  }

  // sort the subgraphs by the topological order
  for (size_t i = 0; i < initialHandlers.size(); ++i) {
    auto &nodes = subGraphsOrder.emplace_back();
    auto &worklist = initialHandlers[i];

    while(!worklist.empty()) {
      auto cur = worklist.front();
      worklist.pop();
      nodes.push_back(cur);
      for (auto &target: graph[cur]) {
        if (--inDegree[target] == 0)
          worklist.push(target);
      }
    }
  }

}

struct HandlerFullName {
  llvm::StringRef event;
  llvm::StringRef atom = "";

  friend bool operator<(const HandlerFullName &l, const HandlerFullName &r) {
    return std::tie(l.event, l.atom) < std::tie(r.event, r.atom);
  }

  HandlerFullName(FuncOp funcOp) {
    assert(funcOp->hasAttrOfType<StringAttr>("event") && "Handler must have an event attribute");
    event = funcOp->getAttr("event").cast<StringAttr>().getValue();

    if (funcOp->hasAttrOfType<StringAttr>("atom"))
      atom = funcOp->getAttrOfType<StringAttr>("atom").getValue();
  }

  HandlerFullName(ReturnOp returnOp) {
    auto eventType = cast<StructType>(returnOp->getOperand(0).getType());
    // TODO(zhiyuang): move this to an verifier
    assert(eventType && eventType.getIsEvent() && "Return type must be an event");

    event = eventType.getName();
    // TODO(zhiyuang): verifier. require all atom type to be at 0
    auto inputOp = returnOp.getInput()[0].getDefiningOp();
    assert(inputOp && isa<InitOp>(inputOp) &&
            "Requires an init op to build return value");

    if (inputOp->getOperand(0).getDefiningOp())
      if (auto constantOp =
              dyn_cast<ConstantOp>(inputOp->getOperand(0).getDefiningOp()))
        atom = constantOp.getValue().cast<StringAttr>().getValue();
  }
};

HandlerDependencyAnalysis::HandlerDependencyAnalysis(Operation *module) {
  auto moduleOp = dyn_cast<ModuleOp>(module);
  
  std::map<StringRef, std::vector<HandlerFullName>> externForwards;

  std::map<HandlerFullName, FuncOp> handlersMap;
  for (auto funcOp : moduleOp.getOps<FuncOp>())
    handlersMap.emplace(funcOp, funcOp);

  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    if (!funcOp.isHandler())
      continue;

    HandlerFullName from(funcOp);

    std::vector<HandlerFullName> to;
    funcOp->walk([&](ReturnOp op) {
      if (op->getNumOperands() == 1)
        to.emplace_back(op);
    });

    if (funcOp.isExtern()) {
      // TODO(zhiyuang): emplace?
      externForwards.emplace(from.event, std::move(to));
    } else { // this is a full handler
      std::vector<FuncOp> targets;
      for (auto &target : to) {
        eventDeps.emplace(from.event.str(), target.event.str());

        auto it = handlersMap.find(target);
        if (it != handlersMap.end()) {
          targets.push_back(it->second);
          continue;
        }

        auto it2 = externForwards.find(target.event);
        if (it2 == externForwards.end())
          assert(false && "Cannot find target handler");
        for (auto &target2 : it2->second) {
          HandlerFullName newTarget(target2);
          newTarget.atom = target.atom;
          auto it = handlersMap.find(target);
          if (it != handlersMap.end()) {
            targets.push_back(it->second);
            continue;
          }
        }
      }

      // insert back to graph
      graph[funcOp] = std::move(targets);
    }

  }

  getConnectedComponents();
}

} // end namespace ep2
} // end namespace mlir
