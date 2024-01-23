
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

// Implement handler dependency
HandlerDependencyAnalysis::HandlerFullName::HandlerFullName(FuncOp funcOp) {
  assert(funcOp->hasAttrOfType<StringAttr>("event") &&
         "Handler must have an event attribute");
  event = funcOp->getAttr("event").cast<StringAttr>().getValue();

  if (funcOp->hasAttrOfType<StringAttr>("atom"))
    atom = funcOp->getAttrOfType<StringAttr>("atom").getValue();
}

void setWithInit(HandlerDependencyAnalysis::HandlerFullName *fullname, InitOp initOp) {
  auto eventType = cast<StructType>(initOp.getType());
  assert(eventType && eventType.getIsEvent() && "init op must be an event");

  fullname->event = eventType.getName();
  if (initOp->getOperand(0).getDefiningOp())
    if (auto constantOp =
            dyn_cast<ConstantOp>(initOp->getOperand(0).getDefiningOp()))
      fullname->atom = constantOp.getValue().cast<StringAttr>().getValue();

}

HandlerDependencyAnalysis::HandlerFullName::HandlerFullName(InitOp initOp) {
  setWithInit(this, initOp);
}

HandlerDependencyAnalysis::HandlerFullName::HandlerFullName(ReturnOp returnOp) {
  auto inputOp = returnOp.getInput()[0].getDefiningOp();
  assert(inputOp && isa<InitOp>(inputOp) &&
         "Requires an init op to build return value");
  setWithInit(this, cast<InitOp>(inputOp));
}

FuncOp HandlerDependencyAnalysis::lookupHandler(HandlerFullName fullname) {
  // try to find exact match
  auto it = handlersMap.find(fullname);
  if (it != handlersMap.end())
    return it->second;

  // try to find general match: event only and do not have an atom
  auto it2 = llvm::find_if(handlersMap, [&](auto &pair) {
    return pair.first.event == fullname.event && pair.first.atom.empty();
  });
  if (it2 != handlersMap.end())
    return it2->second;

  return nullptr;
}

FuncOp HandlerDependencyAnalysis::lookupController(HandlerFullName fullname) {
  auto it = llvm::find_if(controllersMap, [&](auto &pair) {
    return pair.first.event == fullname.event;
  });
  if (it != controllersMap.end())
    return it->second;
  return nullptr;
}

HandlerDependencyAnalysis::HandlerDependencyAnalysis(Operation *module) {
  auto moduleOp = dyn_cast<ModuleOp>(module);
  
  std::map<StringRef, std::vector<HandlerFullName>> externForwards;
  std::vector<FuncOp> controllerOps;
  llvm::copy_if(moduleOp.getOps<FuncOp>(), std::back_inserter(controllerOps),
                [](auto funcOp) { return funcOp.isController(); });
  // for coontrollers
  for (auto funcOp : controllerOps)
    controllersMap.emplace(funcOp, funcOp);

  // for handlers, insert handlers
  std::vector<FuncOp> funcOps;
  llvm::copy_if(moduleOp.getOps<FuncOp>(), std::back_inserter(funcOps),
                [](auto funcOp) { return funcOp.isHandler(); });
  for (auto funcOp : funcOps)
    handlersMap.emplace(funcOp, funcOp);

  for (auto funcOp : funcOps) {
    HandlerFullName from(funcOp);

    std::vector<HandlerFullName> to;
    funcOp->walk([&](ReturnOp op) {
      if (op->getNumOperands() == 1)
        to.emplace_back(op);
    });

    if (funcOp.isExtern()) {
      externForwards.emplace(from.event, std::move(to));
    } else { // this is a full handler
      std::vector<FuncOp> targets;
      for (auto &target : to) {
        eventDeps[from.event.str()].insert(target.event.str());

        // try to find exact match
        auto it = handlersMap.find(target);
        if (it != handlersMap.end()) {
          targets.push_back(it->second);
          continue;
        }

        // try to find general match: event only and do not have an atom
        auto it3 = llvm::find_if(handlersMap, [&](auto &pair) {
          return pair.first.event == target.event && pair.first.atom.empty();
        });
        if (it3 != handlersMap.end())
          targets.push_back(it3->second);

        // If its an extern, we further search for extern forwards
        if (!it3->second.isExtern())
          continue;

        // try to find extern forward
        // TODO(zhiyuang): move this to extern
        auto it2 = externForwards.find(target.event);
        if (it2 != externForwards.end()) {
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
      }

      // insert back to graph
      graph[funcOp] = std::move(targets);
    }

  }

  getConnectedComponents();
}

void HandlerDependencyAnalysis::dump() {
  llvm::errs() << "Found " << subGraphs.size() << " connected components\n";
  for (size_t i = 0; i < subGraphs.size(); ++i) {
    llvm::errs() << "Component " << i << " " << subGraphs[i].size() << " "
                 << subGraphsOrder[i].size() << "\n";
  }

  llvm::errs() << "\nFound " << handlersMap.size() << " handlers:\n";
  for (auto &[handler, funcOp] : handlersMap) {
    llvm::errs() << "  " << handler.mangle() << " | " << funcOp.isHandler()
                 << funcOp.isExtern() << "\n";
  }
  llvm::errs() << "\nFound " << controllersMap.size() << " controllers:\n";
  for (auto &[controller, funcOp] : controllersMap) {
    llvm::errs() << "  " << controller.mangle() << " | " << funcOp.isHandler()
                 << funcOp.isExtern() << "\n";
  }

  llvm::errs() << "\n";
  for (auto &[handler, edges] : graph) {
    Operation *op = handler;
    auto funcOp = dyn_cast<FuncOp>(op);
    llvm::errs() << "Handler " << funcOp.getSymName().str() << " has "
                 << edges.size() << " edges\n";
    for (auto &target : edges) {
      Operation *op = target;
      auto funcOp = dyn_cast<FuncOp>(op);
      llvm::errs() << "  " << funcOp.getSymName().str() << "\n";
    }
  }
}

} // end namespace ep2
} // end namespace mlir
