
#include "mlir/IR/BuiltinDialect.h"
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

ContextAnalysis::ContextAnalysis(Operation* module, AnalysisManager& am) {
  // walk through and find all function targets. 
  HandlerDependencyAnalysis& h = am.getAnalysis<HandlerDependencyAnalysis>();
  EquivalenceClasses<Operation*> ec;
  for (const auto& n_next : h.graph) {
    for (const auto& edge : n_next.second) {
      ec.unionSets(n_next.first, edge.second);
    }
  }
  
  for (EquivalenceClasses<Operation*>::iterator I = ec.begin(), E = ec.end(); I != E; ++I) {
    if (!I->isLeader()) {
      continue;
    }

    this->disj_contexts.emplace_back();
    llvm::StringMap<mlir::Type>& fields = this->disj_contexts.back();

    for (EquivalenceClasses<Operation*>::member_iterator MI = ec.member_begin(I); MI != ec.member_end(); ++MI) {
      (*MI)->walk<WalkOrder::PreOrder>([&](ContextRefOp op) {
        llvm::StringRef ref_name = op->getAttr("name").cast<StringAttr>().getValue();
        auto ty = op->getResult(0).getType().cast<ContextRefType>().getValueType();
        fields.try_emplace(ref_name, ty);
      });
    }
  }
}

}
}
