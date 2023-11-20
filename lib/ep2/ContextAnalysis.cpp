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

    llvm::StringMap<ContextField> fields;
    size_t fieldPlace = 0;

    for (EquivalenceClasses<Operation*>::member_iterator MI = ec.member_begin(I); MI != ec.member_end(); ++MI) {
      (*MI)->walk<WalkOrder::PreOrder>([&](ContextRefOp op) {
        llvm::StringRef ref_name = op->getAttr("name").cast<StringAttr>().getValue();
        auto ty = op->getResult(0).getType().cast<ContextRefType>().getValueType();
        assert(ty.isIntOrFloat() && ty.getIntOrFloatBitWidth() % 8 == 0);

        ContextField cf(fieldPlace++, ty.getIntOrFloatBitWidth() / 8, ty);
        fields.try_emplace(ref_name, cf);
        for (mlir::Operation* lsUse : op->getUsers()) {
          this->disj_groups.emplace(lsUse, *(ec.findLeader(I)));
        }
      });
    }
    this->disj_contexts.emplace(*(ec.findLeader(I)), fields);
  }
}

}
}
