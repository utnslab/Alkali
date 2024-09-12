
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"

#include <queue>
#include <string>

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace ep2 {

LocalAllocAnalysis::LocalAllocAnalysis(Operation* module, AnalysisManager& am) {
  std::unordered_map<mlir::Operation*, std::string> updateAllocs;

  // 2 types of local allocs necessary- contexts and buffers.
  int bufCtr = 0;
  module->walk<WalkOrder::PreOrder>([&](mlir::Operation* op) {
    std::vector<mlir::Operation*> ppgOps;
    if (isa<ep2::StructUpdateOp>(op)) {
      mlir::Operation* itOp = op;
      while (isa<ep2::StructUpdateOp>(itOp)) {
        ppgOps.push_back(itOp);
        itOp = itOp->getOperand(0).getDefiningOp();
      }
      op = itOp;
    }
    if (isa<ep2::ExtractOp>(op) || isa<ep2::ExtractOffsetOp>(op) ||
        isa<ep2::LoadOp>(op) || isa<ep2::LookupOp>(op) || isa<ep2::InitOp>(op)) {
      if (isa<ep2::StructType>(op->getResult(0).getType()) && 
          !cast<ep2::StructType>(op->getResult(0).getType()).getIsEvent()) {
        localAllocs.emplace(op, "_loc_buf_" + std::to_string(bufCtr++));
      }
      for (mlir::Operation* ppgOp : ppgOps) {
        updateAllocs.emplace(ppgOp, localAllocs[op]);
      }
    }
    return;
  });
  module->walk([&](ep2::StoreOp op) {
    if (isa<ep2::StructType>(op.getValue().getType())) {
      // TODO support ctx store of a block argument
      assert(op.getValue().getDefiningOp() != nullptr);
      std::string alloc;

      if (localAllocs.find(op.getValue().getDefiningOp()) ==
          localAllocs.end()) {
        if (updateAllocs.find(op.getValue().getDefiningOp()) ==
          updateAllocs.end()) {
          op.getValue().getDefiningOp()->dump();
          op->dump();
          llvm_unreachable("store of a non local alloc");
        } else {
          alloc = updateAllocs.find(op.getValue().getDefiningOp())->second;
        }
      } else {
        alloc = localAllocs.find(op.getValue().getDefiningOp())->second;
      }

      localAllocs.emplace(op, alloc);
    }
  });
  module->walk([&](ep2::InitOp op) {
    if (isa<ep2::TableType>(op->getResult(0).getType())) {
      localAllocs.emplace(op, "table_" + std::to_string(bufCtr++));
    }
  });
}

}
}
