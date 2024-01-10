
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
  // 2 types of local allocs necessary- contexts and buffers.
  int bufCtr = 0;
  module->walk([&](ep2::ExtractOp op) {
    // perform an escape analyis- 
    // if do a store to "un-ssa" structure with my value - i.e. StoreOp, StructUpdateOp, InitOp return false.
    // CallOp, StructUpdateOp goes through
    // sink: ReturnOp

    std::queue<mlir::Operation*> q;
    q.push(op);
    // TODO should i have a visited set?
    // TODO this is very coarse, right now just ensures safety.

    while (!q.empty()) {
      mlir::Operation* qop = q.front();
      q.pop();

      if (isa<ep2::ReturnOp>(qop)) {
        return;
      }

      for (mlir::Operation* next : qop->getUsers()) {
        if (isa<ep2::InitOp>(next) || 
            (isa<ep2::StructUpdateOp>(next) && cast<ep2::StructUpdateOp>(next).getNewValue().getDefiningOp() == qop)) {
          return;
        }
        if (isa<ep2::CallOp>(next) || isa<ep2::StructUpdateOp>(next)) {
          q.push(next);
        }
      }
    }

    localAllocs.emplace(op, "_loc_buf_" + std::to_string(bufCtr++));
    return;
  });
  module->walk([&](ep2::LoadOp op) {
    if (isa<ep2::StructType>(op->getResult(0).getType())) {
      localAllocs.emplace(op, "_loc_buf_" + std::to_string(bufCtr++));
    }
  });
  module->walk([&](ep2::InitOp op) {
    if (isa<ep2::StructType>(op->getResult(0).getType()) &&
        !cast<ep2::StructType>(op->getResult(0).getType()).getIsEvent()) {
      localAllocs.emplace(op, "_loc_buf_" + std::to_string(bufCtr++));
    } else if (isa<ep2::TableType>(op->getResult(0).getType())) {
      localAllocs.emplace(op, "table_" + std::to_string(bufCtr++));
    }
  });
  module->walk([&](ep2::LookupOp op) {
    if (isa<ep2::StructType>(op->getResult(0).getType())) {
      localAllocs.emplace(op, "lookup_buf_" + std::to_string(bufCtr++));
    }
  });
}

}
}
