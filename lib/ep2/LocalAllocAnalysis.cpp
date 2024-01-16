
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
  module->walk([&](mlir::Operation* op) {
    if (isa<ep2::ExtractOp>(op) || isa<ep2::ExtractOffsetOp>(op) ||
        isa<ep2::LoadOp>(op) || isa<ep2::LookupOp>(op) || isa<ep2::InitOp>(op)) {
      if (isa<ep2::StructType>(op->getResult(0).getType()) && 
          !cast<ep2::StructType>(op->getResult(0).getType()).getIsEvent()) {
        localAllocs.emplace(op, "_loc_buf_" + std::to_string(bufCtr++));
      }
    }
    return;
  });
  module->walk([&](ep2::StoreOp op) {
    if (isa<ep2::StructType>(op.getValue().getType())) {
      // TODO support ctx store of a block argument
      assert(op.getValue().getDefiningOp() != nullptr);
      assert(localAllocs.find(op.getValue().getDefiningOp()) != localAllocs.end());
      localAllocs.emplace(op, localAllocs[op.getValue().getDefiningOp()]);
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
