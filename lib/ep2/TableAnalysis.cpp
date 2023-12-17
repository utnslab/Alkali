
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "mlir/IR/BuiltinDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;

namespace mlir {
namespace ep2 {

TableAnalysis::TableAnalysis(Operation *module) {
  // walk through and find all function targets.

  module->walk([&](InitOp initop) {
    if (isa<ep2::TableType>(initop.getResult().getType())) {
      auto table = initop.getResult();
      int update_index = 0;
      int lookup_index = 0;
      mlir::SmallVector<ep2::LookupOp> tmp_lookup_v;
      mlir::SmallVector<ep2::UpdateOp> tmp_update_v;
      for (auto table_use : table.getUsers()) {
        if(isa<ep2::UpdateOp>(table_use)){
          auto updateop = cast<ep2::UpdateOp, mlir::Operation *>(table_use);
          tmp_update_v.push_back(updateop);
          access_index[updateop] = update_index++;
        }else if (isa<ep2::LookupOp>(table_use)) {
          auto lookupop = cast<ep2::LookupOp, mlir::Operation *>(table_use);
          tmp_lookup_v.push_back(lookupop);
          access_index[lookupop] = lookup_index++;
        } else{
          table_use->dump();
          assert(false);
        }
      }
      
      table_update_uses[table] = tmp_update_v;
      table_lookup_uses[table] = tmp_lookup_v;
    }
  });


}
} // namespace ep2
} // namespace mlir
