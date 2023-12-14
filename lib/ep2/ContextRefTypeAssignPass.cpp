#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/StringMap.h"
#include <iostream>

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace ep2 {

void ContextRefTypeAssignPass::runOnOperation() {
  auto module = getOperation();
  // Fill
  // Construct naming mapping table
  module->walk([&](ContextRefOp op) {
    auto refresult = op.getResult();
    std::string context_field_name =
        op->getAttr("name").cast<StringAttr>().getValue().str();
    auto uses = refresult.getUses();
    for (auto &t : uses) {
      auto uses_op = t.getOwner();
      if (isa<ep2::StoreOp>(uses_op)) {
        // Store OP, assign type to the context ref
        auto uses_stroed_rvalue = uses_op->getOperand(1);
        if (context_ref_name_to_type.find(context_field_name) ==
            context_ref_name_to_type.end()) {
          context_ref_name_to_type[context_field_name] =
              uses_stroed_rvalue.getType();
        } else {
          if (uses_stroed_rvalue.getType() !=
              context_ref_name_to_type[context_field_name]) {
            // To store to the same context field type missmatch
            uses_op->emitError(
                "Mismatch type of two stores to the same context field:")
                << context_field_name;
          }
        }
      }
    }
  });

  OpBuilder builder(module->getContext());
  // Propogate Type
  module->walk([&](ContextRefOp op) {
    auto refresult = op.getResult();
    std::string context_field_name =
        op->getAttr("name").cast<StringAttr>().getValue().str();

    auto newvaluetype = context_ref_name_to_type[context_field_name];
    auto newreftype = builder.getType<ContextRefType>(newvaluetype);

    refresult.setType(newreftype);
    auto uses = refresult.getUses();
    for (auto &t : uses) {
      t.get().setType(newreftype);
    }
  });
}

std::unique_ptr<Pass> createContextRefTypeAssignPass() {
  return std::make_unique<ContextRefTypeAssignPass>();
}

} // namespace ep2
} // namespace mlir
