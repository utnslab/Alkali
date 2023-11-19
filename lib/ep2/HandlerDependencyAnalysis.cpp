
#include "mlir/IR/BuiltinDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;

namespace mlir {
namespace ep2 {

HandlerDependencyAnalysis::HandlerDependencyAnalysis(Operation* module) {
  llvm::StringMap<llvm::StringMap<Operation*>> event_handlers;
  // walk through and find all function targets. 
  module->walk([&](FuncOp op) {
    if (op->getAttr("type").cast<StringAttr>().getValue() == "handler") {
      llvm::StringRef atom_name = op->hasAttr("atom") ? op->getAttr("atom").cast<StringAttr>().getValue() : "";
      llvm::StringRef call_class = op->getAttr("event").cast<StringAttr>().getValue();
      event_handlers[call_class][atom_name] = op;
    }
  });

  // assume handlers only invoke each other via generate statements- no ep2.call's.
  module->walk<WalkOrder::PreOrder>([&](FuncOp curr_func) {
    if (curr_func->getAttr("type").cast<StringAttr>().getValue() == "handler") {
      curr_func.getOperation()->walk<WalkOrder::PreOrder>([&](ReturnOp op) {
        if (op->getNumOperands() == 0) {
          // raises no event.
        } else if (op->getNumOperands() == 1) {
          // returning an event.
          // TODO should be checked in IR verification
          assert(op->getOperand(0).getType().isa<StructType>() && cast<StructType>(op->getOperand(0).getType()).getIsEvent());

          /* Want to see whether we can STATICALLY dispatch the event.
              1) event should have 1 atom
              2) atom is a constant. Else, just say the target can be ANY handler within that event class*/
          llvm::StringRef call_class = cast<StructType>(op->getOperand(0).getType()).getName();

          int atom_pos = -1;
          const auto& elementTypes = cast<StructType>(op->getOperand(0).getType()).getElementTypes();
          for (int i = 0; i<elementTypes.size(); ++i) {
            if (isa<AtomType>(elementTypes[i])) {
              atom_pos = atom_pos == -1 ? i : -1;
            }
          }

          if (atom_pos != -1) {
            
            /* by definition, return type is an event. Assuming std compiler passes like copy propagation,
               constant folding, etc have already run, the def of the return value will be something
               defining a struct- either an InitOp or StructConstantOp. */
            Operation* struct_def_op = op->getOperand(0).getDefiningOp();

            // for now, assuming no other producing relationship here?
            assert(struct_def_op != nullptr);

            if (isa<StructConstantOp>(struct_def_op)) {
              assert(false && "Not implemented yet"); 
            } else if (isa<InitOp>(struct_def_op)) {
              Operation* atom_prod = struct_def_op->getOperand(atom_pos).getDefiningOp(); 
              /* after constant propagation, so just check atom_prod.
                if nullptr, means a basic block argument, so assume not const */
              assert(this->graph.find(curr_func.getOperation()) == this->graph.end());
              if (atom_prod != nullptr && isa<ConstantOp>(atom_prod)) {
                // is a constant- static call graph dependency.
                ConstantOp cop = static_cast<ConstantOp>(atom_prod);
                llvm::StringRef atom = cop.getValue().cast<StringAttr>().getValue();
                if (isa<StringAttr>(cop.getValue()) &&
                    event_handlers.find(call_class) != event_handlers.end() && 
                    event_handlers[call_class].find(atom) != event_handlers[call_class].end()) {
                  this->graph[curr_func.getOperation()].emplace_back(MUST, event_handlers[call_class][atom]);
                }
              } else {
                
                // dynamic dependency.
                if (event_handlers.find(call_class) != event_handlers.end()) {
                  llvm::StringRef atom_name = curr_func->hasAttr("atom") ? curr_func->getAttr("atom").cast<StringAttr>().getValue() : "";
                  if (atom_name != "") {
                    for (const auto& entry : event_handlers[call_class]) {
                      this->graph[curr_func.getOperation()].emplace_back(MAY, entry.second);
                    }
                  } else {
                    llvm::StringRef base_call_class = curr_func->getAttr("event").cast<StringAttr>().getValue();
                    for (const auto& each : event_handlers[base_call_class]) {
                      for (const auto& entry : event_handlers[call_class]) {
                        this->graph[each.second].emplace_back(MAY, entry.second);
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          assert(false);
        }
      });
    }
  });
}

}
}
