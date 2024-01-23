
#include "mlir/IR/BuiltinDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include <queue>
#include <unordered_set>

using namespace mlir;

template<>
struct std::hash<mlir::Value> {
  size_t operator()(const mlir::Value& v) const noexcept {
    return reinterpret_cast<size_t>(v.getAsOpaquePointer());
  }
};

namespace mlir {
namespace ep2 {

/*  Optimizations:
1) Verify no cls_workqueue or memory ops in critical region
2) Lower no_swap_begin/no_swap_end to netronome intrinsics
*/

void LowerNoctxswapPass::runOnOperation() {
  auto module = getOperation();

  /*
  auto findDependentOps = [&](mlir::Operation* baseOp) {
    std::unordered_set<mlir::Value> v;
    std::queue<mlir::Operation*> q;

    q.push(baseOp);
    while (!q.empty()) {
      mlir::Operation* op = q.front();
      q.pop();

      for (const auto& opd : op->getOperands()) {
        v.insert(opd);
        if (opd.getDefiningOp() == nullptr) {
          mlir::Block* b = opd.cast<mlir::BlockArgument>().getOwner();
          int argNum = opd.cast<mlir::BlockArgument>().getArgNumber();
          for (mlir::Block* pred : b->getPredecessors()) {
            mlir::Operation* term = pred->getTerminator();
            mlir::Operation* next = nullptr;
            if (isa<cf::BranchOp>(term)) {
              next = cast<cf::BranchOp>(term).getDestOperands()[argNum].getDefiningOp();
            } else if (isa<cf::CondBranchOp>(term)) {
              cf::CondBranchOp ct = cast<cf::CondBranchOp>(term);
              if (ct.getTrueDest() == b) {
                next = ct.getTrueDestOperands()[argNum].getDefiningOp();
              } else if (ct.getFalseDest() == b) {
                next = ct.getFalseDestOperands()[argNum].getDefiningOp();
              } else {
                assert(false && "Unreachable");
              }
              q.push(cast<mlir::Operation*>(next));
            } else {
              assert(false && "Unsupported terminator");
            }
          }
        } else {
          q.push(opd.getDefiningOp());
        }
      }
    }
    return v;
  };

  module->walk([&](func::FuncOp fop) {
    if (fop.getName().contains("__handler")) {
      // is handler
      mlir::Operation* lookupOp = nullptr;
      mlir::Operation* updateOp = nullptr;

      fop->walk([&](emitc::CallOp op) {
        if (op.getCallee() == "__ep2_intrin_table_lookup") {
          assert(lookupOp == nullptr);
          lookupOp = op;
        } else if (op.getCallee() == "__ep2_intrin_table_update") {
          assert(updateOp == nullptr);
          updateOp = op;
        }
      });

      if (lookupOp != nullptr && updateOp != nullptr) {
        // TODO check if lookup feeds update
        std::unordered_set<mlir::Value> deps = findDependentOps(updateOp);
        fop->walk([&](emitc::CallOp op) {
          bool inSet = false;
          for (const auto& v : op->getOperands()) { 
            if (deps.find(v) != deps.end()) {
              inSet = true;
            }
          }
          for (const auto& v : op->getResults()) { 
            if (deps.find(v) != deps.end()) {
              inSet = true;
            }
          }
        });
      }
    }
  });
  // todo finish
  */
}

} // namespace ep2
} // namespace mlir
