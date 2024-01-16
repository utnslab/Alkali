#include "mlir/IR/BuiltinDialect.h"

#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "ep2/dialect/Analysis/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace ep2 {

namespace {

} // namespace

void DeadParameterEliminatePass::runOnOperation() {
  auto &fieldAnalysis = getAnalysis<ContextVariableAnalysis>();
  auto &result = fieldAnalysis.globalState;

  ModuleOp moduleOp = getOperation();

  for (auto [name, usage]: result.finalInfo) {
    std::string usageStr = usage.used ? "used" : "unused";
    llvm::errs() << "context " << name << " is " << usageStr << "\n";
  }

  // first, we clear the uses of the arguments by rewriting the initOps
  std::vector<Operation *> toRemove;
  moduleOp->walk([&](InitOp initOp) {
    // TODO(zhiyaung): merge this code to the internal event op
    auto event = initOp.getType().dyn_cast<StructType>();
    if (event && event.getIsEvent() &&
        initOp->getAttrOfType<ArrayAttr>("context_names")) {
      auto names = initOp->getAttrOfType<ArrayAttr>("context_names")
                       .getAsValueRange<StringAttr>();
      auto oeprands = initOp->getOperands();

      SmallVector<StringRef> newNames;
      SmallVector<Value> newOperands;
      for (auto [name, value] : llvm::zip_equal(names, oeprands)) {
        if (result.isUnused(name.str()))
          continue;
        newNames.push_back(name);
        newOperands.emplace_back(value);
      }

      OpBuilder builder(initOp);
      auto newInitOp = builder.create<InitOp>(
          initOp.getLoc(), event.getName(), newOperands);
      newInitOp->setAttr("context_names",
                         builder.getStrArrayAttr(newNames));
      initOp.replaceAllUsesWith(newInitOp.getResult());
      toRemove.push_back(initOp);
    }
  });
  for (auto op : toRemove)
    op->erase();


  // we remove the func args.
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    auto toRemove = BitVector(funcOp.getNumArguments());
    for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
      auto attr = funcOp.getArgAttrOfType<StringAttr>(i, "ep2.context_name");
      if (attr && result.isUnused(attr.str())) 
        toRemove.set(i);
    }
    funcOp.eraseArguments(toRemove);
  }
}

} // namespace ep2
} // namespace mlir