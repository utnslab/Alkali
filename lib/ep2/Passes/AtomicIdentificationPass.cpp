#include "mlir/IR/BuiltinDialect.h"

#include "ep2/lang/Lexer.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace ep2 {

// TODO(zhiyuang): this should be an forward dataflow analysis pass
namespace {

void tryRemoveSync(FuncOp funcOp) {
  bool hasNonAtomicUse = false;
  funcOp->walk([&](GlobalImportOp op) {
    for (auto user : op->getUsers()) {
      if (isa<LookupOp, UpdateOp>(user))
        hasNonAtomicUse = true;
    }
  });

  // remove the sync attr
  if (hasNonAtomicUse)
    funcOp->removeAttr("scope");
}

struct LoadModifyWritePattern : public RewritePattern {
  LoadModifyWritePattern(MLIRContext *context)
      : RewritePattern(LookupOp::getOperationName(), 1, context,
                       {UpdateAtomicOp::getOperationName()}) {}

  virtual LogicalResult matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const override {
    auto loadOp = dyn_cast<LookupOp>(op);
    if (!loadOp)
      return rewriter.notifyMatchFailure(op, "not a load op");
    // if its a syncLoad
    auto importOp = dyn_cast<GlobalImportOp>(loadOp.getTable().getDefiningOp());
    if (!importOp)
      return rewriter.notifyMatchFailure(importOp, "load is not from a global import op");

    // TODO(zhiyuang): for this transformation, we only need to check the users
    if (std::distance(loadOp->getUsers().begin(), loadOp->getUsers().end()) != 1)
      return rewriter.notifyMatchFailure(op, "load has more than one user");

    auto userOp = *loadOp->getUsers().begin();
    auto [atomicOpCode, value] =
        llvm::TypeSwitch<Operation *, std::pair<int, Value>>(userOp)
            .Case<AddOp>([&](AddOp op) {
              auto value =
                  op.getRhs() == loadOp.getValue() ? op.getLhs() : op.getRhs();
              return std::make_pair(1, value);
            })
            .Case<SubOp>([&](SubOp op) {
              auto value =
                  op.getRhs() == loadOp.getValue() ? op.getLhs() : op.getRhs();
              return std::make_pair(1, value);
            })
            .Default(
                [](auto op) { return std::make_pair(-1, (Value) nullptr); });
    if (!value)
      return rewriter.notifyMatchFailure(op, "user is not an atomic op");
    // TODO: just not a child value, need analysis
    if (value == loadOp.getValue())
      return rewriter.notifyMatchFailure(op, "another op is not");

    if (std::distance(userOp->getUsers().begin(), userOp->getUsers().end()) != 1)
      return rewriter.notifyMatchFailure(op, "user has more than one user");
    auto storeOp = dyn_cast<UpdateOp>(*userOp->getUsers().begin());
    if (!storeOp)
      return rewriter.notifyMatchFailure(op, "user is not a store op");
    if (storeOp.getTable() != loadOp.getTable())
      return rewriter.notifyMatchFailure(op, "store and load are not on the same table");

    // we get the load, atomic op and store op, erase backward
    rewriter.setInsertionPoint(storeOp);
    rewriter.replaceOpWithNewOp<UpdateAtomicOp>(
        storeOp, loadOp.getTable(), loadOp.getKey(),
        rewriter.getI32IntegerAttr(atomicOpCode), value);
    rewriter.eraseOp(userOp);
    return success();
  }
};

} // local namespace

void AtomicIdentificationPass::runOnOperation() {
    auto moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<LoadModifyWritePattern>(&getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    moduleOp.walk([&](FuncOp funcOp) {
      if (!funcOp->getAttr("scope"))
        return;
      // TODO(zhiyuang): keep track of interested gloabl vars
      if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns)))
        signalPassFailure();

      tryRemoveSync(funcOp);
    });
}

} // namespace ep2
} // namespace mlir