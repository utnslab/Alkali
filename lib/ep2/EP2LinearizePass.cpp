#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace ep2{

namespace {

  Block * getDefiningBlock(Value value) {
    return TypeSwitch<Value, Block *>(value)
        .Case<BlockArgument>([&](BlockArgument arg) {
          return arg.getOwner();
        })
        .Default([&](Value value) {
          return value.getDefiningOp()->getBlock();
        });
  }

  bool foldPredicates(Block *cur, Block *entryBlock,
                      llvm::SmallVector<Value> &preds, llvm::SmallVector<bool> &predAttrs) {
    for (;cur != entryBlock; cur = cur->getSinglePredecessor()) {
      auto pred = cur->getSinglePredecessor();
      if (!pred)
        return false;

      bool foldable =
          TypeSwitch<Operation *, bool>(pred->getTerminator())
              .Case<cf::BranchOp>([&](auto op) { return true; })
              .Case<cf::CondBranchOp>([&](cf::CondBranchOp condOp) {
                // TODO(zhiyuang): for now we lift inputs. But we could also
                // lift outputs
                preds.push_back(condOp.getCondition());
                predAttrs.push_back(condOp.getTrueDest() == cur);
                return entryBlock == getDefiningBlock(condOp.getCondition());
              })
              .Default([&](Operation *op) {
                llvm::errs() << "Unexpected terminator: " << *op << "\n";
                llvm_unreachable("Unexpected terminator");
                return false;
              });
      if (!foldable)
        return false;
    }
    return true;
  }

  struct LiftPurePattern : public OpRewritePattern<FuncOp> {
    struct OpAction {
      Operation *op;
      bool guard; // 1->move and guard. 2->move
      // following fields only for guard == 1
      llvm::SmallVector<Value> preds;
      llvm::SmallVector<bool> predAttrs;
      OpAction(Operation *op, bool guard, llvm::SmallVector<Value> preds = {},
               llvm::SmallVector<bool> predAttrs = {})
          : op(op), guard(guard), preds(std::move(preds)),
            predAttrs(std::move(predAttrs)) {}
    };

    using OpRewritePattern<FuncOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(FuncOp funcOp,
                                  PatternRewriter &rewriter) const final {
      auto &entryBlock = funcOp.getBlocks().front();
      bool changed = false;
      for (auto &block : funcOp.getFunctionBody()) {
        if (block.hasNoPredecessors())
          continue;
        std::vector<OpAction> toMove{};
        for (auto &opRef : block) {
          auto op = &opRef;
          if (isa<cf::BranchOp, cf::CondBranchOp, ep2::TerminateOp>(op))
            continue;

          // we check if all operands are defined in the entry block
          if (!llvm::all_of(op->getOperands(), [&](Value operand) {
                return &entryBlock == getDefiningBlock(operand);
              }))
            continue;

          if (isPure(op))
            toMove.emplace_back(op, false);
          else if (isa<ReturnOp, UpdateOp, UpdateAtomicOp>(op)) {
            // We could move (and guard) those non-Pure Ops, as long as it do not have any return value
            // we only need to transfer one level: the blocks connects to the entry block
            llvm::SmallVector<Value> preds;
            llvm::SmallVector<bool> predAttrs;
            if (foldPredicates(&block, &entryBlock, preds, predAttrs))
              toMove.emplace_back(op, true, std::move(preds),
                                  std::move(predAttrs));
          }
        }

        llvm::errs() << "toMove size: " << toMove.size() << " | " << funcOp.getName() << "\n";
        for (auto &action : toMove) {
          action.op->dump();
        }

        changed |= !toMove.empty();
        for (auto &action : toMove) {
          if (action.guard) {
            rewriter.setInsertionPoint(entryBlock.getTerminator());
            auto guardOp = rewriter.create<GuardOp>(
                action.op->getLoc(), action.preds,
                rewriter.getBoolArrayAttr(action.predAttrs));
            auto &guardBlock = guardOp.getBody().emplaceBlock();
            action.op->moveBefore(&guardBlock, guardBlock.end());
          } else
            action.op->moveBefore(entryBlock.getTerminator());
        }
      }
      return success(changed);
    }
  };
} // local namespace


void EP2LinearizePass::runOnOperation() {
  auto moduleOp = getOperation();

  RewritePatternSet patterns(&getContext());
  patterns.add<LiftPurePattern>(&getContext());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  OpPassManager pm;
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  bool changed = false;
  GreedyRewriteConfig config{};
  config.maxIterations = GreedyRewriteConfig::kNoLimit;

  do {
    // try to lift all ops from the branches
    if (failed(applyPatternsAndFoldGreedily(moduleOp, frozenPatterns,
                                            config, &changed))) {

      moduleOp->emitError("Unexpected failure in lifting pure ops");
      return signalPassFailure();
    }

    // try to fold unused BBs to arith.select
    if (failed(runPipeline(pm, moduleOp))) {
      moduleOp->emitError("Unexpected failure in canonicalization");
      return signalPassFailure();
    }
  } while (changed);
}

} // namespace ep2
} // namespace mlir