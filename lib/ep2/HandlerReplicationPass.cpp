
#include "mlir/IR/BuiltinDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include <map>

using namespace mlir;

namespace mlir {
namespace ep2 {

void HandlerReplicationPass::runOnOperation() {
  auto module = getOperation();

  std::vector<mlir::Operation*> toInsert;
  mlir::Block* parentBlock = nullptr;

  module->walk([&](ep2::FuncOp funcOp) {
    if (funcOp->getAttr("type").cast<StringAttr>().getValue() == "handler" && !funcOp.isExtern()) {
      parentBlock = funcOp->getBlock();
      int sz = cast<mlir::ArrayAttr>(funcOp->getAttr("instances")).size();
      mlir::Operation::CloneOptions options(true, true);

      std::string name = funcOp.getSymName().str();
      for (int i = 1; i<sz+1; ++i) {
        mlir::Operation* cloneOp = funcOp->clone(options);
        cast<ep2::FuncOp>(cloneOp).setSymName(name + "_" + std::to_string(i));
        toInsert.push_back(cloneOp);
      }
      funcOp->erase();
    }
  });

  for (mlir::Operation* op : toInsert) {
    parentBlock->push_back(op);
  }

  /*
  module->walk([&](ep2::FuncOp funcOp) {
    if (funcOp->getAttr("type").cast<StringAttr>().getValue() == "handler" && !funcOp.isExtern()) {
      funcOp->walk[&](ep2::ReturnOp retOp) {
        if (retOp->getOperands().size() != 1) {
          return;
        }
        assert(retOp->getOperand(0).getDefiningOp() != nullptr && isa<ep2::InitOp>(retOp->getOperand(0).getDefiningOp()));

        // encode round-robin, partitioning, which targets, etc. here
        cast<ep2::InitOp>(retOp->getOperand(0).getDefiningOp())->setAttr("enqInfo", rewriter.getStringAttr("INFO_TODO"));
      };
    }
  });

  module->walk([&](ep2::FuncOp funcOp) {
    if (funcOp.isExtern()) {
      return;
    }

    // map from each queue to which return statements feed it
    std::map<std::pair<std::string, int>>, std::vector<ep2::ReturnOp>> qMap;

    if (funcOp->getAttr("type").cast<StringAttr>().getValue() == "controller" && !funcOp.isExtern()) {
      funcOp->walk([&](ep2::ConnectOp op) {
        assert(op.getMethod().cast<mlir::StringAttr>().getValue() == "Queue");
        std::vector<int> replicas;
        for (mlir::Value arg : op.getOuts()) {
          auto portOut = cast<ep2::ConstantOp>(arg.getDefiningOp()).getValue().cast<ep2::PortAttr>();
          for (mlir::Value arg : op.getIns()) {
            auto portIn = cast<ep2::ConstantOp>(arg.getDefiningOp()).getValue().cast<ep2::PortAttr>();
          }
        }
      });
    }
  }
  */
}

} // namespace ep2
} // namespace mlir
