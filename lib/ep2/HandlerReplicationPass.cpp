
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

/*
Replicate handlers based on instances attribute on original handler. Used to
specialize per replicated instance later.
*/

using namespace mlir;

namespace mlir {
namespace ep2 {

enum SprayType {
  UNSUPPORTED,
  ROUND_ROBIN,
  PARTITION,
};

struct SprayInfo {
  SprayType type;
  std::string desc;

  SprayInfo() {}
  SprayInfo(SprayType t, std::string d) : type(t), desc(d) {}
  
  std::string toString() {
    std::string ty;
    switch (type) {
      case SprayType::ROUND_ROBIN:
        ty = "ROUND_ROBIN";
        break;
      case SprayType::PARTITION:
        ty = "PARTITION";
        break;
      default:
        assert(false);
        break;
    }
    return ty + " " + desc;
  }
};

void HandlerReplicationPass::runOnOperation() {
  auto module = getOperation();

  std::vector<mlir::Operation*> toInsert;
  mlir::Block* parentBlock = nullptr;
  mlir::Builder builder(&getContext());

  /*
  Clone handlers, set location attribute for each. E.g. if handler
  is replicated onto island 1, compute units 1,2 then location on
  handler 1 = i1cu1, handler 2 has i1cu2.
  */
  module->walk([&](ep2::FuncOp funcOp) {
    if (funcOp->getAttr("type").cast<StringAttr>().getValue() == "handler" && !funcOp.isExtern()) {
      parentBlock = funcOp->getBlock();
      llvm::errs() << funcOp.getSymName().str() << '\n';
      auto instances = cast<mlir::ArrayAttr>(funcOp->getAttr("instances")).getValue();
      int sz = instances.size();
      mlir::Operation::CloneOptions options(true, true);

      std::string name = funcOp.getSymName().str();
      for (int i = 1; i<=sz; ++i) {
        mlir::Operation* cloneOp = funcOp->clone(options);
        cloneOp->setAttr("location", instances[i-1]);
        
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
  qMap maintains mapping between pair{specific_in_handler, unreplicated_out_handler} to how we transmit
  information between the 2- a SprayInfo to tell whether RR or partitioning, and
  which replicas of out_handler we spray to.
  */
  std::map<std::pair<std::string, std::string>, std::pair<SprayInfo, llvm::SmallVector<int>>> qMap;
  module->walk([&](ep2::FuncOp funcOp) {
    if (funcOp.isExtern()) {
      return;
    }

    auto portToHandlerName = [](ep2::PortAttr port) {
      return std::string{"__handler_"} + port.getHandler().str() + "_" + port.getAtom().str() + "_" + std::to_string(port.getInstance()+1);
    };

    if (funcOp->getAttr("type").cast<StringAttr>().getValue() == "controller" && !funcOp.isExtern()) {
      funcOp->walk([&](ep2::ConnectOp op) {
        assert(op.getMethod() == "Queue" || op.getMethod() == "PartitionByScope");

        for (size_t i = 0; i<op.getOuts().size(); ++i) {
          mlir::Value outArg = op.getOuts()[i];
          auto portOut = cast<ep2::ConstantOp>(outArg.getDefiningOp()).getValue().cast<ep2::PortAttr>();
          for (mlir::Value arg : op.getIns()) {
            auto portIn = cast<ep2::ConstantOp>(arg.getDefiningOp()).getValue().cast<ep2::PortAttr>();
            if (!funcOp->hasAttr("prevEvent")) {
              funcOp->setAttr("prevEvent", builder.getStringAttr(portIn.getHandler()));
            }
            auto k = std::pair<std::string, std::string>{portToHandlerName(portIn), portOut.getHandler().str()};
            if (op.getMethod() == "Queue") {
              qMap[k].first = SprayInfo(SprayType::ROUND_ROBIN, "");
            } else if (op.getMethod() == "PartitionByScope") {
              qMap[k].first = SprayInfo(SprayType::PARTITION, op.getParameter<mlir::StringAttr>(0).getValue().str());
            }
            qMap[k].second.push_back(1 + portOut.getInstance());
          }
         }
      });
    }
  });

  /*
  Walk through return statements per handler to determine which events we generate,
  and mark on the initOp feeder of that returnOp those events (since we use InitOp's,
  not ReturnOp's, for EmitC generation later.
  */
  module->walk([&](ep2::FuncOp funcOp) {
    if (funcOp->getAttr("type").cast<StringAttr>().getValue() == "handler" && !funcOp.isExtern()) {
      funcOp->walk([&](ep2::ReturnOp retOp) {
        if (retOp->getOperands().size() != 1) {
          return;
        }
        assert(retOp->getOperand(0).getDefiningOp() != nullptr && isa<ep2::InitOp>(retOp->getOperand(0).getDefiningOp()));
        ep2::InitOp initOp = cast<ep2::InitOp>(retOp->getOperand(0).getDefiningOp());

        std::string handlerName = funcOp.getName().str();
        std::string outEventName = cast<ep2::StructType>(initOp.getOutput().getType()).getName().str();
        auto k = std::pair<std::string, std::string>{handlerName, outEventName};

        if (qMap.find(k) != qMap.end()) {
          initOp->setAttr("enqInfo", builder.getI32ArrayAttr(qMap[k].second));
          initOp->setAttr("sprayInfo", builder.getStringAttr(qMap[k].first.toString()));
        }
      });
    }
  });
}

} // namespace ep2
} // namespace mlir
