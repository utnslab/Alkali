#include "ep2/passes/Mapping.h"
#include "ep2/dialect/Passes.h"

namespace mlir {
namespace ep2 {

void simpleMapping(HandlerPipeline &pipeline,
                   llvm::SmallVector<int> *replications) {
  // by default, we do a linear mapping
  llvm::SmallVector<int> defaultReplications(pipeline.size(), 1);
  if (!replications)
    replications = &defaultReplications;

  // TODO: pass error out
  if (replications->size() != pipeline.size())
    llvm_unreachable("replications size must match pipeline size");

  for (auto [funcOp, rep] : llvm::zip(pipeline, *replications)) {
    OpBuilder builder(funcOp);
    // TODO: keep this?
    if (!funcOp->hasAttr("instances")) {
      llvm::SmallVector<StringRef> instanceVec{rep, "any"};
      auto instances = builder.getStrArrayAttr(instanceVec);
      funcOp->setAttr("instances", instances);
    }
  }
}

void preMappingCanonicalize(HandlerPipeline &pipeline) {
  StringRef eventName, atomName;

  for (size_t i = 0; i < pipeline.size(); i++) {
    auto funcOp = pipeline[i];
    OpBuilder builder(funcOp);

    if (i == 0) {
      if (!funcOp->hasAttr("atom"))
        funcOp->setAttr("atom", builder.getStringAttr("main"));
      atomName = funcOp->getAttrOfType<StringAttr>("atom").getValue();
      
      if (!funcOp->hasAttr("event"))
        funcOp->setAttr("event", builder.getStringAttr("Main"));
      eventName = funcOp->getAttrOfType<StringAttr>("event").getValue();
    } else { // splited names
      auto newEvent = eventName + "_" + std::to_string(i);
      funcOp->setAttr("event", builder.getStringAttr(newEvent));
      funcOp->setAttr("atom", builder.getStringAttr(atomName));
      HandlerDependencyAnalysis::HandlerFullName fullname(funcOp);

      // rename all the init ops
      auto prevFunc = pipeline[i - 1];
      llvm::SmallVector<ep2::InitOp> toRename;
      prevFunc.walk([&](ep2::InitOp initOp) {
        auto eventType = dyn_cast<StructType>(initOp.getType());
        if (!eventType || !eventType.getIsEvent())
          return;
        // llvm::errs() << "Checking " << initOp << "\n";
        // llvm::errs() << "funcName" << funcOp.getName() << "\n";
        // TODO: rename at gen
        toRename.push_back(initOp);
      });

      for (auto initOp : toRename) {
        auto eventType = cast<ep2::StructType>(initOp.getType());
        builder.setInsertionPoint(initOp);
        // add atom
        llvm::SmallVector<Value> newArgs;
        newArgs.push_back(builder.create<ep2::ConstantOp>(
          initOp.getLoc(), atomName
        ));
        for (auto arg : initOp.getArgs())
          newArgs.push_back(arg);

        auto newTypeArgs = llvm::map_to_vector(newArgs, [&](Value arg) {
          return arg.getType();
        });
        auto newType = builder.getType<ep2::StructType>(true, newTypeArgs, fullname.event);
        auto newInit = builder.create<ep2::InitOp>(initOp.getLoc(), newType, newArgs);
        auto newReturn = builder.create<ep2::ReturnOp>(initOp.getLoc(), ValueRange{newInit});
        // remove old use chain
        for (auto op : initOp->getUsers())
          op->erase();
        initOp->erase();
      }

    }

    // set generate name
    HandlerDependencyAnalysis::HandlerFullName fullname(funcOp);
    funcOp.setName(fullname.mangle());
  }
}

void insertController(HandlerPipeline &pipeline) {
  for (size_t i = 1; i < pipeline.size(); i++) {
    auto prevFuncOp = pipeline[i - 1];
    auto funcOp = pipeline[i];
    HandlerDependencyAnalysis::HandlerFullName prevName(prevFuncOp), curName(funcOp);

    OpBuilder builder(funcOp);
    builder.setInsertionPointAfter(prevFuncOp);

    auto controllerName = "__controller_" + curName.event.str();

    auto controller = builder.create<ep2::FuncOp>(
      funcOp.getLoc(),
      controllerName,
      builder.getFunctionType({}, {}));
    controller->setAttr("type", builder.getStringAttr("controller"));
    controller->setAttr("event", builder.getStringAttr(curName.event));

    builder.setInsertionPointToStart(&controller.getBody().front());

    // Currently only FPGA mode is supported
    llvm::SmallVector<Value> ins, outs;
    // make sure we have the "instances" attr here
    for (auto [index, _] : llvm::enumerate(prevFuncOp->getAttrOfType<ArrayAttr>("instances"))) {
      ins.push_back(builder.create<ep2::ConstantOp>(
        funcOp.getLoc(),
        builder.getType<ep2::PortType>(true, false),
        builder.getAttr<ep2::PortAttr>(prevName.event, prevName.atom, index)
      ));
    }

    for (auto [index, _] : llvm::enumerate(funcOp->getAttrOfType<ArrayAttr>("instances"))) {
      outs.push_back(builder.create<ep2::ConstantOp>(
        funcOp.getLoc(),
        builder.getType<ep2::PortType>(false, true),
        builder.getAttr<ep2::PortAttr>(curName.event, curName.atom, index)
      ));
    }

    // TODO: How to connect them? netronome mode?
    builder.create<ep2::ConnectOp>(
      funcOp.getLoc(),
      "Queue",
      ins, outs, 
      builder.getArrayAttr({
        builder.getI64IntegerAttr(32)
      })
    );

    builder.create<ep2::TerminateOp>(funcOp.getLoc());
  }

}

} // namespace ep2
} // namespace mlir