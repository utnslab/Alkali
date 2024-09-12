#include "ep2/passes/Mapping.h"
#include "ep2/dialect/Passes.h"

namespace mlir {
namespace ep2 {

void simpleMapping(HandlerPipeline &pipeline,
                   llvm::SmallVector<int> *replications) {
  // we keep track of a global CU number
  int cuNumber = 1;
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
      llvm::SmallVector<StringRef> instanceVec;
      for (int i = 0; i < rep; i++)
        instanceVec.push_back("cu" + std::to_string(cuNumber++));
      auto instances = builder.getStrArrayAttr(instanceVec);
      funcOp->setAttr("instances", instances);
    }
  }
}

void simpleGlobalMapping(HandlerPipeline &pipeline) {
  // we keep track of a global CU number
  for (auto funcOp : pipeline) {
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    OpBuilder builder(funcOp);
    funcOp.walk([&](ep2::GlobalImportOp importOp) {
      auto globalName = importOp.getName();
      ep2::GlobalOp globalOp = nullptr;
      moduleOp.walk([&](ep2::GlobalOp op) {
        if (op.getName() == globalName)
          globalOp = op;
      });
      if (!globalOp)
        llvm_unreachable("global not found for globalImport");
      auto globalAttrName = "instances";

      // all map to local
      auto instances = funcOp->getAttrOfType<ArrayAttr>(globalAttrName);
      auto instancesStrs = llvm::map_to_vector(instances, [&](Attribute attr) {
        return "cls_" + attr.cast<StringAttr>().getValue().str();
      });
      llvm::SmallVector<StringRef> globalInstancesStrs = llvm::map_to_vector(instancesStrs, [&](StringRef str) {
        return str;
      });

      // we do not allow. just check for now
      if (globalOp->hasAttr(globalAttrName))
        llvm::errs() << "global already has instances\n";
      globalOp->setAttr(globalAttrName,
        builder.getStrArrayAttr(globalInstancesStrs));
    });
  }
}

void contextIdentification(HandlerPipeline &pipeline) {
  std::string contextName = "context";
  int contextIdx = 0;

  llvm::SmallVector<DenseMap<int, std::string>> funcContextNames(
      pipeline.size());

  // start from second stage, to last stage
  for (size_t i = 1; i < pipeline.size(); i++) {
    OpBuilder builder(pipeline[i]);
    auto funcOp = pipeline[i];
    auto &contextNames = funcContextNames[i];

    for (size_t j = 0; j < funcOp.getNumArguments(); j++) {
      auto arg = funcOp.getArgument(j);
      // TODO(zhiyaung): FIXTHIS we force this to be a struct type for now
      if (isa<ep2::BufferType>(arg.getType())) {
        // limited by lib/ep2/LocalAllocAnalysis.cpp:36
        // do not store the buf type
        continue;
      }
      if (isa<ep2::StructType>(arg.getType())) {
        auto contextName = "context" + std::to_string(contextIdx++);
        funcOp.setArgAttr(j, "ep2.context_name",
                          builder.getStringAttr(contextName));
        contextNames.try_emplace(j, contextName);
      }

      // we only mark init before the final pipeline stage
      if (i + 1 < pipeline.size()) {
        auto &nextContextNames = funcContextNames[i + 1];
        // find the init op and try to propogate...
        for (auto &use : arg.getUses()) {
          auto initOp = dyn_cast<ep2::InitOp>(use.getOwner());
          if (!initOp)
            continue;

          auto structType = dyn_cast<ep2::StructType>(initOp.getType());
          if (!structType || !structType.getIsEvent())
            continue;
          
          // we have a generate usage. try to assign a value
          auto it = contextNames.find(j);
          if (it == contextNames.end()) {
            std::string contextName = "context" + std::to_string(contextIdx++);
            funcOp.setArgAttr(j, "ep2.context_name",
                              builder.getStringAttr(contextName));
            it = contextNames.try_emplace(j, contextName).first;
          }

          // with/without atom
          nextContextNames.try_emplace(use.getOperandNumber() - 1, it->second);
        }
      }
    }
  }

  // we finished the context identification, mark the context
  for (size_t i = 0; i < pipeline.size(); i++) {
    auto funcOp = pipeline[i];
    auto &contextMap = funcContextNames[i];

    OpBuilder builder(funcOp);
    // set the functions' parameter, except the first one
    if (i != 0) {
      for (size_t j = 0; j < funcOp.getNumArguments(); j++) {
        auto it = contextMap.find(j);
        if (it != contextMap.end())
          funcOp.setArgAttr(j, "ep2.context_name", builder.getStringAttr(it->second));
      }
    }

    // except the last one, set init attr
    if (i + 1 < pipeline.size()) {
      auto nextFunc = pipeline[i + 1];
      auto &nextContextMap = funcContextNames[i + 1];
      funcOp.walk([&](ep2::InitOp initOp){
        auto eventType = dyn_cast<StructType>(initOp.getType());
        if (!eventType || !eventType.getIsEvent())
          return;

        HandlerDependencyAnalysis::HandlerFullName fullname(initOp);
        if (fullname.mangle() != nextFunc.getName())
          return;

        // push for atom first
        llvm::SmallVector<StringRef> contextNames;
        contextNames.push_back("");
        for (size_t j = 1; j < initOp.getNumOperands(); j++) {
          auto it = nextContextMap.find(j - 1);
          if (it != nextContextMap.end())
            contextNames.push_back(it->second);
          else
            contextNames.push_back("");
        }
        initOp->setAttr("context_names", builder.getStrArrayAttr(contextNames));
      });
    }
  }
}

void bufferToRef(HandlerPipeline &pipeline) {
  for (auto funcOp : pipeline) {
    OpBuilder builder(funcOp);
    funcOp.walk([&](Operation *op) {
      builder.setInsertionPointAfter(op);
      if (auto bufferOp = dyn_cast<ep2::EmitValueOp>(op)) {
        builder.create<ep2::EmitOp>(bufferOp.getLoc(), bufferOp.getBuffer(),
                                    bufferOp.getValue());
        bufferOp->replaceAllUsesWith(ValueRange{
          bufferOp.getBuffer()
        });
      } else if (auto bufferOp = dyn_cast<ep2::ExtractValueOp>(op)) {
        auto extractOp = builder.create<ep2::ExtractOp>(bufferOp.getLoc(),
                                       bufferOp.getOutput().getType(),
                                       bufferOp.getBuffer());
        bufferOp->replaceAllUsesWith(ValueRange{
          bufferOp.getBuffer(), extractOp.getOutput()
        });
      }
    });
  }
}

void preMappingCanonicalize(HandlerPipeline &pipeline, StringRef mode) {
  std::string eventName, atomName;

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
      // XXX(zhiyuang): netronome backend requires atom name to be unique
      // WTF
      auto newEvent = eventName + "_" + std::to_string(i);
      auto newAtomName = atomName + "_" + std::to_string(i);
      funcOp->setAttr("event", builder.getStringAttr(newEvent));
      if (mode == "netronome")
        funcOp->setAttr("atom", builder.getStringAttr(newAtomName));
      else
        funcOp->setAttr("atom", builder.getStringAttr(atomName));

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

      HandlerDependencyAnalysis::HandlerFullName fullname(funcOp);
      for (auto initOp : toRename) {
        builder.setInsertionPoint(initOp);
        // add atom
        llvm::SmallVector<Value> newArgs;
        newArgs.push_back(builder.create<ep2::ConstantOp>(
          initOp.getLoc(), fullname.atom
        ));
        for (auto arg : initOp.getArgs())
          newArgs.push_back(arg);

        auto newTypeArgs = llvm::map_to_vector(newArgs, [&](Value arg) {
          return arg.getType();
        });
        auto newType = builder.getType<ep2::StructType>(true, newTypeArgs, fullname.event);
        auto newInit = builder.create<ep2::InitOp>(initOp.getLoc(), newType, newArgs);
        builder.create<ep2::ReturnOp>(initOp.getLoc(), ValueRange{newInit});
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