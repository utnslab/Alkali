#include "ep2/passes/Mapping.h"
#include "ep2/dialect/Passes.h"

namespace mlir {
namespace ep2 {

// only for FPGA
static void addTableInlineAttribute(HandlerPipeline &pipeline) {
  for (auto &funcOp : pipeline) {
    funcOp->walk([&](ep2::GlobalImportOp importOp) {
      auto globalName = importOp.getName();
      auto globalAttrName = "instances_" + globalName.str();

      funcOp->setAttr(globalAttrName, BoolAttr::get(funcOp->getContext(), true));
    });
  }
}

static void placeGeneration(DenseMap<int, FuncOp> &funcMap, int generation,
                            HandlerPipeline &pipeline) {
  // the handler is not splited
  auto it = funcMap.find(generation);
  if (it != funcMap.end())
    pipeline.push_back(it->second);
  else {
    placeGeneration(funcMap, generation * 2, pipeline);
    placeGeneration(funcMap, generation * 2 + 1, pipeline);
  }
}

static void getPipeline(Operation *moduleOp, HandlerPipeline &pipeline) {
  DenseMap<int, FuncOp> funcMap;

  moduleOp->walk([&](FuncOp funcOp) {
    if (funcOp.isHandler() && !funcOp.isExtern()) {
      int index = 1;
      auto attr = funcOp->getAttrOfType<IntegerAttr>("generationIndex");
      if (attr)
        index = (int)attr.getInt();
      funcMap.try_emplace(index, funcOp);
    }
  });

  placeGeneration(funcMap, 1, pipeline);
}

void PipelineCanonicalizePass::runOnOperation() {
  auto module = getOperation();
  llvm::SmallVector<ep2::FuncOp> pipeline;
  getPipeline(module, pipeline);

  preMappingCanonicalize(pipeline, mode.getValue());

  if (replications.hasValue()) {
    llvm::SmallVector<StringRef> replicationsVec;
    StringRef ref{replications.getValue()};
    ref.split(replicationsVec, ",");
    auto ints = llvm::map_to_vector(replicationsVec, [](StringRef ref) {
      return std::stoi(ref.str());
    });
    if (ints.size() != pipeline.size()) {
      llvm::errs() << "replications size must match pipeline size\n";
      signalPassFailure();
      return;
    }

    simpleMapping(pipeline, &ints);
  } else
    simpleMapping(pipeline);

  // for netronome, we need to map gloabl state to local state
  if (mode.getValue() == "netronome") {
    simpleGlobalMapping(pipeline, limitLocalTable.getValue());
    bufferToRef(pipeline);
    // add the attribute for adding context
    // TODO(insert DPE, DFE passes)
    contextIdentification(pipeline);
  }

  // add the mode option for the controller
  insertController(pipeline);

  if (inlineTable.getValue())
    addTableInlineAttribute(pipeline);
}

} // namespace ep2
} // namespace mlir