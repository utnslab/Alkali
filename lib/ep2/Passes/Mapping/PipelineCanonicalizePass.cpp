#include "ep2/passes/Mapping.h"
#include "ep2/dialect/Passes.h"

namespace mlir {
namespace ep2 {

void PipelineCanonicalizePass::runOnOperation() {
  auto module = getOperation();
  llvm::SmallVector<ep2::FuncOp> pipeline;
  module->walk([&](FuncOp funcOp) {
    if (funcOp.isHandler() && !funcOp.isExtern())
      pipeline.push_back(funcOp);
  });

  preMappingCanonicalize(pipeline);

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
  insertController(pipeline);
}

} // namespace ep2
} // namespace mlir