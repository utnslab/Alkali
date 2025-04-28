#include "ep2/passes/Mapping.h"
#include "ep2/dialect/Passes.h"

#include <queue>

namespace mlir {
namespace ep2 {


// The main optimization loop
void optimizationLoop(FuncOp funcOp, PipelineMapper &mapper, PipelineCutExplorer &explorer) {
    std::queue<HandlerPipeline> pipelines;
    pipelines.push({funcOp});

    int numNonOpts = 0;
    size_t optPerformance = 0;

    while (!pipelines.empty()) {
        auto &pipeline = pipelines.front();
        auto [updated, result] = mapper.tryMap(pipeline);

        llvm::errs() << "Searching Pipeline size: " + std::to_string(pipeline.size()) + "\n";
        for (auto &funcOp : pipeline) {
            llvm::errs() << funcOp.getName() << " ";
        }
        llvm::errs() << "Result Latency: " << result.latency << "\n";
        llvm::errs() << "Result Mapping: \n";
        for (auto [index, unit] : result.unitMap) {
            llvm::errs() << "Function: " << pipeline[index].getName() << " Unit: ";
            for (auto &u : unit) {
                llvm::errs() << u << " ";
            }
            llvm::errs() << "\n";
        }

        if (updated)
            numNonOpts = 0;
        else
            numNonOpts++;
        
        if (numNonOpts >= PIPELINE_EXTRA_SEARCH)
            break;
        
        // cleanup
        auto nexts = explorer.next(pipeline, result.bottleneckIndex);
        for (auto &next : nexts)
            pipelines.push(std::move(next));

        pipelines.pop();
    }
}

} // namespace mlir
} // namespace ep2