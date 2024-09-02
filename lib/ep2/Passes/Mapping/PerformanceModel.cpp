#include "ep2/passes/Mapping.h"

namespace mlir {
namespace ep2 {


// NetroNome is a simple performance model that assumes a core execution

int SimpleNetronomePerformanceModel::getAccessOverhead(ep2::GlobalOp globalOp) {
  // get the diameter of the global
  return 1;
}

int SimpleNetronomePerformanceModel::getLatency(ep2::FuncOp funcOp) {
  // get the diameter of the function

  for (auto &block : funcOp.getBlocks()) {
    int cycles = 0;
    for (auto &op : block) {
      cycles += TypeSwitch<Operation *, int>(&op)
        // zero width ops
        .Case<ep2::InitOp, ep2::TerminateOp, ep2::ExtractOp>([](auto op) { return 0; })
        .Case<ep2::EmitOp>([](auto op) { return 1; })
        .Default([](auto op) { return 1; });
        ;
    }
    return cycles;
  }
}

} // namespace ep2
} // namespace mlir