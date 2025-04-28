#include "ep2/passes/Mapping.h"

namespace mlir {
namespace ep2 {

namespace {

int getLatencyForUnit(PerformanceModel &model, int idx,
                      llvm::SmallVector<ep2::FuncOp> &ops,
                      PerformanceModel::UnitMap &unitMap) {
  auto func = ops[idx];
  auto commLatency = 0;
  if (idx < ops.size() - 1) {
    commLatency = model.getCommunicationCost(unitMap[idx], unitMap[idx + 1]);
  }
  auto totalLatency = model.getLatency(func) + commLatency;
  return totalLatency;
}

} // namespace

// Simple greedy default mapping
// This function provides a simple, greedy mapping method
PerformanceModel::MappingResult
PerformanceModel::getMapping(llvm::SmallVector<ep2::FuncOp> &ops) {
  // try to assign compute units one by one, and make sure each get enough latency
  auto latencyTarget = getLatencyTarget();
  auto units = getComputeUnits();
  PerformanceModel::UnitMap unitMap;

  for (int idx = 0; idx < ops.size(); idx++)
    unitMap[idx] = {};

  for (auto unit : units) {
    bool assigned = false;
    for (int idx = 0; idx < ops.size(); idx++) {
      auto totalLatency = getLatencyForUnit(*this, idx, ops, unitMap);

      // we need to add more units
      if (totalLatency > latencyTarget * unitMap[idx].size()) {

        // constraint on the number of units
        // TODO(zhiyuang): if its not tbale clean, we cannot replicate. check this.
        if (!isTableClean(ops[idx]) && unitMap[idx].size() >= 1)
          continue;

        unitMap[idx].push_back(unit);

        assigned = true;
        break;
      }
    }


    // if we do not need more unit, we can stop
    if (!assigned)
      break;
  }

  int targetIndex = 0, maxLatency = 0;
  for (int idx = 0; idx < ops.size(); idx++) {
    auto totalLatency = getLatencyForUnit(*this, idx, ops, unitMap) / unitMap[idx].size();
    if (totalLatency > maxLatency) {
      maxLatency = totalLatency;
      targetIndex = idx;
    }
  }

  // we run out of units ...
  return {maxLatency, targetIndex, unitMap};
}

// NetroNome is a simple performance model that assumes a core execution

// int SimpleNetronomePerformanceModel::getAccessOverhead() {
//   // get the diameter of the global
//   return 1;
// }
// 
// int SimpleNetronomePerformanceModel::getLatency(ep2::FuncOp funcOp) {
//   // get the diameter of the function
// 
//   for (auto &block : funcOp.getBlocks()) {
//     int cycles = 0;
//     for (auto &op : block) {
//       cycles += TypeSwitch<Operation *, int>(&op)
//         // zero width ops
//         .Case<ep2::InitOp, ep2::TerminateOp, ep2::ExtractOp>([](auto op) { return 0; })
//         .Case<ep2::EmitOp>([](auto op) { return 1; })
//         .Default([](auto op) { return 1; });
//         ;
//     }
//     return cycles;
//   }
// }

} // namespace ep2
} // namespace mlir