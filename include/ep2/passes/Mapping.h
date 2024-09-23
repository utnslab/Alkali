#ifndef EP2_MAPPING_H
#define EP2_MAPPING_H

#include "ep2/dialect/Dialect.h"

namespace mlir {
namespace ep2 {

// handler a pipeline and insert controller
using HandlerPipeline = llvm::SmallVector<ep2::FuncOp>;

void simpleMapping(HandlerPipeline &pipeline, llvm::SmallVector<int> *replications = nullptr);
void simpleGlobalMapping(HandlerPipeline &pipeline, int localTableNumber = 0);

void preMappingCanonicalize(HandlerPipeline &pipeline, llvm::StringRef mode);
void insertController(HandlerPipeline &pipeline);


// netronome specific passes
void bufferToRef(HandlerPipeline &pipeline);
void contextIdentification(HandlerPipeline &pipeline);

// handler split and pipeline
struct PipelineResult {
  double sourceWeight;
  llvm::DenseSet<mlir::Operation*> sinkOps;
  llvm::DenseSet<mlir::Value> sinkValues;
  std::string err;
  bool dumpFile;
};
struct PipelinePolicy {
  double sourceWeight;
  double tolerance;
  // options
  bool done = false, dumpCuts = true;

  PipelinePolicy(double sourceWeight, double tolerance) : sourceWeight(sourceWeight), tolerance(tolerance) {}

  virtual int typeTransmitCost(mlir::Type t) = 0;
  virtual int operationWeight(mlir::Operation* op) = 0;
  virtual int valueWeight(mlir::Value v) = 0;

  virtual std::pair<std::shared_ptr<PipelinePolicy>, std::shared_ptr<PipelinePolicy>> splitPolicy(PipelineResult &result) = 0;
  virtual std::pair<std::string, std::string> splitName() = 0;
};
bool pipelineHandler(ep2::FuncOp funcOp, PipelinePolicy* policy, PipelineResult* results);

// performance model
class PerformanceModel {
 public:
  virtual int getAccessOverhead(ep2::GlobalOp globalOp) = 0;
  virtual int getLatency(ep2::FuncOp funcOp) = 0;
};

class SimpleNetronomePerformanceModel : public PerformanceModel {
 public:
  enum MemoryLat {
    LMEM = 1,
    CLS = 10,
    SMEM = 15
  };

  llvm::DenseMap<Operation *, MemoryLat> locationMap;

  int getAccessOverhead(ep2::GlobalOp globalOp) override;
  int getLatency(ep2::FuncOp funcOp) override;
};

} // namespace ep2
} // namespace mlir 





#endif // _MAPPING_H_