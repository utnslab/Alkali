#ifndef EP2_DIALECT_COSTMODEL_H
#define EP2_DIALECT_COSTMODEL_H

#include "ep2/dialect/Dialect.h"

struct ArchSpec {
    struct MemoryUnit;
    struct ComputeUnit;
};

/// define a cost model for a given architecture
/// All values in the model should be the normalized cost in unit of packets (PPS)
/// An overall controller further converts this to a bandwidth in the overall constriant system
struct CostModel {

  CostModel(ArchSpec &spec) : spec(spec) {};
  virtual double computationCost(mlir::ep2::FuncOp funcOp, ArchSpec::ComputeUnit &compute) = 0;
  virtual double communicationCost(mlir::ep2::FuncOp funcOp,
                                   ArchSpec::ComputeUnit &compute,
                                   mlir::ep2::FuncOp f2,
                                   ArchSpec::ComputeUnit c2) = 0;
  virtual double memoryCost(mlir::ep2::FuncOp funcOp,
                            ArchSpec::ComputeUnit &compute,
                            ArchSpec::MemoryUnit &mem) = 0;

protected:
  ArchSpec &spec;
}

/// For replications, first we find the subgraph that contains a subgraph
/// For branches, we do not know

#endif // EP2_DIALECT_COSTMODEL_H
