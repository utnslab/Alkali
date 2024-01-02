#ifndef EP2_DIALECT_COSTMODEL_H
#define EP2_DIALECT_COSTMODEL_H

#include <type_traits>
#include "ep2/dialect/Dialect.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

using ThroughputT = int;
using TickT = int;
using ResourceT = int;

struct ArchSpec {
  struct Unit { int id; };
  struct MemoryUnit : public Unit {
    ResourceT mem;
    MemoryUnit(const llvm::json::Value &value) {
      llvm::json::Path::Root root("root");
      llvm::json::ObjectMapper mapper(value, root);
      assert(mapper && mapper.map("id", id) && mapper.map("mem", mem) &&
             "construct memory object failed");
    }
  };
  struct ComputeUnit : public Unit {
    ResourceT cpu, mem;
    ComputeUnit(const llvm::json::Value &value) {
      llvm::json::Path::Root root("root");
      llvm::json::ObjectMapper mapper(value, root);
      assert(mapper && mapper.map("id", id) && mapper.map("mem", mem) &&
             mapper.map("cpu", cpu) && "construct compute unit failed");
    }
  };

  std::tuple<std::vector<ComputeUnit>, std::vector<MemoryUnit>> units;

  template <typename T, typename = std::enable_if_t<std::is_base_of_v<Unit, T>>>
  size_t getNumUnit() {
    return std::get<std::vector<T>>(units).size();
  }

  template <typename T, typename = std::enable_if_t<std::is_base_of_v<Unit, T>>>
  T &get(int id) {
    return std::get<std::vector<T>>(units)[id];
  }
  template <typename T, typename = std::enable_if_t<std::is_base_of_v<Unit, T>>>
  std::vector<T> &getVec() {
    return std::get<std::vector<T>>(units);
  }

  // data
  std::vector<std::vector<TickT>> communicationMatrix;
  std::vector<std::vector<TickT>> memoryMatrix;

  // Arch defined access ticks
  TickT communicationCost(int id1, int id2) {
    return communicationMatrix[id1][id2];
  }

  TickT memoryCost(int id1, int id2) {
    return memoryMatrix[id1][id2];
  }


  // import/export functions
  ArchSpec(llvm::StringRef filename) {
    // read from file. format: json
    auto file = llvm::MemoryBuffer::getFile(filename);
    assert(file && "file not found");

    llvm::Expected<llvm::json::Value> result =
        llvm::json::parse(file.get()->getBuffer());
    assert(result && "invalid arch spec json fromat");

    
    auto top = result->getAsObject();
    assert(top && "single object");

    auto computeUnits = top->getArray("computeUnits");
    assert(computeUnits && "computeUnits is an array");
    for (const auto &unit : *computeUnits)
      getVec<ComputeUnit>().emplace_back(unit);

    auto memoryUnits = top->getArray("memoryUnits");
    assert(computeUnits && "memoryUnit is an array");
    for (const auto &unit : *memoryUnits)
      getVec<MemoryUnit>().emplace_back(unit);

    // read communication matrix
    auto matrix = top->getArray("communicationMatrix");
    assert(matrix && "matrix");
    for (const auto &row : *matrix) {
      auto &rowVec = communicationMatrix.emplace_back();
      llvm::json::Path::Root root("root");
      assert(llvm::json::fromJSON(row, rowVec, root) && "matrix row");
    }
    assert(communicationMatrix.size() == getNumUnit<ComputeUnit>() &&
           llvm::all_of(communicationMatrix,
                        [&](const auto &row) {
                          return row.size() == getNumUnit<ComputeUnit>();
                        }) &&
           "communication matrix size");

    auto matrix2 = top->getArray("communicationMatrix");
    assert(matrix2 && "matrix2");
    for (const auto &row : *matrix2) {
      auto &rowVec = memoryMatrix.emplace_back();
      llvm::json::Path::Root root("root");
      assert(llvm::json::fromJSON(row, rowVec, root) && "matrix row");
    }
    assert(communicationMatrix.size() == getNumUnit<ComputeUnit>() &&
           llvm::all_of(communicationMatrix,
                        [&](const auto &row) {
                          return row.size() == getNumUnit<MemoryUnit>();
                        }) &&
           "memory matrix size");
  }

};



/// define a cost model for a given architecture
/// All values in the model should be the normalized cost in unit of packets (PPS)
/// An overall controller further converts this to a bandwidth in the overall constriant system
struct CostModel {
  static constexpr TickT unitTick = 1000000;

  CostModel(ArchSpec &spec) : spec(spec) {};
  virtual ResourceT computeResource(mlir::ep2::FuncOp funcOp) = 0;

  virtual TickT computationCost(mlir::ep2::FuncOp funcOp, TickT factor = 1) = 0;

  template<typename T>
  TickT memoryCost(mlir::ep2::FuncOp funcOp, T &&opCost, TickT factor = 1);

protected:
  ArchSpec &spec;
};

/// For replications, first we find the subgraph that contains a subgraph
/// For branches, we do not know

struct SimpleCostModel : public CostModel {
  SimpleCostModel(ArchSpec &spec) : CostModel(spec) {};

  ResourceT computeResource(mlir::ep2::FuncOp funcOp) override {
    // count the instructions
    int count = 0;
    for (auto &block : funcOp.getBody()) {
      for (auto &_ : block) {
        // filter on the op type
        count ++;
      }
    }
    return count;
  }

  TickT computationCost(mlir::ep2::FuncOp funcOp, TickT unitCost = 1) override {
    // TODO: consider branch
    // count the instructions
    int count = 0;
    for (auto &block : funcOp.getBody()) {
      for (auto &op : block) {
        // filter on the op type
        count ++;
      }
    }
    return count * unitCost;
  }

  // base function for calculating memory cost
  template<typename T>
  TickT memoryCost(mlir::ep2::FuncOp funcOp, T &&opCost, TickT factor = 1) {
    return 0;
  }

};

#endif // EP2_DIALECT_COSTMODEL_H
