#include "mlir/IR/BuiltinDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

#include "ep2/dialect/CostModel.h"

#include "z3++.h"

namespace mlir {
namespace ep2 {

namespace {

// Util functiosn
void mergeFuncs(FuncOp func1, FuncOp func2, RewriterBase &rewriter) {
  OpBuilder builder(func1);

  // change the mappings (generation and consumption)

}

class MappingSolver {
 public:
  template <typename T>
  struct CostMatrix {
    size_t dims[2];
    T** data;

    CostMatrix(size_t dim1, size_t dim2) : dims{dim1, dim2} {
      data = new T *[dim1];
      for (size_t i = 0; i < dim1; i++) {
        data[i] = new T[dim2];
      }
    }
    ~CostMatrix() {
      for (size_t i = 0; i < dims[0]; i++) {
        delete[] data[i];
      }
      delete[] data;
    }
    void set(size_t i, size_t j, T v) { data[i][j] = v; }
    z3::expr oneOf(z3::context &ctx, z3::expr index1, z3::expr index2) {
      z3::expr_vector matrix(ctx);
      for (size_t i = 0; i < dims[0]; i++)
        for (size_t j = 0; j < dims[1]; j++)
          matrix.push_back(z3::ite(index1 == i && index2 == j, ctx.int_val(data[i][j]), ctx.int_val(0)));
      return z3::sum(matrix);
    }
  };

  MappingSolver(AnalysisManager &am, ArchSpec &spec, CostModel &costModel)
      : ctx{}, solver{ctx}, spec(spec), costModel(costModel),
        communicationMatrix{spec.getNumUnit<ArchSpec::ComputeUnit>(),
                            spec.getNumUnit<ArchSpec::MemoryUnit>()},
        memoryMatrix{spec.getNumUnit<ArchSpec::ComputeUnit>(),
                     spec.getNumUnit<ArchSpec::MemoryUnit>()},
        dependency(am.getAnalysis<HandlerDependencyAnalysis>()) {
    prepareCommunicationCost(spec, costModel);
    prepareMemoryCost(spec, costModel);
  }
  // Data
  z3::context ctx;
  z3::solver solver;
  ArchSpec &spec;
  CostModel &costModel;

  // Analysis

  CostMatrix<TickT> communicationMatrix, memoryMatrix;
  HandlerDependencyAnalysis &dependency;

  std::map<FuncOp, std::vector<z3::expr>> mapping{};
  std::vector<z3::expr> totalTicksPerCU;

 
 // Funcs
  ThroughputT getTargetThroughput(FuncOp func) {
    return 1000;
  }

 // Import functions
  void prepareCommunicationCost(ArchSpec &spec, CostModel &costModel) {
    auto numComputeUnits = spec.getNumUnit<ArchSpec::ComputeUnit>();
    for (size_t i = 0; i < numComputeUnits; i++) {
      for (size_t j = 0; j < numComputeUnits; j++) {
        communicationMatrix.set(i, j, spec.communicationCost(i, j));
      }
    }
  }

  void prepareMemoryCost(ArchSpec &spec, CostModel &costModel) {
    auto numComputeUnits = spec.getNumUnit<ArchSpec::ComputeUnit>();
    auto numMemoryUnits = spec.getNumUnit<ArchSpec::MemoryUnit>();
    for (size_t i = 0; i < numComputeUnits; i++) {
      for (size_t j = 0; j < numMemoryUnits; j++) {
        memoryMatrix.set(i, j, spec.memoryCost(i, j));
      }
    }
  }

  void calculateTicksPerCU() {
    for (size_t i = 0; i < spec.getNumUnit<ArchSpec::ComputeUnit>(); i++) {
      z3::expr_vector ticks(ctx);
      for (auto &[funcOp, reps]: mapping) {
        for (auto &rep: reps) {
          auto cpuResource = costModel.computationCost(funcOp);
          auto res =
              z3::ite(rep == static_cast<int>(i), ctx.int_val(cpuResource), ctx.int_val(0));
          ticks.push_back(res);
        }
      }
      totalTicksPerCU.push_back(z3::sum(ticks));
    }
  }

  std::vector<z3::expr> getEvenPartition(int partitionNumber,
                                         std::vector<z3::expr> &prev,
                                         std::vector<z3::expr> &next) {

    int totalPrev = prev.size();
    int totalNext = next.size();

    int nextBegin = totalNext / totalPrev * partitionNumber;
    int nextEnd = totalNext / totalPrev * (partitionNumber + 1);

    return {next.begin() + nextBegin, next.begin() + nextEnd};
  }

  void calculateTotalTicks(FuncOp func, HandlerDependencyAnalysis &dependency) {
    auto successors = dependency.getSuccessors(func);

    z3::expr_vector ticks(ctx);
    for (size_t i = 0; i < mapping[func].size(); i++) {
      auto &cur = mapping[func][i];

      // same compute unit ticks
      z3::expr_vector cuTicks(ctx);
      for (auto &[funcOp, reps]: mapping) {
        auto cpuResource = costModel.computationCost(funcOp);
        for (auto &rep: reps) {
          auto res =
              z3::ite(cur == rep, ctx.int_val(cpuResource), ctx.int_val(0));
          cuTicks.push_back(res);
        }
      }

      // we use the replication model first
      z3::expr_vector commTicks(ctx);
      for (auto &succ : successors) {
        for (auto succ_rep : getEvenPartition(i, mapping[func], mapping[succ])) {
          auto commTick = communicationMatrix.oneOf(ctx, cur, succ_rep);
          commTicks.push_back(commTick);
        }
      }

      auto tick = z3::sum(cuTicks) + z3::sum(commTicks);
      ticks.push_back(CostModel::unitTick / tick);
    }

    solver.add(z3::sum(ticks) <= getTargetThroughput(func));
  }

  void calculateCost() {

    // create variable for each function
    for (auto &[name, func]: dependency.handlersMap) {
      if (!func.isExtern())
        mapping.try_emplace(func);
    }

    // calucate the max replication of each function. set to 2 * base cost
    for (auto &[_func, vec] : mapping) {
      // TODO(zhiyaung): different unit and get min?
      auto func = _func;

      auto ticks = costModel.computationCost(func);
      auto numReplications = CostModel::unitTick / ticks * 2;
      for (auto i = 0; i < numReplications; i++) {
        auto name = func.getName().str() + "_" + "rep" + "_" + std::to_string(i);
        auto var = ctx.int_const(name.c_str());
        vec.push_back(var);

        solver.add(
            var >= 0 &&
            var <= static_cast<int>(spec.getNumUnit<ArchSpec::ComputeUnit>()));
      }
    }

    // Limitation on the resoruce per compute unit and memory
    // TODO(zhiyuang): add memory limitation
    for (size_t i = 0; i < spec.getNumUnit<ArchSpec::ComputeUnit>(); i++) {
      z3::expr_vector resource(ctx);
      for (auto &[funcOp, reps]: mapping) {
        for (auto &rep: reps) {
          auto cpuResource = costModel.computeResource(funcOp);
          auto res =
              z3::ite(rep == static_cast<int>(i), ctx.int_val(cpuResource), ctx.int_val(0));
          resource.push_back(res);
        }
      }
      solver.add(z3::sum(resource) <= ctx.int_val(spec.get<ArchSpec::ComputeUnit>(i).cpu));
    }

    // Limitation on performance (ticks)
    for (auto &[func, _]: mapping) {
      calculateTotalTicks(func, dependency);
    }
  }

  // TODO: another function for annotation onto IR
  void extractSolution() {
    std::cout << "solving model: \n" << solver << "\n" << "solving...\n" << solver.check() << "\n";
    z3::model m = solver.get_model();
    for (auto &[_func, reps]: mapping) {
      auto func = _func;
      std::cout << func.getName().str() << "\n";
      for (auto &rep: reps) {
        int v = m.eval(rep).get_numeral_int();
        std::cout << v << "\n";
      }
    }
  }


};
} // local namespace

void ArchMappingPass::runOnOperation() {
  auto &dependency = getAnalysis<HandlerDependencyAnalysis>();
  dependency.dump();

  ModuleOp moduleOp = getOperation();
  moduleOp->walk([&](ReturnOp returnOp){
    auto func = dependency.lookupController(returnOp);
    llvm::errs() << "for return op:";
    returnOp->dump();

    if (func != nullptr)
      llvm::errs() << "Found Controller: " << func.getName() << "\n";
    else
      llvm::errs() << "Not found Controller\n";
  });

  return signalPassFailure();

  // prepare the models
  AnalysisManager am = getAnalysisManager();
  ArchSpec spec(archSpecFile.getValue());

  auto name = costModelName.getValue();
  std::unique_ptr<CostModel> costModel;
  if (name == "simple")
    costModel = std::make_unique<SimpleCostModel>(spec);

  if (!costModel) {
    getOperation()->emitError("Cost model not found");
    return signalPassFailure();
  }

  MappingSolver solver(am, spec, *costModel);
  solver.calculateCost();
  solver.extractSolution();
}

} // namespace ep2
} // namespace mlir
