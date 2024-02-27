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

using ComputeUnit = ArchSpec::ComputeUnit;
using MemoryUnit = ArchSpec::MemoryUnit;

class MappingSolver {
  // Util function
  template<typename U1, typename U2>
  z3::expr oneOfMatrix(z3::context &ctx, z3::expr index1, z3::expr index2, std::vector<std::vector<int>> &data) {
    z3::expr_vector matrix(ctx);
    for (size_t i = 0; i < spec.getNumUnit<U1>(); i++)
      for (size_t j = 0; j < spec.getNumUnit<U2>(); j++)
        matrix.push_back(z3::ite(index1 == i && index2 == j, ctx.int_val(data[i][j]), ctx.int_val(0)));
    return z3::sum(matrix);
  }

  z3::expr oneOf(z3::expr index1, z3::expr index2) {
    auto commOverhead = 0;
    if (spec.getNumUnit<ComputeUnit>() > 1)
      commOverhead = spec.communicationMatrix[0][1];
    return z3::ite(index1 == index2, ctx.int_val(0), ctx.int_val(commOverhead));
  }

 public:
  MappingSolver(AnalysisManager &am, ArchSpec &spec, CostModel &costModel)
      : ctx{}, solver{ctx}, spec(spec), costModel(costModel),
        dependency(am.getAnalysis<HandlerDependencyAnalysis>()) {}
  // Data
  z3::context ctx;
  z3::solver solver;
  ArchSpec &spec;
  CostModel &costModel;

  // Analysis
  HandlerDependencyAnalysis &dependency;

  // intermidiate results
  std::map<FuncOp, std::vector<z3::expr>> mapping{};
  std::map<FuncOp, std::vector<z3::expr>> repCost{};
  std::vector<z3::expr> ticksPerCU;
 
 // Funcs
  ThroughputT getTargetThroughput(FuncOp func) {
    // TODO(zhiyuang): based on requriements, add more constraints
    return CostModel::unitTick;
  }

  void calculateTicksPerCU() {
    // as i == 0 means we do not map, we start from 1
    // for each cu we calucate all replications
    for (size_t i = 1; i <= spec.getNumUnit<ArchSpec::ComputeUnit>(); i++) {
      z3::expr_vector ticks(ctx);
      for (auto &[funcOp, reps]: mapping) {
        auto &costs = repCost[funcOp];
        for (auto [cu, cost]: llvm::zip_equal(reps, costs)) {
          auto res =
              z3::ite(cu == static_cast<int>(i), cost, ctx.int_val(0));
          ticks.push_back(res);
        }
      }
      auto cuVar = ctx.int_const(("cu" + std::to_string(i)).c_str());
      solver.add(cuVar == z3::sum(ticks));
      ticksPerCU.push_back(cuVar);
    }
  }

  void calculateCommunicationOverhead() {
    int c = 0;
    for (auto &[func, reps] : mapping) {
      auto successors = dependency.getSuccessors(func);
      auto [it, _] = repCost.try_emplace(func);

      // currently we only support a single mode: one to one mapping.
      // TODO(zhiyuang): add support for partition mode
      for (auto &rep : reps) {
        z3::expr_vector commTicks(ctx);
        for (auto sucFunc : successors) {
          auto it = mapping.find(sucFunc);
          if (it == mapping.end())
            continue;
          for (auto &suc : it->second) {
            // if cur or suc is not mapped, we skip
            // auto commTick = oneOfMatrix<ComputeUnit, ComputeUnit>(
            //     ctx, rep, suc, spec.communicationMatrix);
            auto commTick = oneOf(rep, suc);
            commTicks.push_back(commTick);
          }
        }
        auto tick = commTicks.size() == 0 ? ctx.int_val(0) : z3::sum(commTicks);

        auto costVar = ctx.int_const(("cost" + std::to_string(c++)).c_str());
        solver.add(costVar == tick + costModel.computationCost(func));
        it->second.push_back(costVar);
      }
    }
  }

  std::map<FuncOp, z3::expr> funcTputs{};

  void calculateThroughput() {
    for (auto &[func, reps]: mapping) {
      z3::expr_vector repTputs(ctx);
      for (auto &rep : reps) {
        for (int i = 1; i <= spec.getNumUnit<ArchSpec::ComputeUnit>(); i++) {
          auto repTput = z3::ite(rep == i, CostModel::unitTick / ticksPerCU[i-1] + ctx.int_val(1), ctx.int_val(0));
          repTputs.push_back(repTput);
        }
      }
      auto funcTput = z3::sum(repTputs);
      funcTputs.try_emplace(func, funcTput);
      solver.add(funcTput >= getTargetThroughput(func));
    }
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
      // each unit get at least one replication
      auto maxReplications =
          std::min((size_t)ticks * 3,
                   spec.getNumUnit<ComputeUnit>() - mapping.size() + 1);
      llvm::errs() << "function" << func.getSymName() << " numReplications: " << maxReplications << "\n";
      for (auto i = 0; i < maxReplications; i++) {
        auto name = func.getName().str() + "_" + "rep" + "_" + std::to_string(i);
        auto var = ctx.int_const(name.c_str());
        vec.push_back(var);

        solver.add(
            var >= 0 &&
            var <= static_cast<int>(spec.getNumUnit<ArchSpec::ComputeUnit>()));
      }
      

      for (auto i = 0; i < maxReplications; i++)
        for (auto j = 0; j < i; j++)
          solver.add(vec[i] == 0 || vec[j] == 0 || vec[i] != vec[j]);
    }

    // add constraints on resources
    llvm::errs() << "adding constriaints on resources\n";
    // Limitation on the resoruce per compute unit and memory

    // TODO(zhiyuang): add resource limitation

    // Limitation on performance (ticks)
    llvm::errs() << "calucating communication overhead\n";
    calculateCommunicationOverhead();

    llvm::errs() << "calculating ticks per CU\n";
    calculateTicksPerCU();

    llvm::errs() << "calculating throughput\n";
    calculateThroughput();

    llvm::errs() << "All constraints added\n";
  }

  // TODO: another function for annotation onto IR
  auto extractSolution() {
    llvm::DenseMap<FuncOp,llvm::SmallVector<std::string>> map;
    bool all_eq = true;
    for (auto &[func, reps]: mapping) {
      if (reps.size() != mapping.begin()->second.size()) {
        all_eq = false;
        break;
      }
    }

    // TODO(zhiyuang): add an assumption of all mapped to same
    z3::check_result res;
    if (all_eq) {
      auto size = mapping.begin()->second.size();
      llvm::errs() << "all functions have the same number of replications " << size << ", opt with sym\n";
      z3::expr_vector assumptions(ctx);
      for (auto &[func, reps] : mapping) {
        for (int i = 0; i < size; i++) {
          auto &rep = reps[i];
          assumptions.push_back(rep == ctx.int_val(i + 1));
        }
      }
      res = solver.check(assumptions);
    } else
      res = solver.check();

    if (res == z3::sat) {
      auto m = solver.get_model();
      for (auto &[_func, reps]: mapping) {
        auto [it, _] = map.try_emplace(_func);
        for (auto &rep: reps) {
          int cuIndex = m.eval(rep).get_numeral_int();
          if (cuIndex != 0)
            it->second.push_back(spec.get<ComputeUnit>(cuIndex).id);
        }
      }
    }
    return map;
  }

  void dumpSolution() {
    std::cout << "solving model: \n" << solver << "\n" << "solving...\n" << solver.check() << "\n";
    z3::model m = solver.get_model();
    for (auto &[_func, reps]: mapping) {
      auto func = _func;
      std::cout << func.getName().str() << "\n";
      for (auto &rep: reps) {
        int v = m.eval(rep).get_numeral_int();
        std::cout << v << "\n";
      }

      auto &tputExpr = funcTputs.at(func);
      std::cout << "throughput: " << m.eval(tputExpr).get_numeral_int() << "\n";
    }

    std::cout << "solving cu costs\n";
    for (int i = 0; i < spec.getNumUnit<ComputeUnit>(); i++) {
      int v = m.eval(ticksPerCU[i]).get_numeral_int();
      std::cout << "cu" << i << ": " << v << "\n";
    }
  }
};

} // local namespace

void ArchMappingPass::runOnOperation() {
  // TODO:based on mapping, try to build different models. related to partitioning
  if (!archSpecFile.hasValue()) {
    getOperation()->emitError("arch spec file not found");
    return signalPassFailure();
  }

  // build cost model
  ArchSpec spec(archSpecFile.getValue());

  auto name = costModelName.getValue();
  std::unique_ptr<CostModel> costModel;
  if (name == "simple")
    costModel = std::make_unique<SimpleCostModel>(spec);
  else if (name == "fpga")
    costModel = std::make_unique<FPGACostModel>(spec);

  if (!costModel) {
    getOperation()->emitError("Cost model not found");
    return signalPassFailure();
  }

  // build mapping solver
  AnalysisManager am = getAnalysisManager();
  MappingSolver solver(am, spec, *costModel);
  solver.calculateCost();

  llvm::errs() << "solving...\n";
  solver.dumpSolution();
  auto map = solver.extractSolution();
  if (map.empty()) {
    getOperation()->emitError("No mapping solution found");
    return signalPassFailure();
  }

  // update the mapping to IR
  OpBuilder builder(getOperation());
  getOperation()->walk([&](FuncOp func) {
    auto it = map.find(func);
    if (it != map.end()) {
      auto &cus = it->second;

      auto refs = llvm::map_to_vector(cus, [&](auto &cu) {
        return llvm::StringRef(cu);
      });
      builder.getStrArrayAttr(refs);
      func->setAttr("instances", builder.getStrArrayAttr(refs));
    }
  });

  // TODO(zhiyuang): add based on search partition policy: partition or not
  // partition global state based on the mapping
  auto moduleOp = getOperation();
  getOperation()->walk([&](GlobalOp globalOp) {
    if (!globalOp->hasAttr("scope"))
      return;
    auto handlers = globalOp->getAttrOfType<ep2::ScopeAttr>("scope").getHandlers();
    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      HandlerDependencyAnalysis::HandlerFullName name(funcOp);
      for (auto h : handlers) {
        if (name.join() != h.getValue())
          continue;
        if (auto attr = funcOp->getAttr("instances")) {
          globalOp->setAttr("instances", attr);
          std::string attrName = "instances_" + globalOp.getName().str();
          funcOp->setAttr(attrName, attr);
        }
      }
    }
  });

}

} // namespace ep2
} // namespace mlir
