//===- Passes.h - Toy Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Toy.
//
//===----------------------------------------------------------------------===//

#ifndef EP2_PASSES_H
#define EP2_PASSES_H

#include <memory>
#include <vector>
#include <utility>
#include <optional>
#include <unordered_set>
#include <unordered_map>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/EquivalenceClasses.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "ep2/dialect/Dialect.h"

namespace mlir {
namespace ep2 {

///////////////////
// Analysis
///////////////////
struct LowerStructAnalysis {
    LowerStructAnalysis(mlir::Operation *op);

    // Return Wrapper Struct Types (not pointers)
    ArrayRef<LLVM::LLVMStructType> getWrapperTypes(ep2::FuncOp funcOp);
    const std::string prefix = "__wrapper";
    std::string getEventStructName(llvm::StringRef name) {
        return prefix + "_event_" + name.str();
    }
  private:
    llvm::StringMap<LLVM::LLVMStructType> handlerTypes;
    std::map<std::pair<std::string, std::string>, llvm::SmallVector<LLVM::LLVMStructType>> ioTypes;
};

// Handler dependency analysis pass
struct HandlerDependencyAnalysis {

  // Struct for fullname
  struct HandlerFullName {
    llvm::StringRef event;
    llvm::StringRef atom = "";

    friend bool operator<(const HandlerFullName &l, const HandlerFullName &r) {
      return std::tie(l.event, l.atom) < std::tie(r.event, r.atom);
    }
    std::string mangle() const { 
      auto surffix = atom == "" ? "" : "_" + atom;
      return ("__handler_" + event + surffix).str();
    }

    HandlerFullName(std::string event, std::string atom = "") : event(event), atom(atom) {}
    HandlerFullName(FuncOp funcOp);
    HandlerFullName(ReturnOp returnOp);
    HandlerFullName(InitOp initOp);
  };

  using KeyTy = FuncOp;
  using EdgeTy = FuncOp;
  using GraphType = std::map<KeyTy, std::vector<EdgeTy>>;
  using OrderType = std::vector<FuncOp>;

  GraphType graph;
  std::vector<GraphType> subGraphs;
  std::vector<std::vector<FuncOp>> subGraphsOrder;
  std::unordered_map<std::string, std::unordered_set<std::string>> eventDeps;

  std::map<HandlerFullName, FuncOp> handlersMap;
  FuncOp lookupHandler(HandlerFullName fullname);

  std::map<HandlerFullName, FuncOp> controllersMap;
  FuncOp lookupController(HandlerFullName fullname);

  std::vector<FuncOp> getSuccessors(FuncOp funcOp, bool includeExtern = true) {
    return graph[funcOp];
  }
  std::vector<FuncOp> getPredecessors(FuncOp funcOp);

  bool hasSuccessor(llvm::StringRef eventName) {
    auto it = std::find_if(handlersMap.begin(), handlersMap.end(), [&](auto &pr) {
      return pr.first.event == eventName;
    });
    return it != handlersMap.end() && !it->second.isExtern();
  }
  bool hasPredecessor(FuncOp funcOp) {
    HandlerFullName name(funcOp);
    return llvm::count_if(graph, [&](auto &pr){
      // cast away const
      Operation *op = pr.first;
      return !cast<FuncOp>(op).isExtern() &&
             llvm::count_if(pr.second, [&](FuncOp op) { return op == funcOp; });
    });
  }

  HandlerDependencyAnalysis(Operation *op);

  size_t numComponents() { return subGraphs.size(); }

  template <typename F>
  void forEachComponent(F f) {
    for (size_t i = 0; i < subGraphs.size(); ++i)
      f(i, subGraphs[i], subGraphsOrder[i]);
  }
  void dump();

private:
  void getConnectedComponents();
};

struct BufferAnalysis {
  // Out lattic structure ...
  struct BufferHistory {
    Value source;
    bool known;
    int offset;
    BufferHistory(Value source, bool known, int offset)
        : source(source), known(known), offset(offset) {}
    BufferHistory() : BufferHistory(nullptr, false, 0) {}
    void merge(BufferHistory &rhs) {
      known = std::tie(source, offset, source) ==
              std::tie(rhs.source, rhs.offset, rhs.source);
      known = known && rhs.known;
    }
  };

  using ArgMap = std::map<int, BufferHistory>;

  std::map<Block *, std::vector<ArgMap>> blockInput{};
  llvm::EquivalenceClasses<mlir::detail::ValueImpl *> bufferClasses{};
  std::map<Operation *, int> offsetAt{};

  BufferAnalysis(Operation *op, AnalysisManager &am);
};

/// Analysis for context
struct ContextBufferizationAnalysis {
  using TableT = llvm::StringMap<std::pair<int, mlir::Type>>;

  std::vector<TableT> contextTables;
  std::map<std::string, TableT&> contextMap;
  AnalysisManager& am;

  ContextBufferizationAnalysis(Operation* op, AnalysisManager& am);
  std::pair<int, mlir::Type> getContextType(FunctionOpInterface funcOp, StringRef name);
  TableT &getContextTable(std::string mangledName);
  TableT &getContextTable(FunctionOpInterface funcOp);
  TableT &getContextTable(InitOp initOp);
  StructType getContextAsStruct(FunctionOpInterface op);

  void invalidate() {
    AnalysisManager::PreservedAnalyses preserved;
    preserved.preserve<HandlerDependencyAnalysis>();
    am.invalidate(preserved);
  }
  void dump() {
    for (auto &[opName, table] : contextMap) {
      llvm::errs() << "Context table for " << opName << "\n";
      for (auto &[name, pr] : table) {
        llvm::errs() << "  " << name << " : ";
        pr.second.dump();
      }
    }
    llvm::errs() << "\n";
  }
};

///////////////////
// Passes
///////////////////

struct ContextToArgumentPass :
        public PassWrapper<ContextToArgumentPass, OperationPass<ModuleOp>> {
  ContextToArgumentPass() = default;
  ContextToArgumentPass(const ContextToArgumentPass &pass) {}
  void runOnOperation() final;
  void runOnFunction(FuncOp funcOp, ContextBufferizationAnalysis &analysis,
                     HandlerDependencyAnalysis &dependency);
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<EP2Dialect, scf::SCFDialect, cf::ControlFlowDialect>();
  }
  StringRef getArgument() const final { return "ep2-context-to-argument"; }
  StringRef getDescription() const final {
    return "Dump all ep2 context to value";
  }
  Option<bool> keepLastContext{
      *this, "keep-last",
      llvm::cl::desc("Keep the last context in function call"),
      llvm::cl::init(false)};
};

struct BufferToValuePass :
        public PassWrapper<BufferToValuePass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, scf::SCFDialect, cf::ControlFlowDialect>();
    }
    StringRef getArgument() const final { return "ep2-buffer-to-value"; }
    StringRef getDescription() const final { return "Convert ep2 buffers to a value type"; }
};

struct CFToPredPass :
        public PassWrapper<CFToPredPass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, scf::SCFDialect, cf::ControlFlowDialect, arith::ArithDialect>();
    }
    StringRef getArgument() const final { return "cf-to-pred"; }
    StringRef getDescription() const final { return "General pass for adding a pred value for every block in the function"; }
};

struct EP2LinearizePass :
        public PassWrapper<EP2LinearizePass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, scf::SCFDialect, cf::ControlFlowDialect, arith::ArithDialect>();
    }
    StringRef getArgument() const final { return "ep2-linearize"; }
    StringRef getDescription() const final { return "Linearize all branches. Do not work with ExtractOp"; }
};

/// Mapping. Map from unit values to values
struct ArchMappingPass :
        public PassWrapper<ArchMappingPass, OperationPass<ModuleOp>> {
  // TODO(zhiyuang): copy construction?
  ArchMappingPass() = default;
  ArchMappingPass(const ArchMappingPass &pass) {}
  void runOnOperation() final;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<EP2Dialect, scf::SCFDialect, cf::ControlFlowDialect>();
  }
  StringRef getArgument() const final { return "ep2-mapping"; }
  StringRef getDescription() const final { return "Mapping ep2 program to mlir structures"; }

  Option<std::string> archSpecFile{
      *this, "arch-spec-file", llvm::cl::desc("Filename for arch spec"), llvm::cl::Required};
  Option<std::string> costModelName{
      *this, "cost-model", llvm::cl::desc("Name for cost model"), llvm::cl::init("simple")};
  Option<int> targetThroughput{
      *this, "target-tput", llvm::cl::desc("Target thoughtput. in unit of unitTick / tick (dimensionless)")};
};

// Lower to Emitc pass
struct LowerEmitcPass :
        public PassWrapper<LowerEmitcPass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect, emitc::EmitCDialect>();
    }
    StringRef getArgument() const final { return "ep2-lower-emitc"; }
    StringRef getDescription() const final { return "Rewrite to generate emitc"; }
};

struct ContextTypeInferencePass : PassWrapper<ContextTypeInferencePass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect>();
    }

    StringRef getArgument() const final { return "ep2-context-infer"; }
    StringRef getDescription() const final { return "Infer context types across different handlers"; }
};

enum class MemType {
  LMEM,
  CLS,
  CTM,
  IMEM,
  EMEM,
};

struct TableInfo {
  std::string tableType;
  std::string keyType;
  std::string valType;
  int size;
};

struct CollectInfoAnalysis {
  struct QueueInfo {
    MemType memType;
    int size;
    std::vector<int> replicas;

    QueueInfo() {}
    QueueInfo(MemType mt, int s, std::vector<int> r) : memType(mt), size(s), replicas(r) {}
  };

  std::unordered_set<unsigned> typeBitWidths;
  std::vector<std::pair<std::string, mlir::LLVM::LLVMStructType>> structDefs;
  std::unordered_map<std::string, QueueInfo> eventQueues;
  std::unordered_map<std::string, std::unordered_set<std::string>> eventDeps;
  std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> eventAllocs;
  std::unordered_map<std::string, std::pair<TableInfo, std::vector<std::string>>> tableInfos;

  CollectInfoAnalysis(Operation* op, AnalysisManager& am);

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return false;
  }
};

// Collect defs for header.
struct CollectHeaderPass :
        public PassWrapper<CollectHeaderPass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect, emitc::EmitCDialect>();
    }
    StringRef getArgument() const final { return "ep2-collect-header"; }
    StringRef getDescription() const final { return "Collect header file"; }
};

// Lower intrinsics in emitc.
struct LowerMemcpyPass :
        public PassWrapper<LowerMemcpyPass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect, emitc::EmitCDialect>();
    }
    StringRef getArgument() const final { return "ep2-lower-memcpy"; }
    StringRef getDescription() const final { return "Lower memcpy file"; }
};

struct StructUpdatePropagationPass :
        public PassWrapper<StructUpdatePropagationPass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect, emitc::EmitCDialect>();
    }
    StringRef getArgument() const final { return "ep2-update-ppg"; }
    StringRef getDescription() const final { return "Struct update propagation file"; }
};

struct HandlerReplicationPass :
        public PassWrapper<HandlerReplicationPass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect, emitc::EmitCDialect>();
    }
    StringRef getArgument() const final { return "ep2-handler-repl"; }
    StringRef getDescription() const final { return "Handler replication pass file"; }
};

struct LowerNoctxswapPass :
        public PassWrapper<LowerNoctxswapPass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect, emitc::EmitCDialect>();
    }
    StringRef getArgument() const final { return "ep2-lower-noctxswap"; }
    StringRef getDescription() const final { return "Lower no-context-switch region pass file"; }
};

struct EmitNetronomePass :
        public PassWrapper<EmitNetronomePass, OperationPass<ModuleOp>> {
  // TODO(zhiyuang): copy construction?
  EmitNetronomePass() = default;
  EmitNetronomePass(const EmitNetronomePass &pass) {}
  void runOnOperation() final;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect,
                    emitc::EmitCDialect>();
  }
  StringRef getArgument() const final { return "ep2-emit-netronome"; }
  StringRef getDescription() const final { return "Emit netronome"; }

  Option<std::string> basePathOpt{
      *this, "basePath", llvm::cl::desc("Base path for generated files")};
};

// Handler dependency analysis pass
struct HandlerInOutAnalysis {
  // From funcop -> blockarg
  mlir::DenseMap<mlir::ep2::FuncOp, mlir::Region::BlockArgListType> handler_in_arg_list;

  // From funcop -> returnedValues (Each value is a struct (event) sending to a dest handler)
  mlir::DenseMap<mlir::ep2::FuncOp, mlir::SmallVector<mlir::Value>> handler_returnop_list;

  HandlerInOutAnalysis(Operation* op);
};

// Handler dependency analysis pass
struct TableAnalysis {
  // Table Lookup Information
  mlir::DenseMap<mlir::Value, mlir::SmallVector<ep2::UpdateOp>> table_update_uses;

  // Table Update Information
  mlir::DenseMap<mlir::Value, mlir::SmallVector<ep2::LookupOp>> table_lookup_uses;

  // lookupop/updateop -> # of index in lookup/update
  mlir::DenseMap<mlir::Operation *, int> access_index;

  TableAnalysis(Operation* op);

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return false;
  }
};


struct AtomAnalysis {
  llvm::StringMap<std::pair<std::string, size_t>> atomToNum;

  AtomAnalysis(Operation* op, AnalysisManager& am);

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return false;
  }
};

struct LocalAllocAnalysis {
  std::unordered_map<mlir::Operation*, std::string> localAllocs;

  LocalAllocAnalysis(Operation* op, AnalysisManager& am);

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return false;
  }
};

// LLVM Backends

struct LowerLLVMPass :
        public PassWrapper<LowerLLVMPass, OperationPass<ModuleOp>> {
  LowerLLVMPass() = default;
  LowerLLVMPass(const LowerLLVMPass &pass) {}
  void runOnOperation() final;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect,
                    scf::SCFDialect, cf::ControlFlowDialect>();
  }
  StringRef getArgument() const final { return "ep2-lower-llvm"; }
  StringRef getDescription() const final { return "Rewrite to LLVM dialect"; }

  using TableT = std::map<std::string, LLVM::LLVMFuncOp>;
  TableT apiFunctions{};
  Option<std::string> generateMode{
      *this, "generate",
      llvm::cl::desc("Modes for handling event generating: event, call, raw"),
      llvm::cl::init("call")};

private:
  void populateAPIFunctions(mlir::TypeConverter &converter);
};

struct EmitLLVMHeaderPass
    : public PassWrapper<EmitLLVMHeaderPass, OperationPass<ModuleOp>> {
  void runOnOperation() final;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect,
                    scf::SCFDialect, cf::ControlFlowDialect>();
  }
  StringRef getArgument() const final { return "ep2-emit-llvm-header"; }
  StringRef getDescription() const final { return "Emit LLVM header"; }
};

struct ContextToMemPass :
        public PassWrapper<ContextToMemPass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect, emitc::EmitCDialect>();
    }
    StringRef getArgument() const final { return "ep2-context-to-mem"; }
    StringRef getDescription() const final { return "Restore context to memory"; }
};

struct BufferReusePass :
        public PassWrapper<BufferReusePass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect, emitc::EmitCDialect>();
    }
    StringRef getArgument() const final { return "ep2-buffer-reuse"; }
    StringRef getDescription() const final { return "Reuse buffer using zero copy operators"; }
};

struct DeadFieldEliminatePass :
        public PassWrapper<DeadFieldEliminatePass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, cf::ControlFlowDialect>();
    }
    StringRef getArgument() const final { return "ep2-dfe"; }
    StringRef getDescription() const final { return "Remove the dead field in structs"; }
};

struct DeadParameterEliminatePass :
        public PassWrapper<DeadParameterEliminatePass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, cf::ControlFlowDialect>();
    }
    StringRef getArgument() const final { return "ep2-dpe"; }
    StringRef getDescription() const final { return "Remove the dead parameter in generate calls between handlers"; }
};

struct CanonicalizePass :
        public PassWrapper<CanonicalizePass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, cf::ControlFlowDialect>();
    }
    StringRef getArgument() const final { return "ep2-canon"; }
    StringRef getDescription() const final { return "conversion level connonlicalize pass for ep2"; }
};

struct RepackStructTypesPass :
        public PassWrapper<RepackStructTypesPass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, cf::ControlFlowDialect>();
    }
    StringRef getArgument() const final { return "ep2-repack"; }
    StringRef getDescription() const final { return "Repack struct types"; }
};

} // namespace ep2
} // namespace mlir

#endif // EP2_PASSES_H
