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
#include <unordered_map>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
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

///////////////////
// Passes
///////////////////
// Nop Elimination Pass
struct NopEliminationPass : public PassWrapper<NopEliminationPass, OperationPass<>> {
  void runOnOperation() final;
  void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<EP2Dialect>();
  }
  StringRef getArgument() const final { return "ep2-nop-elim"; }
  StringRef getDescription() const final { return "Eliminate EP2 Nop"; }
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

struct CollectInfoAnalysis {
  std::vector<std::pair<std::string, mlir::LLVM::LLVMStructType>> structDefs;
  std::unordered_map<std::string, std::pair<MemType, int>> eventQueues;
  std::unordered_map<std::string, std::string> eventDeps;
  std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> eventAllocs;

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
struct LowerIntrinsicsPass :
        public PassWrapper<LowerIntrinsicsPass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect, emitc::EmitCDialect>();
    }
    StringRef getArgument() const final { return "ep2-lower-intrinsics"; }
    StringRef getDescription() const final { return "Lower intrinsics file"; }
};

struct EmitFilesPass :
        public PassWrapper<EmitFilesPass, OperationPass<ModuleOp>> {
  // TODO(zhiyuang): copy construction?
  EmitFilesPass() = default;
  EmitFilesPass(const EmitFilesPass &pass) {}
  void runOnOperation() final;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect,
                    emitc::EmitCDialect>();
  }
  StringRef getArgument() const final { return "ep2-emit-files"; }
  StringRef getDescription() const final { return "Emit files"; }

  Option<std::string> basePathOpt{
      *this, "basePath", llvm::cl::desc("Base path for generated files")};
};

// Handler dependency analysis pass
struct HandlerDependencyAnalysis {
  // enum EdgeType { MUST, MAY };
  // using GraphType = std::unordered_map<Operation*, std::vector<std::pair<EdgeType, Operation*>>>;
  using KeyTy = FuncOp;
  using EdgeTy = FuncOp;
  using GraphType = std::map<KeyTy, std::vector<EdgeTy>>;
  using OrderType = std::vector<FuncOp>;

  GraphType graph;
  std::vector<GraphType> subGraphs;
  std::vector<std::vector<FuncOp>> subGraphsOrder;
  std::unordered_map<std::string, std::string> eventDeps;

  HandlerDependencyAnalysis(Operation* op);

  size_t numComponents() { return subGraphs.size(); }

  template<typename F>
  void forEachComponent(F f) {
    for (size_t i = 0; i < subGraphs.size(); ++i)
      f(i, subGraphs[i], subGraphsOrder[i]);
  }

  void dump() {
    llvm::errs() << "Found " << subGraphs.size() << " connected components\n";
    for (size_t i = 0; i < subGraphs.size(); ++i) {
      llvm::errs() << "Component " << i << " " << subGraphs[i].size() << " "
                   << subGraphsOrder.size() << "\n";
    }

    for (auto &[handler, edges] : graph) {
      Operation *op = handler;
      auto funcOp = dyn_cast<FuncOp>(op);
      llvm::errs() << "Handler " << funcOp.getSymName().str() << " has "
                   << edges.size() << " edges\n";
      for (auto &target : edges) {
        Operation *op = target;
        auto funcOp = dyn_cast<FuncOp>(op);
        llvm::errs() << "  " << funcOp.getSymName().str() << "\n";
      }
    }
  }

 private:
  void getConnectedComponents();
};

struct ContextRefTypeAssignPass : public PassWrapper<ContextRefTypeAssignPass, OperationPass<>> {
  std::unordered_map<std::string, mlir::Type> context_ref_name_to_type;
  std::unordered_map<std::string, mlir::Type> unassigned_list;
  void runOnOperation() final;
  void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<EP2Dialect>();
  }
  StringRef getArgument() const final { return "ep2-context_ref_type_assign";}
  StringRef getDescription() const final { return "Assign Type to EP2 Context Ref"; }
  
};

/// Analysis for context
struct ContextBufferizationAnalysis {
  using TableT = llvm::StringMap<std::pair<int, mlir::Type>>;

  std::vector<TableT> contextTables;
  std::map<std::string, TableT&> contextMap;
  AnalysisManager& am;

  ContextBufferizationAnalysis(Operation* op, AnalysisManager& am);
  std::pair<int, mlir::Type> getContextType(FunctionOpInterface funcOp, StringRef name);

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

struct AtomAnalysis {
  llvm::StringMap<size_t> atomToNum;

  AtomAnalysis(Operation* op, AnalysisManager& am);
};

struct LocalAllocAnalysis {
  std::unordered_map<mlir::Operation*, std::string> localAllocs;

  LocalAllocAnalysis(Operation* op, AnalysisManager& am);
};

struct EmitFPGAPass : public PassWrapper<EmitFPGAPass, OperationPass<>> {
  void runOnOperation() final;
  void getDependentDialects(DialectRegistry &registry) const override {
      registry.insert<EP2Dialect>();
  }

  StringRef getArgument() const final { return "ep2-emit-FPGA"; }
  StringRef getDescription() const final { return "Emit FPGA Code"; }
 private:
  mlir::Operation *module;
  OpBuilder *builder;
  // std::unordered_map<mlir::Location, std::string> arg_names;
  mlir::DenseMap<Value, std::string> arg_names;
  std::string getValName(mlir::Value val);
  void UpdateValName(mlir::Value val, std::string name);
  void emitVariableInit(std::ofstream &file, ep2::InitOp initop);
  void emitExtract(std::ofstream &file, ep2::ExtractOp extractop);
  void emitStructAccess(std::ofstream &file, ep2::StructAccessOp structaccessop);
  void emitStructUpdate(std::ofstream &file, ep2::StructUpdateOp structupdateop);
};

} // namespace ep2
} // namespace mlir

#endif // EP2_PASSES_H
