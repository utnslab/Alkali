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

// inline void registerAllocationAnnotationPass() {
inline void registerNopEliminationPass() {
  PassRegistration<NopEliminationPass>();
}

// Function Rewrite Pass
struct FunctionRewritePass :
        public PassWrapper<FunctionRewritePass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, func::FuncDialect, LLVM::LLVMDialect, emitc::EmitCDialect>();
    }
    StringRef getArgument() const final { return "ep2-function-rewrite"; }
    StringRef getDescription() const final { return "Rewrite EP2 Function to generate to functions"; }
};

struct ContextTypeInferencePass : PassWrapper<ContextTypeInferencePass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect>();
    }

    StringRef getArgument() const final { return "ep2-context-infer"; }
    StringRef getDescription() const final { return "Infer context types across different handlers"; }
};

// Lower to LLVM Pass
struct LowerToLLVMPass : public PassWrapper<LowerToLLVMPass, OperationPass<ModuleOp>> {
    void runOnOperation() final;
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<EP2Dialect, LLVM::LLVMDialect>();
    }
    StringRef getArgument() const final { return "ep2-lower-to-llvm"; }
    StringRef getDescription() const final { return "Lower EP2 to LLVM"; }
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
  using TableT = llvm::StringMap<mlir::Type>;

  std::vector<TableT> contextTables;
  std::map<mlir::Operation*, TableT&> contextMap;
  AnalysisManager& am;

  ContextBufferizationAnalysis(Operation* op, AnalysisManager& am);
  mlir::Type getContextType(FuncOp funcOp, StringRef name);
  void setContextType(Operation *op, StringRef name, mlir::Type type);

  void invalidate() {
    AnalysisManager::PreservedAnalyses preserved;
    preserved.preserve<HandlerDependencyAnalysis>();
    am.invalidate(preserved);
  }
  void dump() {
    for (auto &[op, table] : contextMap) {
      auto funcOp = dyn_cast<FuncOp>(op);
      llvm::errs() << "Context table for " << funcOp.getSymName() << "\n";
      for (auto &[name, type] : table) {
        llvm::errs() << "  " << name << " : ";
        type.dump();
      }
    }
    llvm::errs() << "\n";
  }
};


struct ContextAnalysis {
  struct ContextField {
    size_t pos;
    size_t offs;
    mlir::Type ty;

    ContextField() {}
    ContextField(size_t p, size_t o, mlir::Type t) : pos(p), offs(o), ty(t) {}
  };

  std::unordered_map<mlir::Operation*, mlir::Operation*> disj_groups;
  std::unordered_map<mlir::Operation*, llvm::StringMap<ContextField>> disj_contexts;

  ContextAnalysis(Operation* op, AnalysisManager& am);
};

struct AtomAnalysis {
  llvm::StringMap<size_t> atomToNum;

  AtomAnalysis(Operation* op, AnalysisManager& am);
};

} // namespace ep2
} // namespace mlir

#endif // EP2_PASSES_H
