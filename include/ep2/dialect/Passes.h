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
  enum EdgeType { MUST, MAY };
  std::unordered_map<Operation*, std::vector<std::pair<EdgeType, Operation*>>> graph;

  HandlerDependencyAnalysis(Operation* op);
};

struct ContextAnalysis {
  std::unordered_map<mlir::Operation*, mlir::Operation*> disj_groups;
  std::unordered_map<mlir::Operation*, llvm::StringMap<std::pair<int, mlir::Type>>> disj_contexts;

  ContextAnalysis(Operation* op, AnalysisManager& am);
};

} // namespace ep2
} // namespace mlir

#endif // EP2_PASSES_H
