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

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"

#include "ep2/dialect/Dialect.h"

namespace mlir {
namespace ep2 {

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

} // namespace ep2
} // namespace mlir

#endif // EP2_PASSES_H
