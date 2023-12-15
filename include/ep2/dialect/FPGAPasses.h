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

#ifndef EP2_FPGA_PASSES_H
#define EP2_FPGA_PASSES_H

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "ep2/dialect/Dialect.h"

namespace mlir {
namespace ep2 {

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
  ContextAnalysis *contextAnalysis;
  HandlerInOutAnalysis *handlerInOutAnalysis;
  FuncOp *cur_funcop;

  enum VAL_TYPE { CONTEXT, STRUCT, INT, BUF, ATOM, UNKNOWN };

  std::string val_type_str(VAL_TYPE e) {
    switch (e) {
    case CONTEXT:
      return "CONTEXT";
    case STRUCT:
      return "STRUCT";
    case INT:
      return "INT";
    case BUF:
      return "BUF";
    case ATOM:
      return "ATOM";
    case UNKNOWN:
      return "UNKNOWN";
    default:
      assert(false);
      return "";
    }
  }
  // std::unordered_map<mlir::Location, std::string> arg_names;
  mlir::DenseMap<Value, std::string> arg_names;
  std::string getValName(mlir::Value val);
  void UpdateValName(mlir::Value val, std::string name);
  VAL_TYPE GetValTypeAndSize(mlir::Type type, int *size);
  unsigned getContextTotalSize(llvm::StringMap<ContextAnalysis::ContextField> &context);
  unsigned getStructTotalSize(ep2::StructType in_struct);
  unsigned getStructValOffset(ep2::StructType in_struct, int index);
  unsigned getStructValSize(ep2::StructType in_struct, int index);
  void emitFuncHeader(std::ofstream &file, ep2::FuncOp funcop);
  void emitVariableInit(std::ofstream &file, ep2::InitOp initop);
  void emitExtract(std::ofstream &file, ep2::ExtractOp extractop);
  void emitStructAccess(std::ofstream &file,
                        ep2::StructAccessOp structaccessop);
  void emitStructUpdate(std::ofstream &file,
                        ep2::StructUpdateOp structupdateop);
  void emitEmit(std::ofstream &file, ep2::EmitOp emitop);
  void emitReturn(std::ofstream &file, ep2::ReturnOp returnop);
  void emitHandler(ep2::FuncOp funcOp);
  void emitController(ep2::FuncOp funcOp);
};

} // namespace ep2
} // namespace mlir

#endif // EP2_FPGA_PASSES_H
