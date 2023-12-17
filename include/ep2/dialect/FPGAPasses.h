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

#include <iostream>
#include <list>
#include <memory>
#include <string>
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

#define DEFAULT_AXIS_STREAM_SIZE 512

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
  TableAnalysis *tableAnalysis;
  FuncOp *cur_funcop;

  enum VAL_TYPE { CONTEXT, STRUCT, INT, BUF, ATOM, UNKNOWN };

  enum INOUT { IN, OUT };

  enum IF_TYPE { AXIS, TABLE_LOOKUP, TABLE_UPDATE};

  struct axis_config {
    int if_keep;
    int if_last;
    int data_width;
  };

  struct table_if_config {
    int index_width;
    int data_width;
  };

  struct inout_config {
    INOUT direction;
    IF_TYPE type;
    std::string name;
    std::string debuginfo;
    union {
      struct axis_config axis;
      struct table_if_config table_if;
    };
  };

  struct wire_config {
    IF_TYPE type;
    std::string name;
    std::string debuginfo;
    bool if_init_value;
    int init_value;
    bool if_use;
    union {
      struct axis_config axis;
      struct table_if_config table_if;
    };
  };

  struct wire_assign_config {
    int src_wire_offset = -1;
    int src_wire_size = -1;
    int dst_wire_offset = -1;
    int dst_wire_size = -1;

    struct wire_config src_wire;
    struct wire_config dst_wire;
  };

  struct module_param_config {
    std::string paramname;
    int paramval;
  };

  struct module_port_config {
    IF_TYPE type;
    std::vector<std::string> var_name;
    std::string debuginfo;
    std::string port_name;
    union {
      struct axis_config axis;
      struct table_if_config table_if;
    };
  };

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
  int global_var_index = 0;

  // For each lookupop/updateop -> table lookup/update port name
  mlir::DenseMap<mlir::Operation*, struct wire_config> tableops_to_portwire;

  bool has_use(mlir::Value val) { return !val.getUses().empty(); }
  std::string getValName(mlir::Value val);
  void UpdateValName(mlir::Value val, std::string name);
  VAL_TYPE GetValTypeAndSize(mlir::Type type, int *size);

  void emitModuleParameter(std::ofstream &file,
                           std::list<struct inout_config> &wires);
  void emitwire(std::ofstream &file, struct wire_config &wire);
  void emitwireassign(std::ofstream &file, struct wire_assign_config &assign);

  void emitModuleCall(std::ofstream &file, std::string module_type,
                      std::string module_name,
                      std::list<struct module_port_config> &ports,
                      std::list<struct module_param_config> &params);

  unsigned
  getContextTotalSize(llvm::StringMap<ContextAnalysis::ContextField> &context);
  unsigned getStructTotalSize(ep2::StructType in_struct);
  unsigned getStructValOffset(ep2::StructType in_struct, int index);
  unsigned getStructValSize(ep2::StructType in_struct, int index);

  std::string assign_var_name(std::string prefix);

  void emitFuncHeader(std::ofstream &file, ep2::FuncOp funcop);
  void emitVariableInit(std::ofstream &file, ep2::InitOp initop);
  void emitTableInit(std::ofstream &file, ep2::InitOp initop);
  void emitLookup(std::ofstream &file, ep2::LookupOp lookupop);
  void emitUpdate(std::ofstream &file, ep2::UpdateOp updateop);
  void emitExtract(std::ofstream &file, ep2::ExtractOp extractop);
  void emitStructAccess(std::ofstream &file,
                        ep2::StructAccessOp structaccessop);
  void emitStructUpdate(std::ofstream &file,
                        ep2::StructUpdateOp structupdateop);
  void emitEmit(std::ofstream &file, ep2::EmitOp emitop);
  void emitReturn(std::ofstream &file, ep2::ReturnOp returnop);
  void emitConst(std::ofstream &file, ep2::ConstantOp constop);
  void emitHandler(ep2::FuncOp funcOp);
  void emitController(ep2::FuncOp funcOp);
};

} // namespace ep2
} // namespace mlir

#endif // EP2_FPGA_PASSES_H
