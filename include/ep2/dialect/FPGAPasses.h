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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "ep2/dialect/Dialect.h"

namespace mlir {
namespace ep2 {

#define DEFAULT_AXIS_STREAM_SIZE 512

struct EmitFPGAPass : public PassWrapper<EmitFPGAPass, OperationPass<>> {
  void runOnOperation() final;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<EP2Dialect>();
  }

  StringRef getArgument() const final { return "ep2-emit-fpga"; }
  StringRef getDescription() const final { return "Emit FPGA Code"; }

private:
  mlir::Operation *module;
  OpBuilder *builder;
  ContextBufferizationAnalysis *contextbufferAnalysis;
  HandlerInOutAnalysis *handlerInOutAnalysis;
  TableAnalysis *tableAnalysis;
  HandlerDependencyAnalysis *handlerDependencyAnalysis;
  FuncOp *cur_funcop;

  enum VAL_TYPE { CONTEXT, STRUCT, INT, BUF, ATOM, TABLE, UNKNOWN };

  enum INOUT { IN, OUT };

  enum IF_TYPE { AXIS, TABLE_LOOKUP, TABLE_UPDATE, BIT};

  struct axis_config {
    int if_keep;
    int if_last;
    int data_width;
    int user_width;
  };
  
  struct bits_config {
    int size;
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
      struct bits_config bit;
    };
  };

  struct wire_config {
    IF_TYPE type;
    std::string name;
    std::string debuginfo;
    bool if_init_value;
    std::string init_value;
    bool if_use;
    union {
      struct axis_config axis;
      struct table_if_config table_if;
      struct bits_config bit;
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
      struct bits_config bit;
    };
  };

  struct demux_inout_arg{
    int total_in_ports;
    int cur_in_port;
    std::vector<struct wire_config> in_ports_wires;
    struct wire_config out_port_wire;
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
    case TABLE:
      return "TABLE";
    case UNKNOWN:
      return "UNKNOWN";
    default:
      assert(false);
      return "";
    }
  }

  // an edge is an event, with a group of wires (all parameters in an event)
  struct handler_edge {
    INOUT direction;
    // int id;
    std::string eventname;
    bool if_extern;
    std::vector<struct wire_config> event_wires;
  };

  mlir::DenseMap<mlir::ep2::FuncOp, std::vector<struct handler_edge>> handler_in_edge_map, handler_out_edge_map;

  mlir::DenseMap<mlir::ep2::FuncOp, std::vector<struct module_port_config>> global_state_ports;
  struct inout_info {
    FuncOp funcop;
    int replicate_index;
    std::string eventname;
    std::vector<struct wire_config> event_wires;
    bool if_connect_to_extern;
  };

  mlir::DenseMap<mlir::ep2::FuncOp, mlir::DenseMap<Value, struct inout_info>> ctrl_ins, ctrl_outs;

  std::map<std::string, struct inout_config> extern_inouts;

  struct name_and_uses {
    std::string name;
    int total_uses;
    int cur_use;
  };

  std::string getExternArgName(std::string eventname, int argid){
    return eventname + "_" + std::to_string(argid);
  }

  // std::unordered_map<mlir::Location, std::string> arg_names;
  mlir::DenseMap<Value, struct name_and_uses> val_names_and_useid;
  int global_var_index = 0;

  // For each lookupop/updateop -> table lookup/update port name
  mlir::DenseMap<mlir::Operation*, struct wire_config> tableops_to_portwire;


  struct wire_config getBBDemuxInputWire(mlir::Block *bb, int arg_index){
    auto demux_inout = bb_to_demux_inout[bb][arg_index];
    auto wire = demux_inout.in_ports_wires[demux_inout.cur_in_port];
    demux_inout.cur_in_port++;
    bb_to_demux_inout[bb][arg_index] = demux_inout;
    return wire;
  }

  // For each basic block ->  inout demux's agr lists (each arg has multiple inports and one outport)
  mlir::DenseMap<mlir::Block*, std::vector<struct demux_inout_arg>> bb_to_demux_inout;

  bool has_use(mlir::Value val) { return !val.getUses().empty(); }

  int val_use_count(mlir::Value val) {   
    auto uses = val.getUses();
    int i = 0;
    for(auto &u: uses){
      i++;
    } 
    return i;
  }

  bool if_axis_stream(VAL_TYPE valtype) {
    bool if_stream = false;
    if (valtype == BUF) {
      if_stream = true;
    }
    return if_stream;
  }

  struct top_handler_inout_wires{
      bool if_extern;
      bool if_connected;
      std::vector<struct wire_config> event_wires;
  };


  struct global_table_info{
      std::string module_name;
      std::string name;
      std::list<mlir::ep2::EmitFPGAPass::module_port_config> ports;
      std::list<mlir::ep2::EmitFPGAPass::module_param_config> params;
      std::list<mlir::ep2::EmitFPGAPass::wire_config> lookup_wires;
      std::list<mlir::ep2::EmitFPGAPass::wire_config> update_wires;
  };

  std::map<std::string, struct global_table_info> global_tables;

  std::string assignValNameAndUpdate(mlir::Value val, std::string prefix, bool if_add_gindex=true);
  
  std::string getValName(mlir::Value val);
  VAL_TYPE GetValTypeAndSize(mlir::Type type, int *size);

  void emitModuleParameter(std::ofstream &file,
                           std::vector<struct inout_config> &wires);
  void emitReplicationModule(std::ofstream &file, struct wire_config &wire, int replicas=1);
  void emitonewire(std::ofstream &file, struct wire_config &wire);
  void emitwire(std::ofstream &file, struct wire_config &wire, int replicas=1, bool if_emit_replica_src = true);
  void emitwireassign(std::ofstream &file, struct wire_assign_config &assign);

  void emitModuleCall(std::ofstream &file, std::string module_type,
                      std::string module_name,
                      std::list<struct module_port_config> &ports,
                      std::list<struct module_param_config> &params);

 std::vector<mlir::ep2::EmitFPGAPass::wire_config> emitGuardPredModule(std::ofstream &file, ep2::GuardOp gop, int repd_out_num);

  struct wire_config emitGuardModule(std::ofstream &file, struct wire_config inputwire, wire_config predwire);

  unsigned getContextTotalSize();
  unsigned getStructTotalSize(ep2::StructType in_struct);
  unsigned getStructValOffset(ep2::StructType in_struct, int index);
  unsigned getStructValSize(ep2::StructType in_struct, int index);

  std::string assign_name(std::string prefix);

  void emitFuncHeader(std::ofstream &file, ep2::FuncOp funcop);
  void emitVariableInit(std::ofstream &file, ep2::InitOp initop);
  void emitTableInit(std::ofstream &file, ep2::InitOp initop);
  void emitGlobalTableInit(ep2::GlobalOp globalop);
  void emitLookup(std::ofstream &file, ep2::LookupOp lookupop);
  void emitUpdate(std::ofstream &file, ep2::UpdateOp updateop, bool if_guarded = false, ep2::GuardOp gop = nullptr);
  void emitUpdateAtomic(std::ofstream &file, ep2::UpdateAtomicOp updateop, bool if_guarded = false, ep2::GuardOp gop = nullptr);
  void emitExtract(std::ofstream &file, ep2::ExtractValueOp extractop);
  void emitBBInputDemux(std::ofstream &file, ep2::FuncOp funcOp);
  void emitBBCondBranch(std::ofstream &file, cf::CondBranchOp condbranchop);
  void emitBBBranch(std::ofstream &file, cf::BranchOp branchop);
  void emitStructAccess(std::ofstream &file,
                        ep2::StructAccessOp structaccessop);
  void emitStructUpdate(std::ofstream &file,
                        ep2::StructUpdateOp structupdateop);
  void emitEmit(std::ofstream &file, ep2::EmitValueOp emitop);
  void emitReturn(std::ofstream &file, ep2::ReturnOp returnop, bool if_guarded = false, ep2::GuardOp gop = nullptr);
  void emitConst(std::ofstream &file, ep2::ConstantOp constop);
  void emitArithmetic(std::ofstream &file, mlir::Operation *op);
  void emitIfElse(std::ofstream &file, scf::IfOp ifop);
  void emitBitcast(std::ofstream &file, ep2::BitCastOp bitcastop);
  void emitSelect(std::ofstream &file, arith::SelectOp selectop);
  void emitSink(std::ofstream &file, ep2::SinkOp sinkop);
  void emitHandler(ep2::FuncOp funcOp);
  void emitController(ep2::FuncOp funcOp);
  void emitGuard(std::ofstream &file, ep2::GuardOp guardop);
  void emitGlobalImport(std::ofstream &file, ep2::GlobalImportOp importop);

  void emitOp(std::ofstream &file, mlir::Operation *op);

  void emitTop();
  void emitControllerTop();


  void emitControllerInOut(std::ofstream &file, ep2::FuncOp funcOp);
  void emitControllerMux(std::ofstream &file, ep2::ConnectOp conncetop);
};

} // namespace ep2
} // namespace mlir

#endif // EP2_FPGA_PASSES_H
