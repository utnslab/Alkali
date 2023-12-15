#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "ep2/dialect/FPGAPasses.h"
#include "mlir/IR/BuiltinDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <fstream>
#include <iostream>
#include <list>
#include <string>

using namespace mlir;

namespace mlir {
namespace ep2 {
void EmitFPGAPass::emitFuncHeader(std::ofstream &file, ep2::FuncOp funcOp) {
  auto handler_name = funcOp.getName().str();
  file << "module " << handler_name << "#()\n";

  std::list<struct inout_config> inout_wires;

  // push input parameter wire
  auto args = handlerInOutAnalysis->handler_in_arg_list[funcOp];
  for (auto arg : args) {
    auto arg_type = arg.getType();
    bool if_stream;
    int size;
    std::string name, debuginfo;

    auto valtype = GetValTypeAndSize(arg_type, &size);
    debuginfo = "input ports " + val_type_str(valtype);
    if ((valtype == CONTEXT && size == 0) || valtype == ATOM) {
      continue;
    } else if (valtype == CONTEXT || valtype == INT || valtype == STRUCT) {
      if_stream = false;
    } else if (valtype == BUF) {
      if_stream = true;
    } else {
      printf("Error: Cannot generate in parameter wire for\n");
      arg_type.dump();
      assert(false);
    }

    name = "arg" + std::to_string(arg.getArgNumber());
    UpdateValName(arg, name);
    if (!if_stream) {
      assert(size <= 64 * 8);
    } else {
      assert(size % 8 == 0);
    }
    struct axis_config wire = {if_stream, if_stream, size};
    struct inout_config in_if = {IN, AXIS, name, debuginfo, wire};
    inout_wires.push_back(in_if);
  }

  // push output parameter wires
  auto returnvals = handlerInOutAnalysis->handler_returnop_list[funcOp];
  for (auto returned_event : returnvals) {
    auto returned_event_type = returned_event.getType();
    assert(!returned_event.getDefiningOp()->hasAttr("var_name"));
    auto name = assign_var_name("outport");
    UpdateValName(returned_event, name);

    assert(isa<ep2::StructType>(returned_event_type));
    auto return_event_struct = cast<ep2::StructType, Type>(returned_event_type);
    auto field_types = return_event_struct.getElementTypes();
    for (int i = 0; i < field_types.size(); i++) {
      bool if_stream;
      int size;
      std::string portname, debuginfo;

      auto valtype = GetValTypeAndSize(field_types[i], &size);
      debuginfo = "output ports " + val_type_str(valtype);
      if ((valtype == CONTEXT && size == 0) || valtype == ATOM) {
        continue;
      } else if (valtype == CONTEXT || valtype == INT || valtype == STRUCT) {
        if_stream = false;
      } else if (valtype == BUF) {
        if_stream = true;
      } else {
        printf("Error: Cannot generate in parameter wire for\n");
        field_types[i].dump();
        assert(false);
      }

      portname = name + "_" + std::to_string(i);

      struct axis_config wire = {if_stream, if_stream, size};
      struct inout_config out_if = {OUT, AXIS, portname, debuginfo, wire};
      inout_wires.push_back(out_if);
    }
  }

  emitModuleParameter(file, inout_wires);
}

void EmitFPGAPass::emitVariableInit(std::ofstream &file, ep2::InitOp initop) {

  auto arg = initop.getResult();
  auto arg_type = arg.getType();
  bool if_stream;
  int size;
  std::string name, debuginfo;

  auto valtype = GetValTypeAndSize(arg_type, &size);
  debuginfo = "inited_" + val_type_str(valtype);
  if (valtype == INT || valtype == STRUCT) {
    if_stream = false;
  } else if (valtype == BUF) {
    if_stream = true;
  } else {
    printf("Error: Cannot emitVariableInit\n");
    initop.dump();
    assert(false);
  }

  assert(!arg.getDefiningOp()->hasAttr("var_name"));
  name = assign_var_name(debuginfo);
  // defined_value->setAttr("var_name", builder->getStringAttr(name));
  UpdateValName(arg, name);

  if (!if_stream) {
    assert(size <= 64 * 8);
  } else {
    assert(size % 8 == 0);
  }
  struct axis_config axis = {if_stream, if_stream, size};
  struct wire_config wire = {AXIS, name, debuginfo, true, true, axis};
  emitwire(file, wire);

  // printf("Get Init op %s\n", defined_value->hashProperties());
}

bool has_use(mlir::Value val) { return !val.getUses().empty(); }

void EmitFPGAPass::emitExtract(std::ofstream &file, ep2::ExtractOp extractop) {
  // First emit the wire define for: output buf, output struct
  // Then emit the extract module call
  struct module_port_config in_buf_port, out_buf_port, out_struct_port;
  struct wire_config out_buf_wire, out_struct_wire;
  struct axis_config in_buf_axis, out_buf_axis, out_struct_axis;
  auto module_name = assign_var_name("extract_module");

  auto buf = extractop.getBuffer();
  std::string ori_buf_name = getValName(buf);
  in_buf_axis = {1, 1, DEFAULT_AXIS_STREAM_SIZE};
  in_buf_port = {
      AXIS, {ori_buf_name}, "input buf", "s_inbuf_axis", in_buf_axis};

  auto new_buf_name = assign_var_name("bufvar");
  UpdateValName(buf, new_buf_name);
  out_buf_axis = {1, 1, DEFAULT_AXIS_STREAM_SIZE};
  out_buf_port = {
      AXIS, {new_buf_name}, "output buf", "m_outbuf_axis", out_buf_axis};
  out_buf_wire = {AXIS,  new_buf_name, module_name + " output buf",
                  false, has_use(buf), out_buf_axis};

  auto extracted_struct = extractop.getResult();
  assert(!extracted_struct.getDefiningOp()->hasAttr("var_name"));
  auto extracted_struct_name = assign_var_name("structvar");
  UpdateValName(extracted_struct, extracted_struct_name);
  auto extracted_struct_type = extracted_struct.getType();
  int extracted_struct_size = 0;
  // struct extract value can only be struct or int
  auto valtype =
      GetValTypeAndSize(extracted_struct_type, &extracted_struct_size);
  if (!(valtype == INT || valtype == STRUCT)) {
    printf("Error: Cannot calculate emitExtract's output struct size\n");
    extracted_struct.dump();
    assert(false);
  }

  out_struct_axis = {0, 0, extracted_struct_size};
  out_struct_port = {AXIS,
                     {extracted_struct_name},
                     "output struct",
                     "m_extracted_axis",
                     out_struct_axis};
  out_struct_wire = {
      AXIS,  extracted_struct_name,     module_name + " output struct",
      false, has_use(extracted_struct), out_struct_axis};

  std::list<struct module_port_config> ports;
  ports.push_back(in_buf_port);
  ports.push_back(out_buf_port);
  ports.push_back(out_struct_port);

  emitwire(file, out_buf_wire);
  emitwire(file, out_struct_wire);

  std::list<struct module_param_config> params;
  params.push_back({"BUF_DATA_WIDTH", DEFAULT_AXIS_STREAM_SIZE});
  params.push_back({"BUF_KEEP_WIDTH", DEFAULT_AXIS_STREAM_SIZE / 8});
  params.push_back({"EXTRACTED_STRUCT_WIDTH", extracted_struct_size});

  emitModuleCall(file, "extract", module_name, ports, params);
}

void EmitFPGAPass::emitEmit(std::ofstream &file, ep2::EmitOp emitop) {
  // First emit the wire define for: output buf, output struct
  // Then emit the extract module call
  struct module_port_config in_buf_port, in_struct_port, out_buf_port;
  struct wire_config out_buf_wire;
  struct axis_config in_buf_axis, in_struct_axis, out_buf_axis;
  auto module_name = assign_var_name("emit_module");

  auto buf = emitop.getBuffer();
  std::string ori_buf_name = getValName(buf);
  in_buf_axis = {1, 1, DEFAULT_AXIS_STREAM_SIZE};
  in_buf_port = {
      AXIS, {ori_buf_name}, "input buf", "s_inbuf_axis", in_buf_axis};

  auto new_buf_name = assign_var_name("bufvar");
  UpdateValName(buf, new_buf_name);
  out_buf_axis = {1, 1, DEFAULT_AXIS_STREAM_SIZE};
  out_buf_port = {
      AXIS, {new_buf_name}, "output buf", "m_outbuf_axis", out_buf_axis};
  out_buf_wire = {AXIS,  new_buf_name, module_name + " output buf",
                  false, has_use(buf), out_buf_axis};

  auto input_struct = emitop.getValue();
  auto input_struct_name = getValName(input_struct);
  auto input_struct_type = input_struct.getType();
  int input_struct_size = 0;
  int if_input_is_buf = 0;
  // struct emit value can be struct or int or buf
  auto valtype = GetValTypeAndSize(input_struct_type, &input_struct_size);
  if (!(valtype == INT || valtype == STRUCT || valtype == BUF)) {
    printf("Error: Cannot calculate emitEmit's input struct size\n");
    emitop.dump();
    input_struct_type.dump();
    assert(false);
  }
  // if if_input_is_buf, give tkeep and last always 1
  if_input_is_buf = (valtype == BUF);

  in_struct_axis = {if_input_is_buf, if_input_is_buf, input_struct_size};
  in_struct_port = {AXIS,
                    {input_struct_name},
                    "input struct/buf",
                    "s_struct_axis",
                    in_struct_axis};

  std::list<struct module_port_config> ports;
  ports.push_back(in_buf_port);
  ports.push_back(in_struct_port);
  ports.push_back(out_buf_port);

  emitwire(file, out_buf_wire);

  std::list<struct module_param_config> params;
  params.push_back({"BUF_DATA_WIDTH", DEFAULT_AXIS_STREAM_SIZE});
  params.push_back({"BUF_KEEP_WIDTH", DEFAULT_AXIS_STREAM_SIZE / 8});
  params.push_back({"IF_INPUT_BUF", if_input_is_buf});
  params.push_back({"INPUT_BUF_STRUCT_WIDTH", input_struct_size});

  emitModuleCall(file, "emit", module_name, ports, params);
}

void EmitFPGAPass::emitStructAccess(std::ofstream &file,
                                    ep2::StructAccessOp structaccessop) {

  auto outval = structaccessop.getResult();
  auto outval_type = outval.getType();
  auto module_name = assign_var_name("struct_access");
  struct module_port_config outval_port, src_struct_port, new_struct_port;
  struct axis_config outval_axis, src_struct_axis, new_struct_axis;
  struct wire_config outval_wire, src_struct_wire, new_struct_wire;
  // struct wire_assign_config wire_assignment;
  int size;
  std::string debuginfo;

  // struct access value can only be struct or int
  auto valtype = GetValTypeAndSize(outval_type, &size);
  debuginfo = "struct_accessed_" + val_type_str(valtype);
  if (!(valtype == INT || valtype == STRUCT)) {
    printf("Error: Cannot emitStructAccess's output val\n");
    outval_type.dump();
    assert(false);
  }

  assert(!outval.getDefiningOp()->hasAttr("var_name"));
  auto name = assign_var_name(debuginfo);
  // outval.getDefiningOp()->setAttr("var_name", builder->getStringAttr(name));
  UpdateValName(outval, name);
  outval_axis = {0, 0, size};
  outval_wire = {AXIS,       name, "Access Struct", false, has_use(outval),
                 outval_axis};
  outval_port = {AXIS, {name}, "output val", "m_val_axis", outval_axis};

  auto srcval = structaccessop.getInput();
  auto srcval_index = structaccessop.getIndex();
  assert(isa<ep2::StructType>(srcval.getType()));

  auto srcval_type = cast<ep2::StructType, Type>(srcval.getType());
  int src_offset = getStructValOffset(srcval_type, srcval_index);
  int src_size = getStructValSize(srcval_type, srcval_index);
  int src_struct_size = getStructTotalSize(srcval_type);

  auto src_struct_name = getValName(srcval);
  src_struct_axis = {0, 0, src_struct_size};
  src_struct_wire = {AXIS,  src_struct_name, "Struct Assign Src Struct",
                     false, false,           src_struct_axis};
  src_struct_port = {AXIS,
                     {src_struct_name},
                     "struct input",
                     "s_struct_axis",
                     src_struct_axis};

  // TODO, we don't want to update the symbol table, but we want to use another
  // IR op

  auto new_struct_name = assign_var_name("structvar");
  new_struct_wire.name = new_struct_name;
  UpdateValName(srcval, new_struct_name);
  new_struct_axis = {0, 0, src_struct_size};
  ;
  new_struct_wire = {AXIS,  new_struct_name, "Struct Assign new Struct",
                     false, has_use(srcval), new_struct_axis};
  ;
  new_struct_port = {AXIS,
                     {new_struct_name},
                     "struct output",
                     "m_struct_axis",
                     new_struct_axis};

  std::list<struct module_port_config> ports;
  ports.push_back(src_struct_port);
  ports.push_back(new_struct_port);
  ports.push_back(outval_port);

  emitwire(file, outval_wire);
  emitwire(file, new_struct_wire);

  std::list<struct module_param_config> params;
  params.push_back({"STRUCT_WIDTH", src_struct_size});
  params.push_back({"ACCESS_OFFSET", src_offset});
  params.push_back({"ACCESS_SIZE", src_size});

  emitModuleCall(file, "struct_access", module_name, ports, params);
  // wire_assignment = {src_offset, src_size, -1, -1, src_struct_wire,
  // outval_wire}; emitwireassign(file, wire_assignment);
}

void EmitFPGAPass::emitStructUpdate(std::ofstream &file,
                                    ep2::StructUpdateOp structupdateop) {
  struct module_port_config ori_struct_port, in_val_port, new_struct_port;
  struct wire_config new_struct_wire;
  struct axis_config ori_struct_axis, in_val_axis, new_struct_axis;
  auto module_name = assign_var_name("struct_assign");

  auto ori_struct = structupdateop.getInput();
  auto ori_struct_name = getValName(ori_struct);
  assert(isa<ep2::StructType>(ori_struct.getType()));
  int ori_struct_size =
      getStructTotalSize(cast<ep2::StructType, Type>(ori_struct.getType()));
  ori_struct_axis = {0, 0, ori_struct_size};
  ori_struct_port = {AXIS,
                     {ori_struct_name},
                     "input struct",
                     "s_struct_axis",
                     ori_struct_axis};

  auto in_val = structupdateop.getNewValue();
  auto in_val_name = getValName(in_val);
  auto in_vale_type = in_val.getType();
  int in_val_size = 0;
  if (in_vale_type.isIntOrFloat()) {
    in_val_size = in_vale_type.getIntOrFloatBitWidth();
  } else if (isa<ep2::StructType>(in_vale_type)) {
    auto val_struct = cast<ep2::StructType, Type>(in_vale_type);
    in_val_size = getStructTotalSize(val_struct);
  } else {
    printf("Error: Cannot emitStructAccess's output val\n");
    in_val.dump();
    assert(false);
  }
  printf("%s, val size %d\n", in_val_name.c_str(), in_val_size);
  in_val.dump();
  in_vale_type.dump();
  in_val_axis = {0, 0, in_val_size};
  in_val_port = {
      AXIS, {in_val_name}, "input val", "s_assignv_axis", in_val_axis};

  auto new_struct = structupdateop.getOutput();
  auto new_struct_name = assign_var_name("structvar");
  auto new_struct_type = new_struct.getType();
  UpdateValName(new_struct, new_struct_name);
  assert(isa<ep2::StructType>(new_struct_type));
  int new_struct_size =
      getStructTotalSize(cast<ep2::StructType, Type>(new_struct_type));
  assert(new_struct_size == ori_struct_size);
  new_struct_axis = {0, 0, new_struct_size};
  new_struct_port = {AXIS,
                     {new_struct_name},
                     "output struct",
                     "m_struct_axis",
                     new_struct_axis};
  new_struct_wire = {AXIS,  new_struct_name,     module_name + " output struct",
                     false, has_use(new_struct), new_struct_axis};

  std::list<struct module_port_config> ports;
  ports.push_back(ori_struct_port);
  ports.push_back(in_val_port);
  ports.push_back(new_struct_port);

  emitwire(file, new_struct_wire);

  std::list<struct module_param_config> params;
  int assign_index = structupdateop.getIndex();
  int assign_offset = getStructValOffset(new_struct_type, assign_index);
  params.push_back({"STRUCT_WIDTH", new_struct_size});
  params.push_back({"ASSIGN_OFFSET", assign_offset});
  params.push_back({"ASSIGN_SIZE", in_val_size});

  emitModuleCall(file, "struct_assign", module_name, ports, params);
}

void EmitFPGAPass::emitReturn(std::ofstream &file, ep2::ReturnOp returnop) {
  if (returnop.getNumOperands() == 0)
    return;

  assert(returnop.getNumOperands() == 1);
  // get output port struct
  auto out_port = returnop.getOperand(0);
  auto out_port_type = out_port.getType();
  assert(isa<ep2::StructType>(out_port_type));
  auto out_port_struct = cast<ep2::StructType, Type>(out_port_type);
  auto out_port_fileds = out_port_struct.getElementTypes();

  // get val that is going to be connected with the output port
  auto op = returnop.getOperand(0).getDefiningOp();
  assert(isa<ep2::InitOp>(op));
  auto initop = cast<ep2::InitOp, mlir::Operation *>(op);
  auto initop_args = initop.getArgs();

  assert(initop_args.size() == out_port_fileds.size());

  for (int i = 0; i < initop_args.size(); i++) {
    // assign wires to output ports
    struct wire_assign_config wire_assignment;
    struct axis_config value_axis, outports_axis;
    struct wire_config value_wire, outports_wire;

    initop_args[i].getType().dump();
    out_port_fileds[i].dump();
    assert(initop_args[i].getType() == out_port_fileds[i]);

    int size = 0;
    bool if_stream;

    auto src_value = initop_args[i];
    auto valtype = GetValTypeAndSize(src_value.getType(), &size);

    if (valtype == CONTEXT && size == 0) {
      continue;
    }
    if (valtype == CONTEXT || valtype == INT || valtype == STRUCT) {
      if_stream = false;
    } else if (valtype == BUF) {
      if_stream = true;
    }

    auto src_value_name = getValName(src_value);
    value_axis = {if_stream, if_stream, size};
    value_wire = {AXIS,  src_value_name, "Outport Assign Src Value",
                  false, true,           value_axis};

    auto dst_port_base_name = getValName(out_port);
    auto dst_port_name = dst_port_base_name + "_" + std::to_string(i);
    outports_axis = value_axis;
    outports_wire = {AXIS,  dst_port_name, "Outport Assign Dst Port",
                     false, true,          outports_axis};
    wire_assignment = {-1, -1, -1, -1, value_wire, outports_wire};
    emitwireassign(file, wire_assignment);
  }
}
void EmitFPGAPass::emitHandler(ep2::FuncOp funcOp) {
  cur_funcop = &funcOp;
  auto handler_name = funcOp.getName().str();
  std::ofstream fout_stage(handler_name + ".sv");
  emitFuncHeader(fout_stage, funcOp);

  funcOp->walk([&](mlir::Operation *op) {
    if (isa<ep2::InitOp>(op)) {
      auto initop = cast<ep2::InitOp, mlir::Operation *>(op);
      if (initop.getArgs().size() == 0)
        emitVariableInit(fout_stage, initop);
      else {
        // Otherwise this init output event for this hanlder;
      }
    } else if (isa<ep2::ExtractOp>(op)) {
      auto extractop = cast<ep2::ExtractOp, mlir::Operation *>(op);
      emitExtract(fout_stage, extractop);
    } else if (isa<ep2::StructAccessOp>(op)) {
      auto structaccessop = cast<ep2::StructAccessOp, mlir::Operation *>(op);
      emitStructAccess(fout_stage, structaccessop);
    } else if (isa<ep2::StructUpdateOp>(op)) {
      auto structupdateop = cast<ep2::StructUpdateOp, mlir::Operation *>(op);
      emitStructUpdate(fout_stage, structupdateop);
    } else if (isa<ep2::EmitOp>(op)) {
      auto emitop = cast<ep2::EmitOp, mlir::Operation *>(op);
      emitEmit(fout_stage, emitop);
    } else if (isa<ep2::ReturnOp>(op)) {
      auto returnop = cast<ep2::ReturnOp, mlir::Operation *>(op);
      emitReturn(fout_stage, returnop);
    }
    // TODO: Add OP Constant
    // TODO: Change STURCT ACCESS IR to generate new stream for each accessed
    // struct
  });
  fout_stage << "\nendmodule\n";
}

void EmitFPGAPass::emitController(ep2::FuncOp funcOp) {
  cur_funcop = &funcOp;
  auto handler_name = funcOp.getName().str();
  std::ofstream file(handler_name + ".sv");
  file << "module " << handler_name << "#()\n";

  // from m replicas to n replicas
  std::list<struct inout_config> inout_wires;
  std::list<struct wire_config> mux_in_wires;
  std::list<struct wire_config> demux_out_wires;
  // TODO: should have a analysis struct:
  // std::unordered_map<ep2::FuncOp, std::list<ep2::FuncOp>>
  // controller_srcs; std::unordered_map<ep2::FuncOp,
  // std::list<ep2::FuncOp>> controller_dsts;

  // vvv
  int src_count = 2;
  int dst_count = 2;
  mlir::Value tmped_event;
  for (auto &temp : handlerInOutAnalysis->handler_returnop_list) {
    if (temp.second.size() != 0) {
      tmped_event = temp.second[0];
    }
  }
  // ^^^

  auto event_type = tmped_event.getType();

  assert(isa<ep2::StructType>(event_type));
  auto return_event_struct = cast<ep2::StructType, Type>(event_type);
  // TODO: This need special care -- event size contains buf/context and itis
  // not simply a struct..
  int eventsize = 233;
  bool if_stream = true; // event is a stream
  struct axis_config event_axis = {1, 1, eventsize};

  std::string portname, debuginfo;

  for (int src_id = 0; src_id < src_count; src_id++) {
    auto name = assign_var_name("inport");
    debuginfo = "input event from src port " + std::to_string(src_id);
    portname = name + "_" + std::to_string(src_id);
    struct axis_config wire = {if_stream, if_stream, eventsize};
    struct inout_config out_if = {IN, AXIS, portname, debuginfo, wire};
    inout_wires.push_back(out_if);
    // Collect input port wire information for further wiring
    mux_in_wires.push_back(
        {AXIS, portname, debuginfo + "wire", false, true, wire});
  }

  for (int dst_id = 0; dst_id < dst_count; dst_id++) {
    auto name = assign_var_name("outport");
    debuginfo = "output event for dst port " + std::to_string(dst_id);
    portname = name + "_" + std::to_string(dst_id);
    struct axis_config wire = {if_stream, if_stream, eventsize};
    struct inout_config out_if = {OUT, AXIS, portname, debuginfo, wire};
    inout_wires.push_back(out_if);
    // Collect output port wire information for further wiring
    demux_out_wires.push_back(
        {AXIS, portname, debuginfo + "wire", false, true, wire});
  }

  emitModuleParameter(file, inout_wires);

  bool if_enable_mux = (src_count > 1);
  bool if_enable_demux = (dst_count > 1);
  std::vector<std::string> fifo_invar_names;
  if (if_enable_mux) {
    //  emit mux's output wire defination
    struct module_port_config mux_in_port, mux_out_port;
    struct wire_config mux_out_wire;
    auto mux_out_name = assign_var_name("mux_out");
    mux_out_wire.name = mux_out_name;
    mux_out_wire = {AXIS,  mux_out_name, "Mux output wire",
                    false, true,         event_axis};
    mux_out_port = {AXIS, {mux_out_name}, "mux output", "m_axis", event_axis};

    emitwire(file, mux_out_wire);

    // emit mux call
    std::list<struct module_port_config> ports;
    std::vector<std::string> var_names;
    for (auto &w : mux_in_wires) {
      var_names.push_back(w.name);
    }
    mux_in_port = {AXIS, var_names, "mux input", "s_axis", event_axis};

    ports.push_back(mux_in_port);
    ports.push_back(mux_out_port);
    // MUX parameters
    std::list<struct module_param_config> params;
    params.push_back({"S_COUNT  ", src_count});
    params.push_back({"DATA_WIDTH", eventsize});
    params.push_back({"KEEP_ENABLE ", 1});
    params.push_back({"USER_ENABLE ", 0});

    emitModuleCall(file, "axis_arb_mux ", "axis_arb_mux", ports, params);

    fifo_invar_names = {mux_out_name};
  } else {
    fifo_invar_names = {inout_wires.front().name};
  }

  // EMIT FIFO + disptacher
  struct module_port_config fifo_in_port, fifo_out_port;
  std::list<struct module_port_config> fifoports;
  fifo_in_port = {AXIS, fifo_invar_names, "queue input", "s_axis", event_axis};

  std::vector<std::string> fifo_out_var_names;
  for (auto &w : demux_out_wires) {
    fifo_out_var_names.push_back(w.name);
  }
  fifo_out_port = {AXIS, fifo_out_var_names, "queue output", "m_axis",
                   event_axis};
  fifoports.push_back(fifo_in_port);
  fifoports.push_back(fifo_out_port);

  // MUX parameters
  std::list<struct module_param_config> params;
  params.push_back({"D_COUNT  ", dst_count});
  params.push_back({"DATA_WIDTH", eventsize});
  params.push_back({"KEEP_ENABLE ", 1});
  params.push_back({"USER_ENABLE ", 0});
  params.push_back({"QUEUE_TYPE ", 0});
  params.push_back({"QUEUE_SIZE ", 128});

  emitModuleCall(file, "dispatch_queue", "queue", fifoports, params);
  file << "\nendmodule\n";
}

void EmitFPGAPass::runOnOperation() {
  module = getOperation();
  OpBuilder builder_tmp(module->getContext());
  builder = &builder_tmp;

  contextAnalysis = &(getAnalysis<ContextAnalysis>());
  handlerInOutAnalysis = &(getAnalysis<HandlerInOutAnalysis>());
  module->walk([&](ep2::FuncOp funcOp) {
    auto functype = funcOp->getAttr("type").cast<StringAttr>().getValue().str();
    std::cout << functype << "\n";
    if (functype == "handler") {
      printf("ENTERR\n");
      emitHandler(funcOp);
    } else if (functype == "controller") {
      emitController(funcOp);
    }
  });
}

std::unique_ptr<Pass> createEmitFPGAPass() {
  return std::make_unique<EmitFPGAPass>();
}

} // namespace ep2
} // namespace mlir
