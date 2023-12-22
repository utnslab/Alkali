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

  std::vector<struct inout_config> inout_wires;

  // log event wire info, used in generating top module
  std::vector<struct wire_config> in_event_wires;
  // push input parameter wire
  auto args = handlerInOutAnalysis->handler_in_arg_list[funcOp];
  std::string extern_in_event_name = "";
  // TODO: Support out event extern identification
  if(funcOp->hasAttr("in_hw_event")){
    extern_in_event_name = funcOp->getAttr("event").cast<StringAttr>().getValue().str();
  }
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

    if(extern_in_event_name!= "")
      name = extern_in_event_name + "_" + std::to_string(arg.getArgNumber());
    else
      name = assign_var_name("arg");
      
    UpdateValName(arg, name);
    if (!if_stream) {
      assert(size <= 64 * 8);
    } else {
      assert(size % 8 == 0);
    }
    struct axis_config wire = {if_stream, if_stream, size};
    struct wire_config event_wire = {AXIS, name, "", false, -1, true, wire};
    struct inout_config in_if = {IN, AXIS, name, debuginfo, wire};
    inout_wires.push_back(in_if);
    in_event_wires.push_back(event_wire);
    
    if(extern_in_event_name != "")
      extern_inouts.push_back(in_if);
  }
  handler_edge_map[*cur_funcop].push_back({IN, 0, extern_in_event_name != "", in_event_wires});

  // push output parameter wires
  auto returnvals = handlerInOutAnalysis->handler_returnop_list[funcOp];
  int out_event_id = 0;
  for (auto returned_event : returnvals) {
    // log event wire info, used in generating top module
    std::vector<struct wire_config> out_event_wires;

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
      struct wire_config event_wire = {AXIS, portname, "", false, -1, true, wire};
      struct inout_config out_if = {OUT, AXIS, portname, debuginfo, wire};
      inout_wires.push_back(out_if);
      out_event_wires.push_back(event_wire);
    }
    handler_edge_map[*cur_funcop].push_back({OUT, out_event_id++, false, out_event_wires});
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
  struct wire_config wire = {AXIS, name, debuginfo, true, 0, true, axis};
  emitwire(file, wire);
}


void EmitFPGAPass::emitTableInit(std::ofstream &file, ep2::InitOp initop) {
  auto table = cast<ep2::TableType, Type>(initop.getType());
  auto tabel_name = assign_var_name("table");
  // get all lookup uses
  // get all update uses
  auto lookups = tableAnalysis->table_lookup_uses[initop.getResult()];
  auto updates = tableAnalysis->table_update_uses[initop.getResult()];

  int key_size, value_size, table_size;
  auto key_type = table.getKeyType();
  GetValTypeAndSize(key_type, &key_size);
  auto value_type = table.getValueType();
  GetValTypeAndSize(value_type, &value_size);
  table_size = table.getSize();


  struct table_if_config table_if = {key_size, value_size};
  std::list<struct module_port_config> ports;
  std::vector<std::string> lookup_port_wire_names;
  std::vector<std::string> update_port_wire_names;
  for(int i =0; i < lookups.size(); i ++){
    auto i_str = std::to_string(i);
    auto port_wire_name = "lookup_p_" +i_str;
    lookup_port_wire_names.push_back(port_wire_name);
    // emit wires def for port
    struct wire_config port_wires = {TABLE_LOOKUP, port_wire_name, "Table lookup port wire def ", false, -1, true, .table_if=table_if};
    emitwire(file, port_wires);
    tableops_to_portwire[lookups[i]] = port_wires;
  }
  
  for(int i =0; i < updates.size(); i ++){
    auto i_str = std::to_string(i);
    auto port_wire_name = "update_p_" +i_str;
    update_port_wire_names.push_back(port_wire_name);
    // emit wires def for port
    struct wire_config port_wires = {TABLE_UPDATE, port_wire_name, "Table update port wire def ", false, -1, true, .table_if=table_if};
    emitwire(file, port_wires);
    tableops_to_portwire[updates[i]] = port_wires;
  }

  struct module_port_config lport = {TABLE_LOOKUP, {lookup_port_wire_names}, "lookup port ", "s_lookup", .table_if = table_if};
  ports.push_back(lport);
  struct module_port_config uport = {TABLE_UPDATE, {update_port_wire_names}, "update port ", "s_update", .table_if = table_if};
  ports.push_back(uport);

  int min_lports = lookups.size() > 0 ? lookups.size() : 1;
  int min_uports = updates.size() > 0 ? updates.size() : 1;

  std::list<struct module_param_config> params;
  params.push_back({"TABLE_SIZE", table_size});
  params.push_back({"KEY_SIZE", key_size});
  params.push_back({"VALUE_SIZE", value_size});
  params.push_back({"LOOKUP_PORTS", min_lports});
  params.push_back({"UPDATE_PORTS", min_uports});

  emitModuleCall(file, "cam_arbiter", tabel_name, ports, params);
}

void EmitFPGAPass::emitLookup(std::ofstream &file, ep2::LookupOp lookupop) {
  auto table_port_wire = tableops_to_portwire[lookupop];
  auto key_val = lookupop.getKey();
  int size;
  auto key_val_type = GetValTypeAndSize(key_val.getType(), &size);
  if (!(key_val_type == INT || key_val_type == STRUCT)) {
    printf("Error: emitLookup's key can only be int or struct\n");
    lookupop.dump();
    key_val.dump();
    assert(false);
  }
  auto key_val_name = getValName(key_val);
  struct axis_config key_axis =  {0, 0, size};
  struct wire_config key_wire = {AXIS, key_val_name, "", false, -1, true, key_axis};
  struct wire_assign_config key_assign = {-1, -1, -1, -1, key_wire, table_port_wire};
  emitwireassign(file, key_assign);

  // lookup results:
  auto value_val = lookupop.getValue();
  auto value_val_type = GetValTypeAndSize(value_val.getType(), &size);
  if (!(value_val_type == INT || value_val_type == STRUCT)) {
    printf("Error: emitLookup's value can only be int or struct\n");
    lookupop.dump();
    value_val.dump();
    assert(false);
  }

  auto value_name = assign_var_name("lookedup_" + val_type_str(value_val_type));
  UpdateValName(value_val, value_name);
  struct axis_config value_axis = {0, 0, size};
  struct wire_config value_wire = {AXIS,  value_name, "table lookup result" + value_name,
                  false, -1, has_use(value_val), value_axis};
  struct wire_assign_config value_assign = {-1, -1, -1, -1, table_port_wire, value_wire};
  emitwire(file, value_wire);
  emitwireassign(file, value_assign);
}

void EmitFPGAPass::emitUpdate(std::ofstream &file, ep2::UpdateOp updateop) {
  auto table_port_wire = tableops_to_portwire[updateop];
  auto key_val = updateop.getKey();
  int size;
  auto key_val_type = GetValTypeAndSize(key_val.getType(), &size);
  if (!(key_val_type == INT || key_val_type == STRUCT)) {
    printf("Error: emitUpdate's key can only be int or struct\n");
    updateop.dump();
    key_val.dump();
    assert(false);
  }
  auto key_val_name = getValName(key_val);
  struct axis_config key_axis =  {0, 0, size};
  struct wire_config key_wire = {AXIS, key_val_name, "", false, -1, true, key_axis};
  struct wire_assign_config key_assign = {-1, -1, -1, -1, key_wire, table_port_wire};
  emitwireassign(file, key_assign);

  // udpate value:
  auto value_val = updateop.getValue();
  auto value_val_type = GetValTypeAndSize(value_val.getType(), &size);
  if (!(value_val_type == INT || value_val_type == STRUCT)) {
    printf("Error: emitUpdate's value can only be int or struct\n");
    updateop.dump();
    value_val.dump();
    assert(false);
  }

  auto value_val_name = getValName(value_val);
  struct axis_config value_axis =  {0, 0, size};
  struct wire_config value_wire = {AXIS, value_val_name, "", false, -1, true, value_axis};
  struct wire_assign_config value_assign = {-1, -1, -1, -1, table_port_wire, value_wire};
  emitwireassign(file, value_assign);
}

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
                  false, -1, has_use(buf), out_buf_axis};

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
      false, -1, has_use(extracted_struct), out_struct_axis};

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
                  false, -1, has_use(buf), out_buf_axis};

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

  if(outval.getDefiningOp()->hasAttr("var_name"))
  {
    auto name = outval.getDefiningOp()->getAttr("var_name").cast<StringAttr>().getValue().str();
    printf("Error: emitStructAccess's output val already have a name %s\n", name.c_str());
    outval.dump();
    assert(false);
  }

  auto name = assign_var_name(debuginfo);
  // outval.getDefiningOp()->setAttr("var_name", builder->getStringAttr(name));
  UpdateValName(outval, name);
  outval_axis = {0, 0, size};
  outval_wire = {AXIS,       name, "Access Struct", false, -1, has_use(outval),
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
                     false, -1, false,           src_struct_axis};
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
                     false, -1, has_use(srcval), new_struct_axis};
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
                     false, -1, has_use(new_struct), new_struct_axis};

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

void EmitFPGAPass::emitArithmetic(std::ofstream &file,
                                     mlir::Operation *op) {
  mlir::Value result_val;
  mlir::Value lval, rval;
  int op_id;
  std::string op_name;
  if (isa<ep2::SubOp>(op)){
      auto subop = cast<ep2::SubOp, mlir::Operation *>(op);
      result_val = subop.getResult();
      lval = subop.getLhs();
      rval = subop.getRhs();
      op_id = 0;
      op_name = "SUB";
    }
  else if(isa<ep2::AddOp>(op)){
      auto addop = cast<ep2::AddOp, mlir::Operation *>(op);
      result_val = addop.getResult();
      lval = addop.getLhs();
      rval = addop.getRhs();
      op_id = 1;
      op_name = "ADD";
  }

  auto module_name = assign_var_name(op_name);
  struct module_port_config lval_port, rval_port, result_val_port;
  struct axis_config lval_axis, rval_axis, result_val_axis;
  struct wire_config lval_wire, rval_wire, result_val_wire;
  int size;

  auto result_val_type = GetValTypeAndSize(result_val.getType(), &size);
  if (!(result_val_type == INT || result_val_type == STRUCT)) {
    printf("Error: Cannot emitArithmetic's output val\n");
    result_val.dump();
    assert(false);
  }

  assert(!result_val.getDefiningOp()->hasAttr("var_name"));
  auto name = assign_var_name(module_name + "_out_" + val_type_str(result_val_type));
  // outval.getDefiningOp()->setAttr("var_name", builder->getStringAttr(name));
  UpdateValName(result_val, name);
  result_val_axis = {0, 0, size};
  result_val_wire = {AXIS,   name, "Arithmetic OP Out", false, -1, has_use(result_val),
                 result_val_axis};
  result_val_port = {AXIS, {name}, "output val", "m_val_axis", result_val_axis};

  int lval_size, rval_size;

  auto lval_type = GetValTypeAndSize(lval.getType(), &lval_size);
  auto rval_type = GetValTypeAndSize(rval.getType(), &rval_size);
  if (!(lval_type == INT || lval_type == STRUCT)) {
    printf("Error: Cannot emitArithmetic's lval val\n");
    lval.dump();
    assert(false);
  }
  if (!(rval_type == INT || rval_type == STRUCT)) {
    printf("Error: Cannot emitArithmetic's rval val\n");
    rval.dump();
    assert(false);
  }
  lval_axis = {0, 0, lval_size};
  lval_wire = {AXIS,  getValName(lval), "Arithmetic OP lval",
                     false, -1, false,           lval_axis};
  lval_port = {AXIS,
                     {getValName(lval)},
                     "lval input",
                     "s_lval_axis",
                     lval_axis};
  rval_axis = {0, 0, rval_size};
  rval_wire = {AXIS,  getValName(rval), "Arithmetic OP rval",
                     false, -1, false,           rval_axis};
  rval_port = {AXIS,
                     {getValName(rval)},
                     "rval input",
                     "s_rval_axis",
                     rval_axis};

  std::list<struct module_port_config> ports;
  ports.push_back(lval_port);
  ports.push_back(rval_port);
  ports.push_back(result_val_port);

  emitwire(file, result_val_wire);

  std::list<struct module_param_config> params;
  params.push_back({"LVAL_SIZE", lval_size});
  params.push_back({"RVAL_SIZE", rval_size});
  params.push_back({"RESULT_SIZE", size});
  params.push_back({"OPID", op_id});

  emitModuleCall(file, "ALU", module_name, ports, params);
  // wire_assignment = {src_offset, src_size, -1, -1, src_struct_wire,
  // outval_wire}; emitwireassign(file, wire_assignment);
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

    if ((valtype == CONTEXT && size == 0) || valtype == ATOM) {
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
                  false, -1, true,           value_axis};

    auto dst_port_base_name = getValName(out_port);
    auto dst_port_name = dst_port_base_name + "_" + std::to_string(i);
    outports_axis = value_axis;
    outports_wire = {AXIS,  dst_port_name, "Outport Assign Dst Port",
                     false, -1,  true,          outports_axis};
    wire_assignment = {-1, -1, -1, -1, value_wire, outports_wire};
    emitwireassign(file, wire_assignment);
  }
}

void EmitFPGAPass::emitConst(std::ofstream &file, ep2::ConstantOp constop) {
  auto arg = constop.getResult();
  auto arg_type = arg.getType();
  auto init_val = constop.getValue();
  bool if_stream;
  int size;
  std::string name, debuginfo;

  auto valtype = GetValTypeAndSize(arg_type, &size);
  debuginfo = "const_" + val_type_str(valtype);
  if (valtype == INT || valtype == STRUCT) {
    if_stream = false;
  } else if(valtype == ATOM){
    return;
  }
  else {
    printf("Error: Cannot emitConst\n");
    constop.dump();
    assert(false);
  }

  long init_value = 0;
  if (auto intAttr = init_val.dyn_cast<mlir::IntegerAttr>()) {
    init_value = intAttr.getInt();
  } else{
    printf("cannot find attr for ConstantOP's init_val");
    init_val.dump();
    // assert(false);
    // TODO
  }
  assert(!arg.getDefiningOp()->hasAttr("var_name"));
  name = assign_var_name(debuginfo);
  UpdateValName(arg, name);

  if (!if_stream) {
    assert(size <= 64 * 8);
  } else {
    assert(size % 8 == 0);
  }
  struct axis_config axis = {if_stream, if_stream, size};
  struct wire_config wire = {AXIS, name, debuginfo, true, init_value, true, axis};
  emitwire(file, wire);
}

void EmitFPGAPass::emitIfElse(std::ofstream &file, scf::IfOp ifop){
      auto &if_region = ifop.getThenRegion();
      std::vector<mlir::Value> then_yields;
      std::vector<mlir::Value> else_yields;
      auto &else_region = ifop.getElseRegion();
      printf("///////////If region Start//////////////\n");
      if_region.walk([&](mlir::Operation *op) {
        if(isa<scf::YieldOp>(op)){
          auto yieldop = cast<scf::YieldOp, mlir::Operation *>(op);
          auto oprands = yieldop.getOperands()[0];
          oprands.dump();
          then_yields.push_back(oprands);
        }
      });
      printf("///////////If region End//////////////\n");

      file << "///////////Else region Start//////////////\n";
      else_region.walk([&](mlir::Operation *op) {
        if(isa<scf::YieldOp>(op)){
          auto yieldop = cast<scf::YieldOp, mlir::Operation *>(op);
          auto oprands = yieldop.getOperands()[0];
          else_yields.push_back(oprands);
          oprands.dump();
        }
      });
      printf("///////////Else region End//////////////\n");

      // generate demux slecetor
      auto if_cond = ifop.getCondition();
      const auto& if_results = ifop.getResults();

      assert(if_results.size() == then_yields.size());
      assert(if_results.size() == else_yields.size());
      int i = 0;
      for(auto result : if_results){
        auto if_val = then_yields[i];
        auto else_val = else_yields[i];
        struct module_port_config demux_result_port, in_if_port, in_else_port, condition_port;
        struct wire_config demux_result_wire;
        struct axis_config demux_result_axis, in_if_axis, in_else_axis, condition_axis;
        auto module_name = assign_var_name("ifelse_demux_" + std::to_string(i));

        // Type checking
        assert(if_val.getType() == else_val.getType());
        assert(if_val.getType() == result.getType());

        

        int size;
        bool if_stream;
        auto result_type = GetValTypeAndSize(result.getType(), &size);
        if_stream = if_axis_stream(result_type);
        if (result_type != INT && result_type != STRUCT && result_type != BUF)  {
          printf("Error: Cannot generate ifelse result_type wire for\n");
          result.dump();
          assert(false);
        }
        auto result_name = assign_var_name("ifelse_result" + val_type_str(result_type));
        auto debuginfo = "ifelse result" + std::to_string(i);
        UpdateValName(result, result_name);

        demux_result_axis = {if_stream, if_stream, size};
        demux_result_wire = {AXIS,  result_name, debuginfo, false, -1, has_use(result), demux_result_axis};
        demux_result_port = {AXIS, {result_name}, "if else selected val", "m_val_axis", demux_result_axis};
        emitwire(file, demux_result_wire);

        auto in_if_name = getValName(if_val);
        in_if_axis = {if_stream, if_stream, size};
        in_if_port = {AXIS, {in_if_name}, "if input val", "s_if_axis", in_if_axis};

        auto in_else_name = getValName(else_val);
        in_else_axis = {if_stream, if_stream, size};
        in_else_port = {AXIS, {in_else_name}, "else input val", "s_else_axis", in_else_axis};

        int cond_size;
        auto condition_name = getValName(if_cond);
        auto condition_type = GetValTypeAndSize(if_cond.getType(), &cond_size);
        if (condition_type != INT && condition_type != STRUCT)  {
          printf("Error: Condition Mast be INT or STRUCT\n");
          if_cond.dump();
          assert(false);
        }
        condition_axis = {0, 0, cond_size};
        condition_port = {AXIS, {condition_name}, "if condition", "s_cond_axis", condition_axis};

        std::list<struct module_port_config> ports;
        ports.push_back(condition_port);
        ports.push_back(in_if_port);
        ports.push_back(in_else_port);
        ports.push_back(demux_result_port);

        std::list<struct module_param_config> params;
        params.push_back({"VAL_WIDTH", size});
        params.push_back({"COND_WIDTH", cond_size});
        params.push_back({"IF_VAL_BUF", if_stream});

        emitModuleCall(file, "ifelse_demux", module_name, ports, params);

        i ++;
      }
      



      // y.walk()
}

void EmitFPGAPass::emitOp(std::ofstream &file, mlir::Operation *op){
  if (isa<ep2::InitOp>(op)) {
    auto initop = cast<ep2::InitOp, mlir::Operation *>(op);
    if(isa<ep2::TableType>(initop.getResult().getType()))
      emitTableInit(file, initop);
    else if (initop.getArgs().size() == 0)
      emitVariableInit(file, initop);
    else {
      // Otherwise this init output event for this hanlder;
    }
  } else if (isa<ep2::ExtractOp>(op)) {
    auto extractop = cast<ep2::ExtractOp, mlir::Operation *>(op);
    emitExtract(file, extractop);
  } else if (isa<ep2::StructAccessOp>(op)) {
    auto structaccessop = cast<ep2::StructAccessOp, mlir::Operation *>(op);
    emitStructAccess(file, structaccessop);
  } else if (isa<ep2::StructUpdateOp>(op)) {
    auto structupdateop = cast<ep2::StructUpdateOp, mlir::Operation *>(op);
    emitStructUpdate(file, structupdateop);
  } else if (isa<ep2::EmitOp>(op)) {
    auto emitop = cast<ep2::EmitOp, mlir::Operation *>(op);
    emitEmit(file, emitop);
  } else if (isa<ep2::ReturnOp>(op)) {
    auto returnop = cast<ep2::ReturnOp, mlir::Operation *>(op);
    emitReturn(file, returnop);
  } else if (isa<ep2::ConstantOp>(op)){
    auto constop = cast<ep2::ConstantOp, mlir::Operation *>(op);
    emitConst(file, constop);
  } else if (isa<ep2::LookupOp>(op)){
    auto lookupop = cast<ep2::LookupOp, mlir::Operation *>(op);
    emitLookup(file, lookupop);
  } else if (isa<ep2::UpdateOp>(op)){
    auto updateop = cast<ep2::UpdateOp, mlir::Operation *>(op);
    emitUpdate(file, updateop);
  } else if(isa<ep2::SubOp>(op) || isa<ep2::AddOp>(op)){
    emitArithmetic(file, op);
  } else if (isa<scf::IfOp>(op)){
    auto ifop = cast<scf::IfOp, mlir::Operation *>(op);
    emitIfElse(file, ifop);
  }
  // TODO: Change STURCT ACCESS IR to generate new stream for each accessed
    // struct
}

void EmitFPGAPass::emitHandler(ep2::FuncOp funcOp) {
  cur_funcop = &funcOp;
  auto handler_name = funcOp.getName().str();
  std::ofstream fout_stage(handler_name + ".sv");
  emitFuncHeader(fout_stage, funcOp);

  funcOp->walk([&](mlir::Operation *op) {
    emitOp(fout_stage, op);
    
  });
  fout_stage << "\nendmodule\n";
}

void EmitFPGAPass::emitController(ep2::FuncOp funcOp) {
  cur_funcop = &funcOp;
  auto handler_name = funcOp.getName().str();
  std::ofstream file(handler_name + ".sv");
  file << "module " << handler_name << "#()\n";

  // from m replicas to n replicas
  std::vector<struct inout_config> inout_wires;
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
        {AXIS, portname, debuginfo + "wire", false, -1, true, wire});
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
        {AXIS, portname, debuginfo + "wire", false, -1, true, wire});
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
                    false, -1, true,         event_axis};
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

  contextbufferAnalysis = &(getAnalysis<ContextBufferizationAnalysis>());
  handlerInOutAnalysis = &(getAnalysis<HandlerInOutAnalysis>());
  tableAnalysis = &(getAnalysis<TableAnalysis>());
  handlerDependencyAnalysis = &(getAnalysis<HandlerDependencyAnalysis>());
  module->walk([&](ep2::FuncOp funcOp) {
    auto functype = funcOp->getAttr("type").cast<StringAttr>().getValue().str();
    std::cout << functype << "\n";
    if (functype == "handler") {
      emitHandler(funcOp);
    } else if (functype == "controller") {
      emitController(funcOp);
    }
  });


  emitTop();
}

std::unique_ptr<Pass> createEmitFPGAPass() {
  return std::make_unique<EmitFPGAPass>();
}

} // namespace ep2
} // namespace mlir
