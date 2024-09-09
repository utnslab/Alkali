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
  std::string in_event_name = funcOp->getAttr("event").cast<StringAttr>().getValue().str();
  bool if_in_extern_event = false;
  if (!handlerDependencyAnalysis->hasPredecessor(funcOp)){
    if_in_extern_event = true;
  }

  std::vector<std::pair<struct wire_config, int>> replica_args;
  int index = 0;
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

    if(if_in_extern_event) // extern event's val name is determined by its unique name
      name = assignValNameAndUpdate(arg, getExternArgName(in_event_name, index), false);
    else
      name = assignValNameAndUpdate(arg, "arg");
    if (!if_stream) {
      assert(size <= 64 * 8);
    } else {
      assert(size % 8 == 0);
    }
    struct axis_config wire = {if_stream, if_stream, size};
    struct wire_config event_wire = {AXIS, name, "", false, "", true, wire};
    struct inout_config in_if = {IN, AXIS, name, debuginfo, wire};
    inout_wires.push_back(in_if);
    in_event_wires.push_back(event_wire);
    
    if(if_in_extern_event)
      extern_inouts[name] = in_if;

    // save temp info for replication
    if(val_use_count(arg) > 1){
      event_wire.if_use = has_use(arg);
      replica_args.push_back({event_wire, val_use_count(arg)});
    }
    index ++;
  }
  llvm::errs() << "Info: " << funcOp.getName() << " has " << replica_args.size() << " replicated args\n";
  handler_in_edge_map[*cur_funcop].push_back({IN, in_event_name, if_in_extern_event, in_event_wires});

  // push output parameter wires
  auto returnvals = handlerInOutAnalysis->handler_returnop_list[funcOp];
  int out_event_id = 0;
  for (auto returned_event : returnvals) {
    // log event wire info, used in generating top module
    std::vector<struct wire_config> out_event_wires;
    std::string name;

    auto returned_event_type = returned_event.getType();

    assert(isa<ep2::StructType>(returned_event_type));
    auto return_event_struct = cast<ep2::StructType, Type>(returned_event_type);
    auto return_event_name = return_event_struct.getName().str();
    auto if_out_extern_event = false;
    if(!handlerDependencyAnalysis->hasSuccessor(return_event_name)){
      if_out_extern_event = true;
      // ReturnOp returnOp = dyn_cast<ReturnOp>(returned_event.getDefiningOp());
      // if(handlerDependencyAnalysis->lookupController(returnOp)) {
      //   if_out_extern_event = false;
      // }
      // auto next_extern_handler = handlerDependencyAnalysis->getSuccessors(funcOp, true);
      // if(next_extern_handler.size() > 0){ // has an extern event
      //   auto extern_handler = next_extern_handler.front();
      //   if(controllerAnalysis->target_handler_to_ctrl_func.find(extern_handler) != controllerAnalysis->target_handler_to_ctrl_func.end())
      //     if_out_extern_event = false;
      // }
    }
    auto field_types = return_event_struct.getElementTypes();

    
    if(if_out_extern_event) // extern event's val name is determined by its unique name
      name = assignValNameAndUpdate(returned_event, return_event_name, false);
    else
      name = assignValNameAndUpdate(returned_event, "outport");

    int index = 0;
    for (int i = 0; i < field_types.size(); i++) {
      int size;
      std::string portname;

      auto valtype = GetValTypeAndSize(field_types[i], &size);
      bool if_stream = if_axis_stream(valtype);

      if ((valtype == CONTEXT && size == 0) || valtype == ATOM) {
        continue;
      } else if (!(valtype == CONTEXT || valtype == INT || valtype == STRUCT || valtype == BUF)) {
        printf("Error: Cannot generate in parameter wire for\n");
        field_types[i].dump();
        assert(false);
      }

    
      portname = name + "_" + std::to_string(index);
      if(if_out_extern_event) // extern event's val name is determined by its unique name
        portname = getExternArgName(return_event_name, index);

      struct axis_config wire = {if_stream, if_stream, size};
      struct wire_config event_wire = {AXIS, portname, "", false, "", true, wire};
      struct inout_config out_if = {OUT, AXIS, portname, "output ports " + val_type_str(valtype), wire};
      inout_wires.push_back(out_if);
      out_event_wires.push_back(event_wire);

      if(if_out_extern_event)
        extern_inouts[portname] = out_if;
      index ++;
    }
    handler_out_edge_map[*cur_funcop].push_back({OUT, return_event_name, if_out_extern_event, out_event_wires});
  }

  // Find all global variable in and out
  funcOp->walk([&](GlobalImportOp importop) {
    bool if_has_lookup =false, if_has_update=false;
    mlir::SmallVector<mlir::Operation*> tmp_lookup_v;
    mlir::SmallVector<mlir::Operation*> tmp_update_v;
    for(auto users: importop->getUsers()){
      if(isa<ep2::LookupOp>(users)){
        if_has_lookup = true;
        tmp_lookup_v.push_back(users);
      }
      else if(isa<ep2::UpdateOp>(users)){
        if_has_update = true;
        tmp_update_v.push_back(users);
      }
      else if(isa<ep2::UpdateAtomicOp>(users)){
        if_has_update = true;
        tmp_update_v.push_back(users);
      }
    }

    auto tabel_name = importop.getName().str();
    auto lookup_wires = global_tables[tabel_name].lookup_wires;
    auto update_wires = global_tables[tabel_name].update_wires;

    if(if_has_lookup){
      int i = 0;
      for(auto &w : lookup_wires){
        printf("Info: lookup_wires %s\n", importop.getName().str().c_str());
        inout_wires.push_back({IN, w.type, w.name, w.debuginfo, w.axis});
        global_state_ports[*cur_funcop].push_back({w.type, {w.name}, w.debuginfo, w.name, .table_if = w.table_if});
        tableops_to_portwire[tmp_lookup_v[i]] = w;
        i++;
      }
    }
    if (if_has_update){
      int i = 0;
      for(auto &w : update_wires){
        printf("Info: update_wires %s\n", importop.getName().str().c_str());
        inout_wires.push_back({IN, w.type, w.name, w.debuginfo, w.axis});
        global_state_ports[*cur_funcop].push_back({w.type, {w.name}, w.debuginfo, w.name, .table_if = w.table_if});
        tableops_to_portwire[tmp_update_v[i]] = w;
        i++;
      }
    }
  });

  emitModuleParameter(file, inout_wires);

  // emit replication for arg input wires
  for(auto &arg : replica_args){
    emitwire(file, arg.first, arg.second, false);
  }
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

  name = assignValNameAndUpdate(arg, debuginfo);

  if (!if_stream) {
    assert(size <= 64 * 8);
  } else {
    assert(size % 8 == 0);
  }
  struct axis_config axis = {if_stream, if_stream, size};
  struct wire_config wire = {AXIS, name, debuginfo, true, "0", true, axis};
  emitwire(file, wire, val_use_count(arg));
}


void EmitFPGAPass::emitTableInit(std::ofstream &file, ep2::InitOp initop) {
  auto table = cast<ep2::TableType, Type>(initop.getType());
  auto tabel_name = assign_name("table");
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
    auto port_wire_name = tabel_name + "_lookup_p_" +i_str;
    lookup_port_wire_names.push_back(port_wire_name);
    // emit wires def for port
    struct wire_config port_wires = {TABLE_LOOKUP, port_wire_name, "Table lookup port wire def ", false, "", true, .table_if=table_if};
    emitwire(file, port_wires);
    tableops_to_portwire[lookups[i]] = port_wires;
  }
  
  for(int i =0; i < updates.size(); i ++){
    auto i_str = std::to_string(i);
    auto port_wire_name = tabel_name + "_update_p_" +i_str;
    update_port_wire_names.push_back(port_wire_name);
    // emit wires def for port
    struct wire_config port_wires = {TABLE_UPDATE, port_wire_name, "Table update port wire def ", false, "", true, .table_if=table_if};
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


void EmitFPGAPass::emitGlobalTableInit(ep2::GlobalOp globalop) {
  globalop.getResult().getType().dump();
  auto table = cast<ep2::TableType, Type>(globalop.getResult().getType());
  auto tabel_name = assign_name(globalop.getName().str());
  // TODO: need to write a analysis pass to get all lookup uses
  auto lookups = 1;
  auto updates = 1; 

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
  std::list<struct wire_config> lookup_wires, update_wires;
  for(int i =0; i < lookups; i ++){
    auto i_str = std::to_string(i);
    auto port_wire_name = tabel_name + "lookup_p_" +i_str;
    lookup_port_wire_names.push_back(port_wire_name);
    // emit wires def for port
    struct wire_config port_wires = {TABLE_LOOKUP, port_wire_name, "Table lookup port wire def ", false, "", true, .table_if=table_if};
    lookup_wires.push_back(port_wires);
    // emitwire(file, port_wires);
    // tableops_to_portwire[lookups[i]] = port_wires;
  }
  
  for(int i =0; i < updates; i ++){
    auto i_str = std::to_string(i);
    auto port_wire_name = tabel_name + "update_p_" +i_str;
    update_port_wire_names.push_back(port_wire_name);
    // emit wires def for port
    struct wire_config port_wires = {TABLE_UPDATE, port_wire_name, "Table update port wire def ", false, "", true, .table_if=table_if};
    update_wires.push_back(port_wires);
    // emitwire(file, port_wires);
    // tableops_to_portwire[updates[i]] = port_wires;
  }

  struct module_port_config lport = {TABLE_LOOKUP, {lookup_port_wire_names}, "lookup port ", "s_lookup", .table_if = table_if};
  ports.push_back(lport);
  struct module_port_config uport = {TABLE_UPDATE, {update_port_wire_names}, "update port ", "s_update", .table_if = table_if};
  ports.push_back(uport);

  int min_lports = lookups > 0 ? lookups : 1;
  int min_uports = updates > 0 ? updates : 1;

  std::list<struct module_param_config> params;
  // params.push_back({"TABLE_SIZE", table_size});
  // params.push_back({"KEY_SIZE", key_size});
  // params.push_back({"VALUE_SIZE", value_size});
  // params.push_back({"LOOKUP_PORTS", min_lports});
  // params.push_back({"UPDATE_PORTS", min_uports});

  global_tables[globalop.getName().str()] = { "atom_CAM", tabel_name, ports, params, lookup_wires, update_wires};
  // emitModuleCall(file, "cam_arbiter", tabel_name, ports, params);
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
  struct wire_config key_wire = {AXIS, key_val_name, "", false, "", true, key_axis};
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

  auto value_name = assignValNameAndUpdate(value_val, "lookedup_" + val_type_str(value_val_type));
  struct axis_config value_axis = {0, 0, size};
  struct wire_config value_wire = {AXIS,  value_name, "table lookup result" + value_name,
                  false, "", has_use(value_val), value_axis};
  struct wire_assign_config value_assign = {-1, -1, -1, -1, table_port_wire, value_wire};
  emitwire(file, value_wire, val_use_count(value_val));
  emitwireassign(file, value_assign);
}

void EmitFPGAPass::emitUpdate(std::ofstream &file, ep2::UpdateOp updateop, bool if_guarded, ep2::GuardOp gop) {
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
  struct wire_config key_wire = {AXIS, key_val_name, "", false, "", true, key_axis};

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
  struct wire_config value_wire = {AXIS, value_val_name, "", false, "", true, value_axis};

  if(if_guarded){
    auto pred_wires = emitGuardPredModule(file, gop, 2); // for key and value
    key_wire = emitGuardModule(file, key_wire, pred_wires[0]);
    value_wire = emitGuardModule(file, value_wire, pred_wires[1]);
  }

  struct wire_assign_config key_assign = {-1, -1, -1, -1, key_wire, table_port_wire};
  emitwireassign(file, key_assign);
  struct wire_assign_config value_assign = {-1, -1, -1, -1, table_port_wire, value_wire};
  emitwireassign(file, value_assign);
}


void EmitFPGAPass::emitUpdateAtomic(std::ofstream &file, ep2::UpdateAtomicOp updateop, bool if_guarded, ep2::GuardOp gop) {
  assert(updateop->getAttr("opValue").cast<IntegerAttr>().getValue() == 1);
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
  struct wire_config key_wire = {AXIS, key_val_name, "", false, "", true, key_axis};

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
  struct wire_config value_wire = {AXIS, value_val_name, "", false, "", true, value_axis};

  if(if_guarded){
    auto pred_wires = emitGuardPredModule(file, gop, 2); // for key and value
    key_wire = emitGuardModule(file, key_wire, pred_wires[0]);
    value_wire = emitGuardModule(file, value_wire, pred_wires[1]);
  }

  struct wire_assign_config key_assign = {-1, -1, -1, -1, key_wire, table_port_wire};
  emitwireassign(file, key_assign);
  struct wire_assign_config value_assign = {-1, -1, -1, -1, table_port_wire, value_wire};
  emitwireassign(file, value_assign);
}

void EmitFPGAPass::emitExtract(std::ofstream &file, ep2::ExtractValueOp extractop) {
  // First emit the wire define for: output buf, output struct
  // Then emit the extract module call
  struct module_port_config in_buf_port, out_buf_port, out_struct_port;
  struct wire_config out_buf_wire, out_struct_wire;
  struct axis_config in_buf_axis, out_buf_axis, out_struct_axis;
  auto module_name = assign_name("extract_module");

  auto buf = extractop.getBuffer();
  std::string ori_buf_name = getValName(buf);
  in_buf_axis = {1, 1, DEFAULT_AXIS_STREAM_SIZE};
  in_buf_port = {
      AXIS, {ori_buf_name}, "input buf", "s_inbuf_axis", in_buf_axis};

  auto new_buf = extractop.getResult(0);
  auto new_buf_name = assignValNameAndUpdate(new_buf, "bufvar");
  int new_buf_size = 0;
  auto bufvaltype = GetValTypeAndSize(new_buf.getType(), &new_buf_size);
  if (!(bufvaltype == BUF)) {
    printf("Error: emitExtract's output buf is not buf type\n");
    new_buf.dump();
    assert(false);
  }
  out_buf_axis = {1, 1, DEFAULT_AXIS_STREAM_SIZE};
  out_buf_port = {
      AXIS, {new_buf_name}, "output buf", "m_outbuf_axis", out_buf_axis};
  out_buf_wire = {AXIS,  new_buf_name, module_name + " output buf",
                  false, "", has_use(new_buf), out_buf_axis};

  auto extracted_struct = extractop.getResult(1);
  auto extracted_struct_name = assignValNameAndUpdate(extracted_struct, "structvar");
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
      false, "", has_use(extracted_struct), out_struct_axis};

  std::list<struct module_port_config> ports;
  ports.push_back(in_buf_port);
  ports.push_back(out_buf_port);
  ports.push_back(out_struct_port);

  emitwire(file, out_buf_wire, val_use_count(new_buf));
  emitwire(file, out_struct_wire, val_use_count(extracted_struct));

  std::list<struct module_param_config> params;
  params.push_back({"BUF_DATA_WIDTH", DEFAULT_AXIS_STREAM_SIZE});
  params.push_back({"BUF_KEEP_WIDTH", DEFAULT_AXIS_STREAM_SIZE / 8});
  params.push_back({"EXTRACTED_STRUCT_WIDTH", extracted_struct_size});

  emitModuleCall(file, "extract", module_name, ports, params);
}

void EmitFPGAPass::emitEmit(std::ofstream &file, ep2::EmitValueOp emitop) {
  // First emit the wire define for: output buf, output struct
  // Then emit the extract module call
  struct module_port_config in_buf_port, in_struct_port, out_buf_port;
  struct wire_config out_buf_wire;
  struct axis_config in_buf_axis, in_struct_axis, out_buf_axis;
  auto module_name = assign_name("emit_module");

  auto buf = emitop.getBuffer();
  std::string ori_buf_name = getValName(buf);
  in_buf_axis = {1, 1, DEFAULT_AXIS_STREAM_SIZE};
  in_buf_port = {
      AXIS, {ori_buf_name}, "input buf", "s_inbuf_axis", in_buf_axis};

  auto new_buf = emitop.getResult();
  auto new_buf_name = assignValNameAndUpdate(new_buf, "bufvar");
  int new_buf_size = 0;
  auto bufvaltype = GetValTypeAndSize( new_buf.getType(), &new_buf_size);
  if (!(bufvaltype == BUF)) {
    printf("Error: Cannot calculate emitEmit's output buf size\n");
    new_buf.dump();
    assert(false);
  }
  out_buf_axis = {1, 1, DEFAULT_AXIS_STREAM_SIZE};
  out_buf_port = {
      AXIS, {new_buf_name}, "output buf", "m_outbuf_axis", out_buf_axis};
  out_buf_wire = {AXIS,  new_buf_name, module_name + " output buf",
                  false, "", has_use(new_buf), out_buf_axis};

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

  emitwire(file, out_buf_wire, val_use_count(new_buf));

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
  auto module_name = assign_name("struct_access");
  struct module_port_config outval_port, src_struct_port;
  struct axis_config outval_axis, src_struct_axis;
  struct wire_config outval_wire, src_struct_wire;
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

  auto name = assignValNameAndUpdate(outval, debuginfo);
  outval_axis = {0, 0, size};
  outval_wire = {AXIS,       name, "Access Struct", false, "", has_use(outval),
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
                     false, "", false,           src_struct_axis};
  src_struct_port = {AXIS,
                     {src_struct_name},
                     "struct input",
                     "s_struct_axis",
                     src_struct_axis};

  std::list<struct module_port_config> ports;
  ports.push_back(src_struct_port);
  ports.push_back(outval_port);

  emitwire(file, outval_wire, val_use_count(outval));

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
  auto module_name = assign_name("struct_assign");

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
  auto new_struct_name = assignValNameAndUpdate(new_struct, "structvar");
  auto new_struct_type = new_struct.getType();
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
                     false, "", has_use(new_struct), new_struct_axis};

  std::list<struct module_port_config> ports;
  ports.push_back(ori_struct_port);
  ports.push_back(in_val_port);
  ports.push_back(new_struct_port);

  emitwire(file, new_struct_wire, val_use_count(new_struct));

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
  else if (isa<arith::SubIOp>(op)){
    auto subop = cast<arith::SubIOp, mlir::Operation *>(op);
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
  else if(isa<arith::AndIOp>(op)){
      auto andop = cast<arith::AndIOp, mlir::Operation *>(op);
      result_val = andop.getResult();
      lval = andop.getLhs();
      rval = andop.getRhs();
      op_id = 2;
      op_name = "AND";
  }  else if(isa<ep2::CmpOp>(op)){
      auto cmpop = cast<ep2::CmpOp, mlir::Operation *>(op);
      result_val = cmpop.getResult();
      lval = cmpop.getLhs();
      rval = cmpop.getRhs();
      int cmpop_id = cmpop.getPredicate();
      if(cmpop_id == 60){
        op_id = 3;
        op_name = "LT";
      }
      else if(cmpop_id == 62){
        op_id = 4;
        op_name = "GT";
      }
      else if(cmpop_id == 40){
        op_id = 5;
        op_name = "EQ";
      }
      else if(cmpop_id == 41){
        op_id = 6;
        op_name = "LE";
      }
      else if(cmpop_id == 42){
        op_id = 7;
        op_name = "GE";
      }
      else{
        printf("Error: Cannot emitArithmetic's cmpop\n");
        cmpop.dump();
        assert(false);
      }
  } 

  auto module_name = assign_name(op_name);
  struct module_port_config lval_port, rval_port, result_val_port;
  struct axis_config lval_axis, rval_axis, result_val_axis;
  struct wire_config lval_wire, rval_wire, result_val_wire;
  int size;

  auto result_val_type = GetValTypeAndSize(result_val.getType(), &size);
  if (!(result_val_type == INT )) {
    printf("Error: Cannot emitArithmetic's output val\n");
    result_val.dump();
    assert(false);
  }

  auto name = assignValNameAndUpdate(result_val, module_name + "_out_" + val_type_str(result_val_type));
  result_val_axis = {0, 0, size};
  result_val_wire = {AXIS,   name, "Arithmetic OP Out", false, "", has_use(result_val),
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
  auto lval_name = getValName(lval);
  lval_axis = {0, 0, lval_size};
  lval_wire = {AXIS, lval_name, "Arithmetic OP lval",
                     false, "", false,           lval_axis};
  lval_port = {AXIS,
                     {lval_name},
                     "lval input",
                     "s_lval_axis",
                     lval_axis};
  auto rval_name = getValName(rval);
  rval_axis = {0, 0, rval_size};
  rval_wire = {AXIS,  rval_name, "Arithmetic OP rval",
                     false, "", false,           rval_axis};
  rval_port = {AXIS,
                     {rval_name},
                     "rval input",
                     "s_rval_axis",
                     rval_axis};

  std::list<struct module_port_config> ports;
  ports.push_back(lval_port);
  ports.push_back(rval_port);
  ports.push_back(result_val_port);

  emitwire(file, result_val_wire, val_use_count(result_val));

  std::list<struct module_param_config> params;
  params.push_back({"LVAL_SIZE", lval_size});
  params.push_back({"RVAL_SIZE", rval_size});
  params.push_back({"RESULT_SIZE", size});
  params.push_back({"OPID", op_id});

  emitModuleCall(file, "ALU", module_name, ports, params);
  // wire_assignment = {src_offset, src_size, -1, -1, src_struct_wire,
  // outval_wire}; emitwireassign(file, wire_assignment);
}


std::vector<mlir::ep2::EmitFPGAPass::wire_config> EmitFPGAPass::emitGuardPredModule(std::ofstream &file, ep2::GuardOp gop, int repd_out_num){

  std::vector <std::string> in_cond_port_names, out_cond_port_names;
  std::vector <struct wire_config> out_cond_wires;
  struct axis_config cond_axis = {0, 0, 1};
  for(auto cond : gop.getPreds()){
    auto in_cond_name = getValName(cond);
    in_cond_port_names.push_back(in_cond_name);
  }

  for(int i = 0; i < repd_out_num; i ++){
    auto out_cond_name = assign_name("replicated_guard_cond");
    struct wire_config out_cond_wire = {AXIS,  out_cond_name, "replicated guard condition", false, "", true, cond_axis};
    out_cond_wires.push_back(out_cond_wire);
    out_cond_port_names.push_back(out_cond_name);
    emitwire(file, out_cond_wire, 1);
  }

  struct module_port_config in_cond_port, out_cond_port;
  in_cond_port = {AXIS, in_cond_port_names, "guard condition", "s_guard_cond", cond_axis};
  out_cond_port = {AXIS, out_cond_port_names, "replicated guard condition", "m_guard_cond", cond_axis};

  std::list<struct module_port_config> ports;
  ports.push_back(in_cond_port);
  ports.push_back(out_cond_port);

  auto pre_attr = gop.getPredAttrs();
  int ground_truth = 0;
  assert(pre_attr.size() < 32);
  for(auto attr : pre_attr){
    ground_truth = ground_truth << 1;
    ground_truth += attr.cast<BoolAttr>().getValue();
  }
  std::list<struct module_param_config> params;
  params.push_back({"COND_WIDTH", (int)in_cond_port_names.size()});
  params.push_back({"REPLICATED_OUT_NUM", repd_out_num});
  params.push_back({"GROUND_TRUTH", ground_truth}); // TODO: calculate ground truth

  emitModuleCall(file, "guard_pred", assign_name("guard_pred"), ports, params);
  return out_cond_wires;
}

mlir::ep2::EmitFPGAPass::wire_config EmitFPGAPass::emitGuardModule(std::ofstream &file, struct wire_config inputwire, wire_config predwire){
  auto outputwire = inputwire;
  outputwire.name = inputwire.name + "_guarded";
  outputwire.debuginfo = inputwire.debuginfo + "-- guarded";

  emitwire(file, outputwire, 1);
  auto module_name = assign_name("guard");
  struct module_port_config input_port, output_port, cond_port;
  input_port = {AXIS, {inputwire.name}, "input val", "s_guard_axis", inputwire.axis};
  output_port = {AXIS, {outputwire.name}, "output val", "m_guard_axis", outputwire.axis};
  cond_port = {AXIS, {predwire.name}, "guard condition", "s_guard_cond", {0, 0, 1}};
  
  
  std::list<struct module_port_config> ports;
  ports.push_back(input_port);
  ports.push_back(output_port);
  ports.push_back(cond_port);

  std::list<struct module_param_config> params;
  params.push_back({"DATA_WIDTH", inputwire.axis.data_width});
  params.push_back({"IF_STREAM", inputwire.axis.if_keep});

  emitModuleCall(file, "guard", module_name, ports, params);
  return outputwire;
}

void EmitFPGAPass::emitReturn(std::ofstream &file, ep2::ReturnOp returnop, bool if_guarded, ep2::GuardOp gop) {
  if (returnop.getNumOperands() == 0)
    return;

  assert(returnop.getNumOperands() == 1);
  // get output port struct
  auto out_port = returnop.getOperand(0);
  auto out_port_type = out_port.getType();
  auto dst_port_base_name = getValName(out_port);
  assert(isa<ep2::StructType>(out_port_type));
  auto out_port_struct = cast<ep2::StructType, Type>(out_port_type);
  auto out_port_fileds = out_port_struct.getElementTypes();

  // get val that is going to be connected with the output port
  auto op = returnop.getOperand(0).getDefiningOp();
  assert(isa<ep2::InitOp>(op));
  auto initop = cast<ep2::InitOp, mlir::Operation *>(op);
  auto initop_args = initop.getArgs();

  assert(initop_args.size() == out_port_fileds.size());

  std::vector<struct wire_config> guarded_cond_wires;
  if(if_guarded){
    int count = 0;
    for (int i = 0; i < initop_args.size(); i++) {
      int size;
      auto valtype = GetValTypeAndSize(initop_args[i].getType(), &size);
      if ((valtype == CONTEXT && size == 0) || valtype == ATOM) {
        continue;
      }
      count ++;
    }
    guarded_cond_wires = emitGuardPredModule(file, gop, count);
  }

  int pred_id = 0;
  int index = 0;
  for (int i = 0; i < initop_args.size(); i++) {
    // assign wires to output ports
    struct wire_assign_config wire_assignment;
    struct axis_config value_axis, outports_axis;
    struct wire_config value_wire, outports_wire;

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
                  false, "", true,           value_axis};

    if(if_guarded){
      value_wire = emitGuardModule(file, value_wire, guarded_cond_wires[pred_id]);
      pred_id ++;
    }
    auto dst_port_name = dst_port_base_name + "_" + std::to_string(index);
    outports_axis = value_axis;
    outports_wire = {AXIS,  dst_port_name, "Outport Assign Dst Port",
                     false, "",  true,          outports_axis};
    wire_assignment = {-1, -1, -1, -1, value_wire, outports_wire};
    emitwireassign(file, wire_assignment);
    index ++;
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
  name = assignValNameAndUpdate(arg, debuginfo);

  if (!if_stream) {
    assert(size <= 64 * 8);
  } else {
    assert(size % 8 == 0);
  }
  struct axis_config axis = {if_stream, if_stream, size};
  struct wire_config wire = {AXIS, name, debuginfo, true, std::to_string(init_value), true, axis};
  emitwire(file, wire, val_use_count(arg));
}

void EmitFPGAPass::emitBBCondBranch(std::ofstream &file, cf::CondBranchOp condbranchop){
  auto cond = condbranchop.getCondition();
  auto true_bb = condbranchop.getTrueDest();
  auto false_bb = condbranchop.getFalseDest();
  auto true_param_list = condbranchop.getTrueOperands();
  auto false_param_list = condbranchop.getFalseOperands();

  // cond value is not used, just assign 1 to tready
  auto cond_name = getValName(cond);
    file << " assign " << cond_name << "_tready"
         << " = 1" << ";\n";
  for(int i = 0; i < true_param_list.size(); i ++){
    auto true_param = true_param_list[i];
    auto bb_wire = getBBDemuxInputWire(true_bb, i);

    int size;
    auto true_param_type = GetValTypeAndSize(true_param.getType(), &size);
    auto if_stream = if_axis_stream(true_param_type);
    if (!(true_param_type == INT || true_param_type == STRUCT || true_param_type == BUF)) {
      printf("Error: emitBBCondBranch's true_param can only be int or struct or buf\n");
      true_param.dump();
      assert(false);
    }

    auto true_param_name = getValName(true_param);
    struct axis_config true_param_axis =  {if_stream, if_stream, size};
    struct wire_config true_param_wire = {AXIS, true_param_name, "", false, "", true, true_param_axis};
    struct wire_assign_config true_param_assign = {-1, -1, -1, -1, true_param_wire, bb_wire};
    emitwireassign(file, true_param_assign); 
  }

  for(int i = 0; i < false_param_list.size(); i ++){
    auto false_param = false_param_list[i];
    auto bb_wire = getBBDemuxInputWire(false_bb, i);

    int size;
    auto false_param_type = GetValTypeAndSize(false_param.getType(), &size);
    auto if_stream = if_axis_stream(false_param_type);
    if (!(false_param_type == INT || false_param_type == STRUCT || false_param_type == BUF)) {
      printf("Error: emitBBCondBranch's false_param can only be int or struct or buf\n");
      false_param.dump();
      assert(false);
    }

    auto false_param_name = getValName(false_param);
    struct axis_config false_param_axis =  {if_stream, if_stream, size};
    struct wire_config false_param_wire = {AXIS, false_param_name, "", false, "", true, false_param_axis};
    struct wire_assign_config false_param_assign = {-1, -1, -1, -1, false_param_wire, bb_wire};
    emitwireassign(file, false_param_assign); 
  }
}


void EmitFPGAPass::emitBBBranch(std::ofstream &file,cf::BranchOp branchop){
  auto true_bb = branchop.getDest();
  auto true_param_list = branchop.getOperands();

  for(int i = 0; i < true_param_list.size(); i ++){
    auto true_param = true_param_list[i];
    auto bb_wire = getBBDemuxInputWire(true_bb, i);

    int size;
    auto true_param_type = GetValTypeAndSize(true_param.getType(), &size);
    auto if_stream = if_axis_stream(true_param_type);
    if (!(true_param_type == INT || true_param_type == STRUCT || true_param_type == BUF)) {
      printf("Error: emitBBCondBranch's true_param can only be int or struct or buf\n");
      true_param.dump();
      assert(false);
    }

    auto true_param_name = getValName(true_param);
    struct axis_config true_param_axis =  {if_stream, if_stream, size};
    struct wire_config true_param_wire = {AXIS, true_param_name, "", false, "", true, true_param_axis};
    struct wire_assign_config true_param_assign = {-1, -1, -1, -1, true_param_wire, bb_wire};
    emitwireassign(file, true_param_assign); 
  }

}

void EmitFPGAPass::emitSelect(std::ofstream &file, arith::SelectOp selectop)
{
  auto cond = selectop.getCondition();
  auto true_val = selectop.getTrueValue();
  auto false_val = selectop.getFalseValue();
  auto result = selectop.getResult();

  assert(cond.getType().isInteger(1));
  assert(true_val.getType() == false_val.getType());
  assert(true_val.getType() == result.getType());

  auto cond_name = getValName(cond);
  auto true_val_name = getValName(true_val);
  auto false_val_name = getValName(false_val);

  auto result_name = assignValNameAndUpdate(result, "select_result");
  struct axis_config cond_axis, val_axis;
  struct wire_config result_wire;
  struct module_port_config cond_port, true_val_port, false_val_port, result_port;

  int val_size, cond_size;  
  auto cond_type = GetValTypeAndSize(cond.getType(), &cond_size);
  auto val_type = GetValTypeAndSize(true_val.getType(), &val_size);
  bool if_stream = if_axis_stream(val_type);

  if (!(val_type == INT || val_type == STRUCT )) {
    printf("Error: emitSelect's val can only be int or struct\n");
    selectop.dump();
    assert(false);
  }

  val_axis = {if_stream, if_stream, val_size};
  result_wire = {AXIS, result_name, "", false, "", has_use(result), val_axis};
  result_port = {AXIS, {result_name}, "select result", "m_val_axis", val_axis};
  emitwire(file, result_wire, val_use_count(result));

  cond_axis = {0, 0, cond_size};
  cond_port = {AXIS, {cond_name}, "select condition", "s_cond_axis", cond_axis};

  true_val_port = {AXIS, {true_val_name}, "select true val", "s_true_val_axis", val_axis};
  false_val_port = {AXIS, {false_val_name}, "select false val", "s_false_val_axis", val_axis};

  std::list<struct module_port_config> ports;
  ports.push_back(cond_port);
  ports.push_back(true_val_port);
  ports.push_back(false_val_port);
  ports.push_back(result_port);

  std::list<struct module_param_config> params;
  params.push_back({"VAL_WIDTH", val_size});
  params.push_back({"COND_WIDTH", cond_size});

  emitModuleCall(file, "select", assign_name("select"), ports, params);
}

void EmitFPGAPass::emitIfElse(std::ofstream &file, scf::IfOp ifop){
      auto &if_region = ifop.getThenRegion();
      std::vector<mlir::Value> then_yields;
      std::vector<mlir::Value> else_yields;
      auto &else_region = ifop.getElseRegion();
      if_region.walk([&](mlir::Operation *op) {
        if(isa<scf::YieldOp>(op)){
          auto yieldop = cast<scf::YieldOp, mlir::Operation *>(op);
          auto oprands = yieldop.getOperands()[0];
          then_yields.push_back(oprands);
        }
      });

      file << "///////////Else region Start//////////////\n";
      else_region.walk([&](mlir::Operation *op) {
        if(isa<scf::YieldOp>(op)){
          auto yieldop = cast<scf::YieldOp, mlir::Operation *>(op);
          auto oprands = yieldop.getOperands()[0];
          else_yields.push_back(oprands);
        }
      });

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
        auto module_name = assign_name("ifelse_demux_" + std::to_string(i));

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
        auto result_name = assignValNameAndUpdate(result, "ifelse_result" + val_type_str(result_type));
        auto debuginfo = "ifelse result" + std::to_string(i);

        demux_result_axis = {if_stream, if_stream, size};
        demux_result_wire = {AXIS,  result_name, debuginfo, false, "", has_use(result), demux_result_axis};
        demux_result_port = {AXIS, {result_name}, "if else selected val", "m_val_axis", demux_result_axis};
        emitwire(file, demux_result_wire, val_use_count(result));

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
        params.push_back({"IF_STREAM", if_stream});

        emitModuleCall(file, "ifelse_demux", module_name, ports, params);

        i ++;
      }
}

void EmitFPGAPass::emitGuard(std::ofstream &file, ep2::GuardOp guardop){
  auto gop = guardop.getGuardingOp();
   if (isa<ep2::ReturnOp>(gop)) {
    auto returnop = cast<ep2::ReturnOp, mlir::Operation *>(gop);
    emitReturn(file, returnop, true, guardop);
   } else if (isa<ep2::UpdateOp>(gop)) {
    auto updateop = cast<ep2::UpdateOp, mlir::Operation *>(gop);
    emitUpdate(file, updateop, true, guardop);
   } else if(isa<ep2::UpdateAtomicOp>(gop)){
    auto updateatomicop = cast<ep2::UpdateAtomicOp, mlir::Operation *>(gop);
    emitUpdateAtomic(file, updateatomicop, true, guardop);
   }
}


void EmitFPGAPass::emitGlobalImport(std::ofstream &file,  ep2::GlobalImportOp importop){
}

void  EmitFPGAPass::emitSink(std::ofstream &file, ep2::SinkOp sinkop){
  auto val = sinkop.getOperand(0);
  auto val_name = getValName(val);
  file << " assign " << val_name << "_tready"
         << " = 1" << ";\n";
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
  } else if (isa<ep2::ExtractValueOp>(op)) {
    auto extractop = cast<ep2::ExtractValueOp, mlir::Operation *>(op);
    emitExtract(file, extractop);
  } else if (isa<ep2::StructAccessOp>(op)) {
    auto structaccessop = cast<ep2::StructAccessOp, mlir::Operation *>(op);
    emitStructAccess(file, structaccessop);
  } else if (isa<ep2::StructUpdateOp>(op)) {
    auto structupdateop = cast<ep2::StructUpdateOp, mlir::Operation *>(op);
    emitStructUpdate(file, structupdateop);
  } else if (isa<ep2::EmitValueOp>(op)) {
    auto emitop = cast<ep2::EmitValueOp, mlir::Operation *>(op);
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
  } else if (isa<ep2::UpdateAtomicOp>(op)){
    auto updateop = cast<ep2::UpdateAtomicOp, mlir::Operation *>(op);
    emitUpdateAtomic(file, updateop);
  } 
  else if(isa<ep2::SubOp>(op) || isa<ep2::AddOp>(op) || isa<arith::SubIOp>(op) || isa<arith::AndIOp>(op) || isa<ep2::CmpOp>(op)){
    emitArithmetic(file, op);
  } else if (isa<scf::IfOp>(op)){
    auto ifop = cast<scf::IfOp, mlir::Operation *>(op);
    emitIfElse(file, ifop);
  } else if (isa<cf::CondBranchOp>(op)){
    auto condop = cast<cf::CondBranchOp, mlir::Operation *>(op);
    emitBBCondBranch(file, condop);
  } else if (isa<cf::BranchOp>(op)){
    auto branchop = cast<cf::BranchOp, mlir::Operation *>(op);
    emitBBBranch(file, branchop);
  } else if (isa<ep2::SinkOp>(op)){
    auto sinkop = cast<ep2::SinkOp, mlir::Operation *>(op);
    emitSink(file, sinkop);
  } else if(isa<ep2::BitCastOp>(op)){
    auto bitcastop = cast<ep2::BitCastOp, mlir::Operation *>(op);
    emitBitcast(file, bitcastop);
  } else if(isa<arith::SelectOp>(op)){
    auto selectop = cast<arith::SelectOp, mlir::Operation *>(op);
    emitSelect(file, selectop);
  } else if(isa<ep2::GuardOp>(op)){
    auto guardop = cast<ep2::GuardOp, mlir::Operation *>(op);
    emitGuard(file, guardop);
  } else if(isa<ep2::GlobalImportOp>(op)){
    auto globalimportop = cast<ep2::GlobalImportOp, mlir::Operation *>(op);
    emitGlobalImport(file, globalimportop);
  }
  // TODO: Change STURCT ACCESS IR to generate new stream for each accessed
    // struct
}

void EmitFPGAPass::emitBBInputDemux(std::ofstream &file, ep2::FuncOp funcOp){
  auto it = funcOp.getBody().begin();
  int bb_count = 1; // bb0 is jumpped
  for (it++; it != funcOp.getBody().end(); it++) {
    auto &block = *it;
    auto args = block.getArguments();

    // how many input mux ports for each arg
    int port_count = llvm::count_if(block.getPredecessors(), [](auto) { return true; });
    int arg_count = 0;
    int totoal_arg_count = args.size();
    std::vector<struct demux_inout_arg> demux_inout_args;
    struct axis_config pred_axis;
    struct wire_config pred_wire;
    std::vector<std::string> local_pred_out_names;
    for (auto &arg: args) {
      // for each arg, generate "port" number demux input wires
      std::vector<std::string> demux_in_wire_names;
      std::list<struct module_port_config> ports;
      int arg_size;
      bool if_stream;
      auto valtype = GetValTypeAndSize( arg.getType(), &arg_size);
      if_stream = if_axis_stream(valtype);
      if (valtype != INT && valtype != STRUCT)  {
        printf("Error: Cannot generate BBInputDemux's input wire, Note that we currently don't support buf as tlast constraint\n");
        arg.dump();
        assert(false);
      }

      // single bit cond valid put in axis user
      struct axis_config arg_axis = {if_stream, if_stream, arg_size};

      std::vector<struct wire_config> in_args_wires;

      std::string module_name = "arg_demux";
      if(arg_count == 0)
        module_name = "pred_demux";
        
      // generate demux input wires
      for(int i = 0; i < port_count; i ++){
        auto demux_in_wire_name = "BB" + std::to_string(bb_count) +  module_name + "_in" + std::to_string(i);
        struct wire_config in_arg_wire = {AXIS,  demux_in_wire_name, demux_in_wire_name, false, "", true, arg_axis};
        demux_in_wire_names.push_back(demux_in_wire_name);
        emitwire(file, in_arg_wire);
        in_args_wires.push_back(in_arg_wire);
      }
      struct module_port_config inport = {AXIS, {demux_in_wire_names}, "demux in port ", "s_demux_in", arg_axis};
      ports.push_back(inport);

      // generate demux output wire
      auto out_val_name = assignValNameAndUpdate(arg, "BB" + std::to_string(bb_count) + module_name + "_out" +  std::to_string(arg_count));
      struct wire_config out_val_wire = {AXIS,  out_val_name, out_val_name, false, "", has_use(arg), arg_axis};
      emitwire(file, out_val_wire, val_use_count(arg));

      if(arg_count == 0){
        // pred case: replicate pred wire based on the total arg count
        pred_axis = arg_axis;
        pred_wire = out_val_wire;
        
        if(totoal_arg_count > 1)
        {
          for(int p = 0; p < totoal_arg_count -1; p ++){
            struct wire_config local_pred_out_wire = out_val_wire;
            local_pred_out_wire.axis.data_width = port_count;
            local_pred_out_wire.name = out_val_wire.name +"_local_pred_" + std::to_string(p);
            emitwire(file, local_pred_out_wire, 1);
            local_pred_out_names.push_back(local_pred_out_wire.name);
          }
          struct module_port_config predout_to_demux = {AXIS, {local_pred_out_names}, "local pred out", "m_pred_out", pred_axis};
          ports.push_back(predout_to_demux);
        }
      }else {
        // arg case
        struct module_port_config predinport = {AXIS, {local_pred_out_names[arg_count-1]}, "pred in port ", "s_pred_in", pred_axis};
        ports.push_back(predinport);
      }
      
      struct module_port_config outport = {AXIS, {out_val_name}, "demux out port ", "m_demux_out", arg_axis};
      ports.push_back(outport);

      std::list<struct module_param_config> params;
      params.push_back({"VAL_WIDTH", arg_size});
      params.push_back({"PORT_COUNT", port_count});
      params.push_back({"IF_STREAM", if_stream});
      if(arg_count == 0 )
        params.push_back({"IF_LOCAL_PRED_OUT", totoal_arg_count > 1});
      
      emitModuleCall(file, module_name, "BB" + std::to_string(bb_count) + "demux_arg" + std::to_string(arg_count), ports, params);

      demux_inout_args.push_back({port_count, 0, in_args_wires, out_val_wire});
      arg_count ++;
    }
    bb_to_demux_inout[&block] = demux_inout_args;

    bb_count ++;
  }
}

void EmitFPGAPass::emitBitcast(std::ofstream &file, ep2::BitCastOp bitcastop) {
  auto src_val = bitcastop.getInput();
  auto dst_val = bitcastop.getOutput();
  auto src_type = src_val.getType();
  auto dst_type = dst_val.getType();
  int src_size, dst_size;
  auto src_val_type = GetValTypeAndSize(src_type, &src_size);
  auto dst_val_type = GetValTypeAndSize(dst_type, &dst_size);
  if (src_val_type != INT || dst_val_type != INT) {
    printf("Error: emitBitcast's src and dst can only be stream\n");
    bitcastop.dump();
    assert(false);
  }

  auto src_val_name = getValName(src_val);
  auto dst_val_name = assignValNameAndUpdate(dst_val, "bitcasted");
  struct axis_config src_axis = {0, 0, src_size};
  struct axis_config dst_axis = {0, 0, dst_size};
  struct wire_config src_wire = {AXIS,  src_val_name, "bitcast src", false, "", true, src_axis};
  struct wire_config dst_wire = {AXIS,  dst_val_name, "bitcast dst", false, "", has_use(dst_val), dst_axis};
  emitwire(file, dst_wire, val_use_count(dst_val));
  struct wire_assign_config wire_assignment = {-1, -1, -1, -1, src_wire, dst_wire};
  emitwireassign(file, wire_assignment);
}

void EmitFPGAPass::emitHandler(ep2::FuncOp funcOp) {
  cur_funcop = &funcOp;
  auto handler_name = funcOp.getName().str();
  std::ofstream fout_stage(handler_name + ".sv");
  emitFuncHeader(fout_stage, funcOp);
  emitBBInputDemux(fout_stage, funcOp);
  // funcOp->walk([&](mlir::Operation *op) {
  //   emitOp(fout_stage, op);
  // });
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (auto &op : block.getOperations()) {
      emitOp(fout_stage, &op);
    }
  }
  fout_stage << "\nendmodule\n";
}


void EmitFPGAPass::runOnOperation() {
  module = getOperation();
  OpBuilder builder_tmp(module->getContext());
  builder = &builder_tmp;

  contextbufferAnalysis = &(getAnalysis<ContextBufferizationAnalysis>());
  handlerInOutAnalysis = &(getAnalysis<HandlerInOutAnalysis>());
  tableAnalysis = &(getAnalysis<TableAnalysis>());
  handlerDependencyAnalysis = &(getAnalysis<HandlerDependencyAnalysis>());

    module->walk([&](ep2::GlobalOp globalOp) {
        emitGlobalTableInit( globalOp);
    });

  module->walk([&](ep2::FuncOp funcOp) {
    auto functype = funcOp->getAttr("type").cast<StringAttr>().getValue().str();
    std::cout << functype << "\n";
    if (functype == "handler" && !funcOp->hasAttr("extern")) {
      emitHandler(funcOp);
    } 
  });

  module->walk([&](ep2::FuncOp funcOp) {
    auto functype = funcOp->getAttr("type").cast<StringAttr>().getValue().str();
    std::cout << functype << "\n";
    if (functype == "controller") {
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
