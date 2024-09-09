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

void EmitFPGAPass::emitModuleParameter(std::ofstream &file,
                                       std::vector<struct inout_config> &wires) {
  file << "(\n"
       << "\t input  wire clk, \n"
       << "\t input  wire rst";
  for (auto &wire : wires) {
    if (wire.type == AXIS) {
      file << ",\n";
      std::string inoutstr, reverse_inoutstr, tab;
      if (wire.direction == IN) {
        inoutstr = "input";
        reverse_inoutstr = "output";
      } else {
        inoutstr = "output";
        reverse_inoutstr = "input";
      }
      tab = "\t";
      std::string datawidthstr =
          "[" + std::to_string(wire.axis.data_width) + "-1:0]";
      std::string keepwidthstr =
          "[" + std::to_string(wire.axis.data_width / 8) + "-1:0]";

      file << tab << "//" << wire.debuginfo << "\n";
      file << tab << inoutstr << " wire " << datawidthstr << " " << wire.name
           << "_tdata ,\n";
      if (wire.axis.if_keep)
        file << tab << inoutstr << " wire " << keepwidthstr << " " << wire.name
             << "_tkeep ,\n";
      if (wire.axis.if_last)
        file << tab << inoutstr << " wire "
             << " " << wire.name << "_tlast ,\n";
      file << tab << inoutstr << " wire "
           << " " << wire.name << "_tvalid ,\n";
      file << tab << reverse_inoutstr << " wire "
           << " " << wire.name << "_tready";
    }

    if(wire.type == TABLE_LOOKUP){
      file << ",\n";
      std::string inoutstr, reverse_inoutstr, tab;
      assert(wire.direction == IN);
      tab = "\t";
      std::string datawidthstr =
        "[" + std::to_string(wire.table_if.data_width) + "-1:0]";
      std::string indexwidthstr =
          "[" + std::to_string(wire.table_if.index_width) + "-1:0]";
    
      file << tab << "//" << wire.debuginfo << "\n";
      file << tab << "output" << " wire " << indexwidthstr << " " << wire.name << "_req_index,\n";
      file << tab << "output" << " wire " << " " << wire.name << "_req_valid,\n";
      file << tab << "input" << " wire " << " " << wire.name << "_req_ready,\n";
      file << tab << "input" << " wire " << " " << wire.name << "_value_valid,\n";
      file << tab << "output" << " wire " << " " << wire.name << "_value_ready,\n";
      file << tab << "input" << " wire " << datawidthstr << " " << wire.name << "_value_data"; 
    }


  if(wire.type == TABLE_UPDATE){
    file << ",\n";
    std::string inoutstr, reverse_inoutstr, tab;
    assert(wire.direction == IN);
    tab = "\t";
    std::string datawidthstr =
        "[" + std::to_string(wire.table_if.data_width) + "-1:0]";
    std::string indexwidthstr =
        "[" + std::to_string(wire.table_if.index_width) + "-1:0]";
    
    file << tab << "//" << wire.debuginfo << "\n";
    file << tab <<  "output" << " wire " << indexwidthstr << " " << wire.name << "_req_index,\n";
    file << tab <<  "output" << " wire " << " " << wire.name << "_req_index_valid,\n";
    file << tab << "input" << " wire " << " " << wire.name << "_req_index_ready,\n";
    file << tab <<  "output" << " wire " << datawidthstr << " " << wire.name << "_req_data,\n";
    file << tab <<  "output" << " wire " << " " << wire.name << "_req_data_valid,\n";
    file << tab << "input" << " wire " << " " << wire.name << "_req_data_ready";
  }
  }

  file << "\n);\n";
}
void EmitFPGAPass::emitwire(std::ofstream &file, struct wire_config &wire, int replicas, bool if_emit_replica_src){
  if(replicas <= 1){
    emitonewire(file, wire);
    return;
  } else {
    std::list<struct module_port_config> ports;
    auto module_name = assign_name("axis_replication");

    if(if_emit_replica_src)
      emitonewire(file, wire);
    assert(wire.if_use == true);
    
    assert(wire.type == AXIS);
    struct module_port_config in_port_config, out_port_config;
    in_port_config = {wire.type, {wire.name}, "", "s_axis_in", .axis = wire.axis};
    std::vector<std::string> replicas_names;
    for(int i = 0; i < replicas; i++){
      struct wire_config new_wire = wire;
      new_wire.name = wire.name + "_" + std::to_string(i);
      // replicas don't have self drivend init value
      new_wire.if_init_value = false;
      emitonewire(file, new_wire);
      replicas_names.push_back(new_wire.name);
    }
    out_port_config = {wire.type, replicas_names, "", "m_axis_out", .axis = wire.axis};
    ports.push_back(in_port_config);
    ports.push_back(out_port_config);
    
    std::list<struct module_param_config> params;
    params.push_back({"DATA_WIDTH", wire.axis.data_width});
    params.push_back({"IF_STREAM", wire.axis.if_keep});
    params.push_back({"REAPLICA_COUNT", replicas});
    emitModuleCall(file, "axis_replication", module_name, ports, params);
  }
}

void EmitFPGAPass::emitonewire(std::ofstream &file, struct wire_config &wire) {
  if (wire.type == AXIS) {
    std::string inoutstr, reverse_inoutstr, tab;

    std::string datawidthstr =
        "[" + std::to_string(wire.axis.data_width) + "-1:0]";
    std::string keepwidthstr =
        "[" + std::to_string(wire.axis.data_width / 8) + "-1:0]";
    std::string initvalue = "";
    std::string initdatavalue = "";
    std::string readyvalue = "";
    if (wire.if_init_value) {
      initvalue = "=1";
      initdatavalue = "=" + wire.init_value;
    }
    if (!wire.if_use)
      readyvalue = "=1";

    file << "//" << wire.debuginfo << "\n";
    file << " wire " << datawidthstr << " " << wire.name << "_tdata"
         << initdatavalue << ";\n";
    if (wire.axis.if_keep)
      file << " wire " << keepwidthstr << " " << wire.name << "_tkeep"
           << initdatavalue << ";\n";
    file << " wire "
         << " " << wire.name << "_tvalid" << initvalue << ";\n";
    file << " wire "
         << " " << wire.name << "_tready" << readyvalue << ";\n";
    if (wire.axis.if_last)
      file << " wire "
           << " " << wire.name << "_tlast" << initvalue << ";\n";
  } else if(wire.type == TABLE_LOOKUP){
    std::string datawidthstr =
        "[" + std::to_string(wire.table_if.data_width) + "-1:0]";
    std::string indexwidthstr =
        "[" + std::to_string(wire.table_if.index_width) + "-1:0]";
    
    file << "//" << wire.debuginfo << "\n";
    file << " wire " << indexwidthstr << " " << wire.name << "_req_index"
         << ";\n";
    file << " wire " << " " << wire.name << "_req_valid"  << ";\n";
    file << " wire " << " " << wire.name << "_req_ready" << ";\n";
    file << " wire " << " " << wire.name << "_value_valid"  << ";\n";
    file << " wire " << " " << wire.name << "_value_ready"  << ";\n";
    file << " wire " << datawidthstr << " " << wire.name << "_value_data"
        << ";\n";
  } else if(wire.type == TABLE_UPDATE){
    std::string datawidthstr =
        "[" + std::to_string(wire.table_if.data_width) + "-1:0]";
    std::string indexwidthstr =
        "[" + std::to_string(wire.table_if.index_width) + "-1:0]";
    
    file << "//" << wire.debuginfo << "\n";
    file << " wire " << indexwidthstr << " " << wire.name << "_req_index"
         << ";\n";
    file << " wire " << " " << wire.name << "_req_index_valid"  << ";\n";
    file << " wire " << " " << wire.name << "_req_index_ready" << ";\n";
    
    file << " wire " << datawidthstr << " " << wire.name << "_req_data"
        << ";\n";
    file << " wire " << " " << wire.name << "_req_data_valid"  << ";\n";
    file << " wire " << " " << wire.name << "_req_data_ready" << ";\n";
  } else if (wire.type == BIT) {
    std::string initvalue = "";
    if (wire.if_init_value) {
      initvalue = "=" + wire.init_value;
    }

    file << "//" << wire.debuginfo << "\n";
    std::string datawidthstr =
        "[" + std::to_string(wire.bit.size) + "-1:0]";
    file << " wire " << datawidthstr << wire.name << initvalue << ";\n";
  }
  file << "\n";
}

void EmitFPGAPass::emitwireassign(std::ofstream &file,
                                  struct wire_assign_config &assign) {
  if (assign.src_wire.type == AXIS && assign.dst_wire.type == AXIS) {
    std::string src_offset_str = "";
    std::string dst_offset_str = "";
    if (assign.src_wire_offset != -1)
      src_offset_str = "[" + std::to_string(assign.src_wire_offset) +
                       "+:" + std::to_string(assign.src_wire_size) + "]";

    if (assign.dst_wire_offset != -1)
      dst_offset_str = "[" + std::to_string(assign.dst_wire_offset) +
                       "+:" + std::to_string(assign.dst_wire_size) + "]";

    file << " assign " << assign.dst_wire.name << "_tdata" << dst_offset_str
         << " = " << assign.src_wire.name << "_tdata" << src_offset_str
         << ";\n";

    file << " assign " << assign.dst_wire.name << "_tvalid"
         << " = " << assign.src_wire.name << "_tvalid"
         << ";\n";
    file << " assign " << assign.src_wire.name << "_tready"
         << " = " << assign.dst_wire.name << "_tready"
         << ";\n";

    if (assign.src_wire.axis.if_keep || assign.src_wire.axis.if_last) {
      // Current not support have offset for stream;
      assert(assign.src_wire_offset == -1 && assign.dst_wire_offset == -1);
    }
    if (assign.src_wire.axis.if_keep)
      file << " assign " << assign.dst_wire.name << "_tkeep"
           << " = " << assign.src_wire.name << "_tkeep"
           << ";\n";
    if (assign.src_wire.axis.if_last)
      file << " assign " << assign.dst_wire.name << "_tlast"
           << " = " << assign.src_wire.name << "_tlast"
           << ";\n";
  } else if(assign.src_wire.type == AXIS && assign.dst_wire.type == TABLE_LOOKUP){
     file << " assign " << assign.dst_wire.name << "_req_index" 
         << " = " << assign.src_wire.name << "_tdata"
         << ";\n";
    file << " assign " << assign.dst_wire.name << "_req_valid"
         << " = " << assign.src_wire.name << "_tvalid"
         << ";\n";
    file << " assign " << assign.src_wire.name << "_tready"
         << " = " << assign.dst_wire.name << "_req_ready"
         << ";\n";
  } else if (assign.src_wire.type == TABLE_LOOKUP  && assign.dst_wire.type == AXIS){
     file << " assign " << assign.dst_wire.name << "_tdata" 
         << " = " << assign.src_wire.name << "_value_data"
         << ";\n";
    file << " assign " << assign.dst_wire.name << "_tvalid"
         << " = " << assign.src_wire.name << "_value_valid"
         << ";\n";
    file << " assign " << assign.src_wire.name << "_value_ready"
         << " = " << assign.dst_wire.name << "_tready"
         << ";\n";
   } else if(assign.src_wire.type == AXIS && assign.dst_wire.type == TABLE_UPDATE){
    // UPDATE key
     file << " assign " << assign.dst_wire.name << "_req_index" 
         << " = " << assign.src_wire.name << "_tdata"
         << ";\n";
    file << " assign " << assign.dst_wire.name << "_req_index_valid"
         << " = " << assign.src_wire.name << "_tvalid"
         << ";\n";
    file << " assign " << assign.src_wire.name << "_tready"
         << " = " << assign.dst_wire.name << "_req_index_ready"
         << ";\n";
  } else if (assign.src_wire.type == TABLE_UPDATE  && assign.dst_wire.type == AXIS){
    // UPDATE value
     file << " assign " << assign.src_wire.name << "_req_data" 
         << " = " << assign.dst_wire.name << "_tdata"
         << ";\n";
    file << " assign " << assign.src_wire.name << "_req_data_valid"
         << " = " << assign.dst_wire.name << "_tvalid"
         << ";\n";
    file << " assign " << assign.dst_wire.name << "_tready"
         << " = " << assign.src_wire.name << "_req_data_ready"
         << ";\n";
   } else if (assign.src_wire.type == BIT && assign.dst_wire.type == BIT) {
    file << " assign " << assign.dst_wire.name << " = " << assign.src_wire.name
         << ";\n";
  }
  file << "\n";
}

void EmitFPGAPass::emitModuleCall(
    std::ofstream &file, std::string module_type, std::string module_name,
    std::list<struct module_port_config> &ports,
    std::list<struct module_param_config> &params) {
  file << module_type << "#(\n";
  int param_num = params.size();
  int i = 0;
  std::string tab = "\t";
  for (auto &param : params) {
    i++;
    file << "." << param.paramname << "(" << std::to_string(param.paramval)
         << ")";
    if (i != param_num)
      file << ",";
    file << "\n";
  }

  file << ")" << module_name << "(\n"
       << "\t .clk(clk), \n"
       << "\t .rst(rst) ";
  for (auto &port : ports) {
    if (port.type == AXIS) {
      std::string tdata_var_names = "{";
      std::string tkeep_var_names = "{";
      std::string tlast_var_names = "{";
      std::string tvalid_var_names = "{";
      std::string tready_var_names = "{";
      // if multiple input vars, cocact them
      for (int i = 0; i < port.var_name.size(); i++) {
        auto optional_comma = (i == port.var_name.size() - 1) ? "}" : ",";
        tdata_var_names += (port.var_name[i] + "_tdata" + optional_comma);
        tkeep_var_names += (port.var_name[i] + "_tkeep" + optional_comma);
        tlast_var_names += (port.var_name[i] + "_tlast" + optional_comma);
        tvalid_var_names += (port.var_name[i] + "_tvalid" + optional_comma);
        tready_var_names += (port.var_name[i] + "_tready" + optional_comma);
      }
      file << ",\n";

      file << tab << "//" << port.debuginfo << "\n";
      file << tab << "." << port.port_name << "_tdata(" << tdata_var_names
           << "),\n";
      if (port.axis.if_keep)
        file << tab << "." << port.port_name << "_tkeep(" << tkeep_var_names
             << "),\n";
      if (port.axis.if_last)
        file << tab << "." << port.port_name << "_tlast(" << tlast_var_names
             << "),\n";
      file << tab << "." << port.port_name << "_tvalid(" << tvalid_var_names
           << "),\n";
      file << tab << "." << port.port_name << "_tready(" << tready_var_names
           << ")";
    } else if (port.type == TABLE_LOOKUP) {
      std::string req_index_var_names = port.var_name.size() == 0 ? "": "{";
      std::string req_valid_var_names = port.var_name.size() == 0 ? "": "{";
      std::string req_ready_var_names = port.var_name.size() == 0 ? "": "{";
      std::string value_valid_var_names = port.var_name.size() == 0 ? "": "{";
      std::string value_data_var_names = port.var_name.size() == 0 ? "": "{";
      std::string value_ready_var_names = port.var_name.size() == 0 ? "": "{";
      // if multiple input vars, cocact them
      for (int i = 0; i < port.var_name.size(); i++) {
        auto optional_comma = (i == port.var_name.size() - 1) ? "}" : ",";
        req_index_var_names += (port.var_name[i] + "_req_index" + optional_comma);
        req_valid_var_names += (port.var_name[i] + "_req_valid" + optional_comma);
        req_ready_var_names += (port.var_name[i] + "_req_ready" + optional_comma);
        value_valid_var_names += (port.var_name[i] + "_value_valid" + optional_comma);
        value_data_var_names += (port.var_name[i] + "_value_data" + optional_comma);
        value_ready_var_names += (port.var_name[i] + "_value_ready" + optional_comma);
      }
      file << ",\n";
      file << tab << "//" << port.debuginfo << "\n";
      file << tab << "." << port.port_name << "_req_index(" << req_index_var_names
           << "),\n";
      file << tab << "." << port.port_name << "_req_valid(" << req_valid_var_names
           << "),\n";
      file << tab << "." << port.port_name << "_req_ready(" << req_ready_var_names
           << "),\n";

      file << tab << "." << port.port_name << "_value_valid("
           << value_valid_var_names << "),\n";
      file << tab << "." << port.port_name << "_value_data(" << value_data_var_names
           << "),\n";
      file << tab << "." << port.port_name << "_value_ready("
           << value_ready_var_names << ")";
    } else if (port.type == TABLE_UPDATE) {
      std::string req_index_var_names = port.var_name.size() == 0 ? "": "{";
      std::string req_valid_var_names = port.var_name.size() == 0 ? "": "{";
      std::string req_ready_var_names = port.var_name.size() == 0 ? "": "{";
      std::string req_data_var_names = port.var_name.size() == 0 ? "": "{";
      std::string req_data_valid_var_names = port.var_name.size() == 0 ? "": "{";
      std::string req_data_ready_var_names = port.var_name.size() == 0 ? "": "{";
      // if multiple input vars, cocact them
      for (int i = 0; i < port.var_name.size(); i++) {
        auto optional_comma = (i == port.var_name.size() - 1) ? "}" : ",";
        req_index_var_names += (port.var_name[i] + "_req_index" + optional_comma);
        req_valid_var_names += (port.var_name[i] + "_req_index_valid" + optional_comma);
        req_ready_var_names += (port.var_name[i] + "_req_index_ready" + optional_comma);
        req_data_var_names += (port.var_name[i] + "_req_data" + optional_comma);
        req_data_valid_var_names += (port.var_name[i] + "_req_data_valid" + optional_comma);
        req_data_ready_var_names += (port.var_name[i] + "_req_data_ready" + optional_comma);
      }
      file << ",\n";
      file << tab << "//" << port.debuginfo << "\n";
      file << tab << "." << port.port_name << "_req_index(" << req_index_var_names
           << "),\n";
      file << tab << "." << port.port_name << "_req_index_valid(" << req_valid_var_names
           << "),\n";
      file << tab << "." << port.port_name << "_req_index_ready(" << req_ready_var_names
           << "),\n";
      file << tab << "." << port.port_name << "_req_data(" << req_data_var_names
           << "),\n";
      file << tab << "." << port.port_name << "_req_data_valid(" << req_data_valid_var_names
           << "),\n";
      file << tab << "." << port.port_name << "_req_data_ready(" << req_data_ready_var_names
           << ")";
    } else if (port.type == BIT) {
      std::string var_names = "{";
      for (int i = 0; i < port.var_name.size(); i++) {
        auto optional_comma = (i == port.var_name.size() - 1) ? "}" : ",";
        var_names += (port.var_name[i] + optional_comma);
      }
      file << ",\n";
      file << tab << "//" << port.debuginfo << "\n";
      file << tab << "." << port.port_name << "(" << var_names << ")";
    }
  }
  file << "\n);\n\n";
}

unsigned EmitFPGAPass::getStructTotalSize(ep2::StructType in_struct) {
  unsigned total_size = 0;
  auto elements = in_struct.getElementTypes();
  for (auto &e : elements) {
    int t;
    GetValTypeAndSize(e, &t);
    total_size += t;

    if (!e.isIntOrFloat() && !isa<ep2::StructType>(e)) {
      printf("Error: Cannot getStructTotalSize\n");
      e.dump();
      assert(false);
    }
  }
  return total_size;
}

unsigned EmitFPGAPass::getContextTotalSize() {
  unsigned total_size = 0;
  auto &table = contextbufferAnalysis->getContextTable(*cur_funcop);
  for (auto &[_, p] : table) {
    auto type = p.second;
    int t;
    GetValTypeAndSize(type, &t);
    total_size += t;

    if (!type.isIntOrFloat() && !isa<ep2::StructType>(type)) {
      printf("Error: Cannot getContextTotalSize\n");
      type.dump();
      assert(false);
    }
  }
  return total_size;
}

unsigned EmitFPGAPass::getStructValOffset(ep2::StructType in_struct,
                                          int index) {
  unsigned offset = 0;
  int cur_index = 0;
  auto elements = in_struct.getElementTypes();
  assert(index < elements.size());
  for (auto &e : elements) {
    if (cur_index == index)
      break;

    int t;
    GetValTypeAndSize(e, &t);
    offset += t;
    if (!e.isIntOrFloat() && !isa<ep2::StructType>(e)) {
      printf("Error: Cannot getStructValOffset\n");
      e.dump();
      assert(false);
    }
    cur_index++;
  }
  return offset;
}

unsigned EmitFPGAPass::getStructValSize(ep2::StructType in_struct, int index) {
  unsigned size = 0;
  int cur_index = 0;
  auto elements = in_struct.getElementTypes();
  assert(index < elements.size());
  auto e = elements[index];

  int t;
  GetValTypeAndSize(e, &t);
  size = t;
  return size;
}

std::string EmitFPGAPass::assign_name(std::string prefix) {
  return prefix + "_" + std::to_string(global_var_index++);
}

std::string EmitFPGAPass::assignValNameAndUpdate(mlir::Value val, std::string prefix, bool if_add_gindex) {
  if(val_names_and_useid.find(val) != val_names_and_useid.end()){
    val.dump();
    printf("Error: val already defined\n");
    assert(false);
  }

  auto name = prefix;
  if(if_add_gindex)
    name = name + "_" + std::to_string(global_var_index++);
  int i = val_use_count(val);
  val_names_and_useid[val] = {name, i, 0};
  bool if_buf_block_arg = isa<mlir::BlockArgument>(val);
  // for debug purpose
  if (!if_buf_block_arg) {
    val.getDefiningOp()->setAttr("var_name", builder->getStringAttr(name));
  }
  return name;
}

std::string EmitFPGAPass::getValName(mlir::Value val) {
  if(val_names_and_useid.find(val) == val_names_and_useid.end()){
    val.dump();
    assert(false);
  }
  auto name_and_uses = val_names_and_useid[val];
  if(name_and_uses.cur_use >=  name_and_uses.total_uses){
    printf("Error: val %s's use exceed totoal use %d\n", name_and_uses.name.c_str(), name_and_uses.total_uses);
    val.dump();
    assert(false);
  }

  if(name_and_uses.total_uses == 1){
    name_and_uses.cur_use++;
    val_names_and_useid[val] = name_and_uses;
    return name_and_uses.name;
  } else {
    auto return_name = name_and_uses.name + "_" + std::to_string(name_and_uses.cur_use);
    name_and_uses.cur_use++;
    val_names_and_useid[val] = name_and_uses;
    return return_name;
  }
}

EmitFPGAPass::VAL_TYPE EmitFPGAPass::GetValTypeAndSize(mlir::Type type,
                                                       int *outsize) {
  int size;
  VAL_TYPE enum_type;
  if (isa<ep2::ContextType>(type)) {
    // push context wire
    assert(cur_funcop != NULL);
    size = getContextTotalSize();
    enum_type = CONTEXT;
  } else if (isa<ep2::BufferType>(type)) {
    size = DEFAULT_AXIS_STREAM_SIZE;
    enum_type = BUF;
  } else if (type.isIntOrFloat()) {
    size = type.getIntOrFloatBitWidth();
    enum_type = INT;
  } else if (isa<ep2::StructType>(type)) {
    auto arg_struct = cast<ep2::StructType, Type>(type);
    size = getStructTotalSize(arg_struct);
    enum_type = STRUCT;
  } else if (isa<ep2::AtomType>(type)) {
    enum_type = ATOM;
    size = -1;
  } else {
    enum_type = UNKNOWN;
    size = -1;
  }

  *outsize = size;
  return enum_type;
}

} // namespace ep2
} // namespace mlir