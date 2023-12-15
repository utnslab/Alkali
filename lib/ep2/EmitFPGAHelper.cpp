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
#define DEFAULT_AXIS_STREAM_SIZE 512
enum INOUT { IN, OUT };

enum IF_TYPE { AXIS };

struct axis_config {
  int if_keep;
  int if_last;
  int data_width;
};

struct inout_config {
  INOUT direction;
  IF_TYPE type;
  std::string name;
  std::string debuginfo;
  union {
    struct axis_config axis;
  };
};

struct wire_config {
  IF_TYPE type;
  std::string name;
  std::string debuginfo;
  bool if_init_value;
  bool if_use;
  union {
    struct axis_config axis;
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
  };
};

void emitModuleParameter(std::ofstream &file,
                         std::list<struct inout_config> &wires) {
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
  }

  file << "\n);\n";
}

void emitwire(std::ofstream &file, struct wire_config &wire) {
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
      initdatavalue = "=0";
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
  }
  file << "\n";
}

void emitwireassign(std::ofstream &file, struct wire_assign_config &assign) {
  assert(assign.src_wire.type == assign.dst_wire.type);
  if (assign.src_wire.type == AXIS) {
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
  }
  file << "\n";
}

void emitModuleCall(std::ofstream &file, std::string module_type,
                    std::string module_name,
                    std::list<struct module_port_config> &ports,
                    std::list<struct module_param_config> &params) {
  file << module_type << "#(\n";
  int param_num = params.size();
  int i = 0;
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
    std::string tdata_var_names = "{";
    std::string tkeep_var_names = "{";
    std::string tlast_var_names= "{";
    std::string tvalid_var_names= "{";
    std::string tready_var_names= "{";

    if (port.type == AXIS) {
      // if multiple input vars, cocact them
      for(int i = 0; i < port.var_name.size(); i ++){
        auto optional_comma = (i == port.var_name.size()-1) ? "}" : ",";
        tdata_var_names += (port.var_name[i] + "_tdata" + optional_comma);
        tkeep_var_names += (port.var_name[i] + "_tkeep" + optional_comma);
        tlast_var_names += (port.var_name[i] + "_tlast" + optional_comma);
        tvalid_var_names += (port.var_name[i] + "_tvalid" + optional_comma);
        tready_var_names += (port.var_name[i] + "_tready" + optional_comma);
      }
      file << ",\n";
      std::string tab = "\t";

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
    }
  }
  file << "\n);\n\n";
}

unsigned EmitFPGAPass::getStructTotalSize(ep2::StructType in_struct) {
  unsigned total_size = 0;
  auto elements = in_struct.getElementTypes();
  for (auto &e : elements) {
    if (e.isIntOrFloat()) {
      total_size += e.getIntOrFloatBitWidth();
    } else if (isa<ep2::StructType>(e)) {
      auto tmps = cast<ep2::StructType, Type>(e);
      total_size += getStructTotalSize(tmps);
    } else {
      printf("Error: Cannot getStructTotalSize\n");
      e.dump();
      assert(false);
    }
  }
  return total_size;
}

unsigned EmitFPGAPass::
getContextTotalSize(llvm::StringMap<ContextAnalysis::ContextField> &context) {
  unsigned total_size = 0;
  printf("Check Context Size %d\n", context.size());
  for (auto &c : context) {
    auto e = c.second.ty;
    if (e.isIntOrFloat()) {
      total_size += e.getIntOrFloatBitWidth();
    } else if (isa<ep2::StructType>(e)) {
      auto tmps = cast<ep2::StructType, Type>(e);
      total_size += getStructTotalSize(tmps);
    } else {
      printf("Error: Cannot getContextTotalSize\n");
      e.dump();
      assert(false);
    }
  }
  return total_size;
}

unsigned EmitFPGAPass::getStructValOffset(ep2::StructType in_struct, int index) {
  unsigned offset = 0;
  int cur_index = 0;
  auto elements = in_struct.getElementTypes();
  assert(index < elements.size());
  for (auto &e : elements) {
    if (cur_index == index)
      break;
    if (e.isIntOrFloat()) {
      offset += e.getIntOrFloatBitWidth();
    } else if (isa<ep2::StructType>(e)) {
      auto tmps = cast<ep2::StructType, Type>(e);
      offset += getStructTotalSize(tmps);
    } else {
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

  if (e.isIntOrFloat()) {
    size = e.getIntOrFloatBitWidth();
  } else if (isa<ep2::StructType>(e)) {
    auto tmps = cast<ep2::StructType, Type>(e);
    size = getStructTotalSize(tmps);
  } else {
    printf("Error: Cannot getStructValSize\n");
    e.dump();
    assert(false);
  }
  return size;
}

int global_var_index = 0;
std::string assign_var_name(std::string prefix) {
  return prefix + "_" + std::to_string(global_var_index++);
}

std::string EmitFPGAPass::getValName(mlir::Value val) {
  std::string name;
  bool if_buf_block_arg = isa<mlir::BlockArgument>(val);
  if (if_buf_block_arg) {
    assert(arg_names.find(val) != arg_names.end());
    name = arg_names[val];
  } else {
    assert(val.getDefiningOp()->hasAttr("var_name"));
    name = val.getDefiningOp()
               ->getAttr("var_name")
               .cast<StringAttr>()
               .getValue()
               .str();
  }
  return name;
}

void EmitFPGAPass::UpdateValName(mlir::Value val, std::string name) {
  bool if_buf_block_arg = isa<mlir::BlockArgument>(val);
  if (if_buf_block_arg) {
    arg_names[val] = name;
  } else {
    val.getDefiningOp()->setAttr("var_name", builder->getStringAttr(name));
  }
}

EmitFPGAPass::VAL_TYPE EmitFPGAPass::GetValTypeAndSize(mlir::Type type,
                                                       int *outsize) {
  int size;
  VAL_TYPE enum_type;
  if (isa<ep2::ContextType>(type)) {
    // push context wire
    assert(cur_funcop != NULL);
    auto group = contextAnalysis->disj_func_groups[*cur_funcop];
    auto fileds = contextAnalysis->disj_contexts[group];
    size = getContextTotalSize(fileds);
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

}
}