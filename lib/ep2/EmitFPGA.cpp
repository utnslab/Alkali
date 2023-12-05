#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
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
  std::string var_name;
  std::string debuginfo;
  std::string port_name;
  union {
    struct axis_config axis;
  };
};

void emitFuncHeader(std::ofstream &file, std::string name) {
  file << "module " << name << "#()\n";
}
void emitModuleParameter(std::ofstream &file,
                         std::list<struct inout_config> &wires) {
  file << "(\n"
       << "\t input  wire clk, \n"
       << "\t input  wire rst, \n";
  for (auto &wire : wires) {
    if (wire.type == AXIS) {
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
      file << tab << inoutstr << " wire "
           << " " << wire.name << "_tvalid ,\n";
      file << tab << reverse_inoutstr << " wire "
           << " " << wire.name << "_tready ,\n";
      if (wire.axis.if_last)
        file << tab << inoutstr << " wire "
             << " " << wire.name << "_tlast ,\n";
    }
    file << "\n";
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
    if (wire.if_init_value)
      initvalue = "=1";

    file << "//" << wire.debuginfo << "\n";
    file << " wire " << datawidthstr << " " << wire.name << "_tdata ;\n";
    if (wire.axis.if_keep)
      file << " wire " << keepwidthstr << " " << wire.name << "_tkeep ;\n";
    file << " wire "
         << " " << wire.name << "_tvalid" << initvalue << ";\n";
    file << " wire "
         << " " << wire.name << "_tready ;\n";
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
    assert(!assign.dst_wire.axis.if_last && !assign.src_wire.axis.if_last &&
           !assign.dst_wire.axis.if_keep && !assign.src_wire.axis.if_keep);
  }
  file << "\n";
}

void emitModuleCall(std::ofstream &file, std::string module_type,
                    std::string module_name,
                    std::list<struct module_port_config> &ports,
                    std::list<struct module_param_config> &params) {
  file << module_type << "#(\n";
  for (auto &param : params) {
    file << "." << param.paramname << "(" << std::to_string(param.paramval)
         << "),\n";
  }

  file << ")" << module_name << "(\n"
       << "\t .clk(clk), \n"
       << "\t .rst(rst), \n\n";
  for (auto &port : ports) {
    if (port.type == AXIS) {
      std::string tab = "\t";

      file << tab << "//" << port.debuginfo << "\n";
      file << tab << "." << port.port_name << "_tdata(" << port.var_name
           << "_tdata),\n";
      if (port.axis.if_keep)
        file << tab << "." << port.port_name << "_tkeep(" << port.var_name
             << "_tkeep),\n";
      file << tab << "." << port.port_name << "_tvalid(" << port.var_name
           << "_tvalid),\n";
      file << tab << "." << port.port_name << "_tready(" << port.var_name
           << "_tready),\n";
      if (port.axis.if_last)
        file << tab << "." << port.port_name << "_tlast(" << port.var_name
             << "_tlast),\n";
    }
  }
  file << ")\n\n";
}

unsigned getStructTotalSize(ep2::StructType in_struct) {
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

unsigned
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

unsigned getStructValOffset(ep2::StructType in_struct, int index) {
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

unsigned getStructValSize(ep2::StructType in_struct, int index) {
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

void EmitFPGAPass::emitVariableInit(std::ofstream &file, ep2::InitOp initop) {

  auto arg = initop.getResult();
  auto arg_type = arg.getType();
  bool if_stream;
  int size;
  std::string name, debuginfo;
  if (arg_type.isIntOrFloat()) {
    size = arg_type.getIntOrFloatBitWidth();
    if_stream = false;
    debuginfo = "inited_int_var";
  } else if (isa<ep2::StructType>(arg_type)) {
    auto arg_struct = cast<ep2::StructType, Type>(arg_type);
    size = getStructTotalSize(arg_struct);
    if_stream = false;
    debuginfo = "inited_struct_var";
  } else if (isa<ep2::BufferType>(arg_type)) {
    if_stream = true;
    size = DEFAULT_AXIS_STREAM_SIZE;
    debuginfo = "inited_buf_var";
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
  struct wire_config wire = {AXIS, name, debuginfo, true, axis};
  emitwire(file, wire);

  // printf("Get Init op %s\n", defined_value->hashProperties());
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
  // bool if_buf_block_arg = isa<mlir::BlockArgument>(buf);
  // if (if_buf_block_arg) {
  //   assert(arg_names.find(buf) != arg_names.end());
  //   ori_buf_name = arg_names[buf];
  // } else {
  //   assert(buf.getDefiningOp()->hasAttr("var_name"));
  //   ori_buf_name = buf.getDefiningOp()
  //                      ->getAttr("var_name")
  //                      .cast<StringAttr>()
  //                      .getValue()
  //                      .str();
  // }
  in_buf_axis = {1, 1, DEFAULT_AXIS_STREAM_SIZE};
  in_buf_port = {AXIS, ori_buf_name, "input buf", "s_inbuf_axis", in_buf_axis};

  auto new_buf_name = assign_var_name("bufvar");
  UpdateValName(buf, new_buf_name);
  // if (if_buf_block_arg) {
  //   arg_names[buf] = new_buf_name;
  // } else {
  //   buf.getDefiningOp()->setAttr("var_name",
  //                                builder->getStringAttr(new_buf_name));
  // }
  out_buf_axis = {1, 1, DEFAULT_AXIS_STREAM_SIZE};
  out_buf_port = {AXIS, new_buf_name, "output buf", "m_outbuf_axis",
                  out_buf_axis};
  out_buf_wire = {AXIS, new_buf_name, module_name + " output buf", 0,
                  out_buf_axis};

  auto extracted_struct = extractop.getResult();
  assert(!extracted_struct.getDefiningOp()->hasAttr("var_name"));
  auto extracted_struct_name = assign_var_name("structvar");
  UpdateValName(extracted_struct, extracted_struct_name);
  // extracted_struct.getDefiningOp()->setAttr(
  //     "var_name", builder->getStringAttr(extracted_struct_name));
  auto extracted_struct_type = extracted_struct.getType();
  int extracted_struct_size = 0;
  // struct extract value can only be struct or int
  if (extracted_struct_type.isIntOrFloat()) {
    extracted_struct_size = extracted_struct_type.getIntOrFloatBitWidth();
  } else if (isa<ep2::StructType>(extracted_struct_type)) {
    auto val_struct = cast<ep2::StructType, Type>(extracted_struct_type);
    extracted_struct_size = getStructTotalSize(val_struct);
  } else {
    printf("Error: Cannot calculate emitExtract's output struct size\n");
    extracted_struct.dump();
    assert(false);
  }
  out_struct_axis = {0, 0, extracted_struct_size};
  out_struct_port = {AXIS, extracted_struct_name, "output struct",
                     "m_extracted_axis", out_struct_axis};
  out_struct_wire = {AXIS, extracted_struct_name,
                     module_name + " output struct", 0, out_struct_axis};

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

  emitModuleCall(file, "extract_eth", module_name, ports, params);
}

void EmitFPGAPass::emitStructAccess(std::ofstream &file,
                                    ep2::StructAccessOp structaccessop) {

  auto outval = structaccessop.getResult();
  auto outval_type = outval.getType();
  struct axis_config outval_axis, srcval_axis;
  struct wire_config outval_wire, srcval_wire;
  struct wire_assign_config wire_assignment;
  int size;
  std::string debuginfo;

  // struct access value can only be struct or int
  if (outval_type.isIntOrFloat()) {
    size = outval_type.getIntOrFloatBitWidth();
    debuginfo = "struct_accessed_int_var";
  } else if (isa<ep2::StructType>(outval_type)) {
    auto val_struct = cast<ep2::StructType, Type>(outval_type);
    size = getStructTotalSize(val_struct);
    debuginfo = "struct_accessed_struct_var";
  } else {
    printf("Error: Cannot emitStructAccess's output val\n");
    outval.dump();
    assert(false);
  }

  assert(!outval.getDefiningOp()->hasAttr("var_name"));
  auto name = assign_var_name(debuginfo);
  // outval.getDefiningOp()->setAttr("var_name", builder->getStringAttr(name));
  UpdateValName(outval, name);
  outval_axis = {0, 0, size};
  outval_wire = {AXIS, name, "Access Struct", 0, outval_axis};
  emitwire(file, outval_wire);
  printf("Access Struct out val %s\n",name.c_str() );
  outval.getType().dump();

  auto srcval = structaccessop.getInput();
  auto srcval_index = structaccessop.getIndex();
  assert(isa<ep2::StructType>(srcval.getType()));

  auto srcval_type = cast<ep2::StructType, Type>(srcval.getType());
  int src_offset = getStructValOffset(srcval_type, srcval_index);
  int src_size = getStructValSize(srcval_type, srcval_index);
  int src_struct_size = getStructTotalSize(srcval_type);

  // if (!srcval.getDefiningOp()->hasAttr("var_name"))
  //   return;
  auto src_struct_name = getValName(srcval);
  // assert(srcval.getDefiningOp()->hasAttr("var_name"));
  // auto src_struct_name = srcval.getDefiningOp()
  //                            ->getAttr("var_name")
  //                            .cast<StringAttr>()
  //                            .getValue()
  //                            .str();

  srcval_axis = {0, 0, src_struct_size};
  srcval_wire = {AXIS, src_struct_name, "Struct Assign Src Struct", 0,
                 srcval_axis};

  wire_assignment = {src_offset, src_size, -1, -1, srcval_wire, outval_wire};
  emitwireassign(file, wire_assignment);
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
  ori_struct_port = {AXIS, ori_struct_name, "input struct", "s_struct_axis",
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
  in_val_port = {AXIS, in_val_name, "input val", "s_assignv_axis", in_val_axis};

  auto new_struct = structupdateop.getOutput();
  auto new_struct_name = assign_var_name("structvar");
  auto new_struct_type = new_struct.getType();
  UpdateValName(new_struct, new_struct_name);
  assert(isa<ep2::StructType>(new_struct_type));
  int new_struct_size =
      getStructTotalSize(cast<ep2::StructType, Type>(new_struct_type));
  assert(new_struct_size == ori_struct_size);
  new_struct_axis = {0, 0, new_struct_size};
  new_struct_port = {AXIS, new_struct_name, "output struct", "m_struct_axis",
                     new_struct_axis};
  new_struct_wire = {AXIS, new_struct_name, module_name + " output struct", 0,
                     new_struct_axis};

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

  // printf("Struct Update\n");
  // ori_struct.dump();
}

void EmitFPGAPass::runOnOperation() {
  module = getOperation();
  OpBuilder builder_tmp(module->getContext());
  builder = &builder_tmp;
  // install analysis
  // LowerStructAnalysis &lowerStructAnalysis =
  // getAnalysis<LowerStructAnalysis>();
  ContextAnalysis &contextAnalysis = getAnalysis<ContextAnalysis>();

  module->walk([&](ep2::FuncOp funcOp) {
    auto handler_name = funcOp.getName().str();
    std::ofstream fout_stage(handler_name + ".sv");
    emitFuncHeader(fout_stage, handler_name);

    std::list<struct inout_config> input_wires;

    // push parameter wire
    auto args = funcOp.getArguments();
    for (auto arg : args) {
      printf("%s parameter wire for\n", handler_name.c_str());
      arg.dump();

      auto arg_type = arg.getType();
      bool if_stream;
      int size;
      std::string name, debuginfo;

      if (isa<ep2::ContextType>(arg_type)) {
        // push context wire
        auto group = contextAnalysis.disj_func_groups[funcOp];
        auto fileds = contextAnalysis.disj_contexts[group];
        size = getContextTotalSize(fileds);
        if (size == 0)
          continue;
        if_stream = false;
        debuginfo = "Context";
      } else if (isa<ep2::BufferType>(arg_type)) {
        if_stream = true;
        size = DEFAULT_AXIS_STREAM_SIZE;
        debuginfo = "Inbuf";
      } else if (arg_type.isIntOrFloat()) {
        size = arg_type.getIntOrFloatBitWidth();
        if_stream = false;
        debuginfo = "IntorFloat";
      } else if (isa<ep2::StructType>(arg_type)) {
        auto arg_struct = cast<ep2::StructType, Type>(arg_type);
        size = getStructTotalSize(arg_struct);
        if_stream = false;
        debuginfo = "Struct";
      } else if (isa<ep2::AtomType>(arg_type)) {
        continue;
      } else {
        printf("Error: Cannot generate in parameter wire for\n");
        arg_type.dump();
        assert(false);
      }

      name = "arg" + std::to_string(arg.getArgNumber());
      assert(arg_names.find(arg) == arg_names.end());
      // auto x = mlir::hash_value(arg);
      arg_names[arg] = name;

      if (!if_stream) {
        assert(size <= 64 * 8);
        struct axis_config wire = {0, 0, size};
        struct inout_config in_if = {IN, AXIS, name, debuginfo, wire};
        input_wires.push_back(in_if);
      } else {
        assert(size % 8 == 0);
        struct axis_config wire = {1, 1, size};
        struct inout_config in_if = {IN, AXIS, name, debuginfo, wire};
        input_wires.push_back(in_if);
      }
    }

    emitModuleParameter(fout_stage, input_wires);

    funcOp->walk([&](mlir::Operation *op) {
      if (isa<ep2::InitOp>(op)) {
        auto initop = cast<ep2::InitOp, mlir::Operation *>(op);
        if (initop.getArgs().size() == 0)
          emitVariableInit(fout_stage, initop);
        else {
          printf("Warning TODO: output wire for this interface.\n");
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
      }
      // TODO: Add OP Constant
      // TODO: Change STURCT ACCESS IR to generate new  stream for each accessed struct
      // TODO: Output port
      // TODO: EMIT
    });
  });
}

std::unique_ptr<Pass> createEmitFPGAPass() {
  return std::make_unique<EmitFPGAPass>();
}

} // namespace ep2
} // namespace mlir
