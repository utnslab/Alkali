#include "mlir/IR/BuiltinDialect.h"

#include "ep2/lang/Lexer.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {
namespace ep2 {

namespace {

struct Global {
  std::string type;
  std::string name;
  std::vector<std::string> args;
  Global(std::string type, std::string name, std::vector<std::string> args)
    : type(type), name(name), args(std::move(args)) {}
};

struct Emitter {
  std::vector<Global> globals;

  void addGlobal(std::string type, std::string name, std::vector<std::string> args) {
    globals.emplace_back(type, name, std::move(args));
  }

  llvm::raw_ostream &out;
  Emitter(llvm::raw_ostream &out) : out(out) {}
  virtual void emit() = 0;
};

struct HeaderEmitter : public Emitter {
  using Emitter::Emitter;

  std::map<std::string, int> handlers;
  std::map<std::string, int> handlerReplications;
  std::map<std::string, std::vector<int>> externs;


  static constexpr auto header = R"(
#ifndef _EP2_INC_H_
#define _EP2_INC_H_

#include "runtime.h"

)";

  static constexpr auto footer = R"(
#endif // _EP2_INC_H

)";

  void emitDefine(std::string name, int value) {
    out << "#define " << name << " (" << value << ")\n";
  }

  void emitHandlers() {
    for (auto &[name, in] : handlers) {
      out << "void " << name << "(void * event);\n";
      out << "WORKER_FUNCTION(" << in << "," << name << ");\n";
    }

    out << "handler_worker_t handler_workers[NUM_HANDLERS] = {\n";
    for (auto &[name, in] : handlers)
      out << "    __thread" << name << ",\n";
    out << "};\n";
  }

  void emitHandlerInstances() {
    out << "int handler_replications[NUM_HANDLERS] = {";
    for (auto &[name, in] : handlerReplications)
      out << in << ",";
    out << "};\n";
  }

  int instances = 0;
  void init() {
    for (auto &[name, in] : handlerReplications)
      instances += in;
  }

  void emitExtern() {}

  void emitGlobal() {
    for (auto &global : globals)
      out << "extern " << global.type << "_t " << global.name << ";\n";
    out << "void __ep2_init();";
  }

  void emit() override {
    init();
    out << header;

    emitGlobal();

    out << footer;
  }
};

struct CPPEmitter : public Emitter {
  using Emitter::Emitter;
  
  static constexpr auto header = R"(
#include "ep2.inc.h"

)";

  void emit() override {
    out << header;
    for (auto &global : globals) {
      out << global.type << "_t " << global.name << ";\n";
    }

    out << "void __ep2_init() {\n";
    for (auto &global : globals) {
      auto params = global.args;
      params.insert(params.begin(), "&" + global.name);
      out << "  __rt_" << global.type << "_init(" << llvm::join(params, ",") << ");\n";
    }
    out << "}\n";
  }

};

} // local namespace

void EmitLLVMHeaderPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  // Collect the headers

  std::vector<Emitter*> emitters;

  if (outputDir.empty()) {
    emitError(UnknownLoc::get(moduleOp.getContext()), "output directory not specified");
    return signalPassFailure();
  }
  std::string headerFilename = outputDir + "/ep2.inc.h";
  std::string cppFilename = outputDir + "/ep2.inc.c";

  // open the output file
  std::string errorMessage;
  auto headerFile = mlir::openOutputFile(headerFilename, &errorMessage);
  if (!headerFile) {
    emitError(UnknownLoc::get(moduleOp.getContext()), errorMessage);
    return signalPassFailure();
  }

  HeaderEmitter headerEmitter(headerFile->os());
  emitters.push_back(&headerEmitter);

  // open the output cpp
  auto cppFile = mlir::openOutputFile(cppFilename, &errorMessage);
  if (!cppFile) {
    emitError(UnknownLoc::get(moduleOp.getContext()), errorMessage);
    return signalPassFailure();
  }
  CPPEmitter cppEmitter(cppFile->os());
  emitters.push_back(&cppEmitter);

  moduleOp.walk([&](GlobalOp global){
    std::vector<std::string> args;
    auto type = llvm::TypeSwitch<Type, std::string>(global.getType())
      .Case([&](TableType type) {
        args.push_back(std::to_string(type.getSize()));
        return "table";
      })
      .Default([&](Type type) {
        type.dump();
        llvm_unreachable("unsupported global type");
        return "";
      });
    auto name = global.getName().str();

    for (auto emitter : emitters)
      emitter->addGlobal(type, name, args);
    if (auto ins = global->getAttrOfType<ArrayAttr>("instances")) {
      // with replications. We do the rename
      for (auto placement : ins.getAsValueRange<StringAttr>()) {
        auto newName = name + "_" + placement.str();
        for (auto emitter : emitters)
          emitter->addGlobal(type, newName, args);
      }
    }
  });

  for (auto emitter : emitters)
    emitter->emit();

  // on success
  headerFile->keep();
  cppFile->keep();
}

} // namespace ep2
} // namespace mlir
