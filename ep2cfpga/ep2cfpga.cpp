//===- ep2fpga.cpp - The EP2 Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "ep2/lang/Parser.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/MLIRGen.h"
#include "ep2/dialect/Passes.h"
#include "ep2/dialect/FPGAPasses.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllPasses.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace ep2;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
                                  
namespace {
enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpMLIRLLVM,
  DumpLLVMIR,
};
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump"))
);

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<ep2::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int main(int argc, char **argv) {
  // mlir::registerAsmPrinterCLOptions();
  // mlir::registerMLIRContextCLOptions();
  // mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "ep2 compiler\n");

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  // Register MLIR dialects and names
  mlir::DialectRegistry registry;
  mlir::MLIRContext context(registry);
  // Load our Dialect in this MLIR Context.

  registerAllDialects(registry); 
  registry.insert<mlir::ep2::EP2Dialect>();
  context.getOrLoadDialect<mlir::ep2::EP2Dialect>();

  mlir::registerAllPasses();
  // mlir::registerCSEPass();

  // Load MLIR
  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (emitAction != Action::DumpAST) {
    module = ep2::mlirGen(context, *moduleAST);
    if (!module) {
      llvm::errs() << "Could not generate MLIR module from source\n";
      return 1;
    }
  }

  switch (emitAction) {
  case Action::DumpAST:
    dump(*moduleAST);
    return 0;
  case Action::DumpMLIR:{
    module->dump();
    return 0;
  }
  default: {
    mlir::PassManager pm(&context);
    pm.addPass(std::make_unique<mlir::ep2::EmitFPGAPass>());
    
    pm.run(*module);
  }
  }

  return 0;
}
