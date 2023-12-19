#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "ep2/lang/Parser.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/MLIRGen.h"
#include "ep2/dialect/Passes.h"

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
 
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::ep2::EP2Dialect>();

  mlir::PassRegistration<mlir::ep2::ContextTypeInferencePass>();
  mlir::PassRegistration<mlir::ep2::NopEliminationPass>();
  mlir::PassRegistration<mlir::ep2::CollectHeaderPass>();
  mlir::PassRegistration<mlir::ep2::LowerEmitcPass>();
  mlir::PassRegistration<mlir::ep2::LowerIntrinsicsPass>();
  mlir::PassRegistration<mlir::ep2::EmitFilesPass>();
  mlir::registerAllPasses();
 
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
