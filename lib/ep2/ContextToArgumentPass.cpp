#include "mlir/IR/BuiltinDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"

using namespace mlir;

namespace mlir {
namespace ep2 {

//===----------------------------------------------------------------------===//
// Load & Store Ops
//===----------------------------------------------------------------------===//

namespace {

void insertContextToArguments(FuncOp funcOp,
                              ContextBufferizationAnalysis &analysis) {
  OpBuilder builder(funcOp);

  auto args = funcOp.getArguments();
  if (llvm::none_of(args, [](BlockArgument &ba) {
        return isa<ContextType>(ba.getType());
      }))
    return;

  // insert all parameter as names
  auto &table = analysis.getContextTable(funcOp);
  auto last = funcOp.getNumArguments();
  for (auto &pair : table) {
    auto name = builder.getNamedAttr("ep2.context_name",
                                     builder.getStringAttr(pair.first()));
    auto dict = builder.getDictionaryAttr({name});
    auto type = pair.second.second;
    funcOp.insertArgument(last++, type, dict, funcOp.getLoc());
  }
}

void insertContextRefs(FuncOp funcOp, std::map<StringRef, ContextRefOp> &refs,
                      ContextBufferizationAnalysis &analysis) {
  OpBuilder builder(funcOp);
  auto it = llvm::find_if(funcOp.getArguments(), [](BlockArgument &ba) {
    return ba.getType().isa<ContextType>();
  });
  if (it == funcOp.getArguments().end())
    return;
  auto context = *it;

  // insert all reference at begining
  auto &table = analysis.getContextTable(funcOp);
  builder.setInsertionPointToStart(&funcOp.getBody().front());
  for (auto &pair : table) {
    auto refType = builder.getType<ContextRefType>(pair.second.second);
    auto refOp = builder.create<ContextRefOp>(funcOp.getLoc(), refType,
                                              pair.first(), context);
    refs.insert({pair.first(), refOp});
  }
}

void rewriteEventInit(InitOp initOp, ContextBufferizationAnalysis &analysis) {
  OpBuilder builder(initOp);

  auto result = initOp.getResult();
  auto type = result.getType().dyn_cast<StructType>();
  if (!type || !type.getIsEvent())
    return;

  auto argList = llvm::to_vector(initOp.getArgs());
  auto it = llvm::find_if(
      argList, [](Value &t) { return isa<ContextType>(t.getType()); });
  if (it == argList.end())
    return;
  auto context = *it;
  argList.erase(it);

  // insert new types and values
  auto funcOp = initOp->getParentOfType<FuncOp>();
  auto &table = analysis.getContextTable(funcOp);
  // TODO(zhiyuang): switch to a initOp-driven method
  // auto &table = analysis.getContextTable(initOp);
  for (auto &pair : table) {
    auto valueType = builder.getType<ContextRefType>(pair.second.second);
    auto ref = builder.create<ContextRefOp>(initOp.getLoc(), valueType,
                                            pair.first(), context);
    auto load =
        builder.create<LoadOp>(initOp.getLoc(), pair.second.second, ref);
    argList.push_back(load.getResult());
  }

  // build
  auto typeList =
      llvm::map_to_vector(argList, [](Value &v) { return v.getType(); });
  auto newType = builder.getType<StructType>(true, typeList, type.getName());
  auto newInitOp = builder.create<InitOp>(initOp.getLoc(), newType, argList);
  initOp.replaceAllUsesWith(newInitOp.getResult());
  initOp.erase();
}

void removeContextArgument(FuncOp funcOp) {
  for (auto &ba : funcOp.getArguments()) {
    if (ba.getType().isa<ContextType>()) {
      funcOp.eraseArgument(ba.getArgNumber());
      return;
    }
  }
}
} // namespace Patterns

// Conversion Pass
void ContextToArgumentPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  auto &analysis = getAnalysis<ContextBufferizationAnalysis>();
  auto &depency = getAnalysis<HandlerDependencyAnalysis>();

  // insert arguments
  std::map<StringRef, ContextRefOp> refs;
  moduleOp.walk([&](FuncOp funcOp){
    if (depency.hasPredecessor(funcOp))
      insertContextToArguments(funcOp, analysis);
    insertContextRefs(funcOp, refs, analysis);
  });

  // rewrite all return with context read
  // TODO: remove if its the last op
  moduleOp.walk([&](InitOp initOp) {
    rewriteEventInit(initOp, analysis);
  });

  // mark all ref ops
  moduleOp.walk([&](ContextRefOp refOp) {
    refOp->setAttr("transferToValue",
                   BoolAttr::get(moduleOp.getContext(), true));
    // replace them with top level ops, as all context refs could be global
    auto topRefOp = refs[refOp.getName()];
    if (topRefOp != refOp)
      refOp.getResult().replaceAllUsesWith(topRefOp.getResult());
  });

  // TODO()
  // execute mem2reg on all transformed funcs
  OpPassManager pm;
  auto &funcPm = pm.nest<FuncOp>();
  funcPm.addPass(createCanonicalizerPass());
  funcPm.addPass(createConvertSCFToCFPass());
  funcPm.addPass(createMem2Reg());
  funcPm.addPass(createCanonicalizerPass());
  funcPm.addPass(createCSEPass());

  if (failed(runPipeline(pm, moduleOp)))
    return signalPassFailure();

  // rewrite the init and remove
  moduleOp.walk([&](FuncOp funcOp) {
    removeContextArgument(funcOp);
  });
}

} // namespace ep2
} // namespace mlir
