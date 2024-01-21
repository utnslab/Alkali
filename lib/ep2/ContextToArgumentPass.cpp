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

void rewriteEventInit(InitOp initOp, ContextBufferizationAnalysis &analysis,
                      HandlerDependencyAnalysis &dependency, bool noContext) {
  OpBuilder builder(initOp);

  auto type = initOp.getType().cast<StructType>();

  auto argList = llvm::to_vector(initOp.getArgs());
  auto it = llvm::find_if(
      argList, [](Value &t) { return isa<ContextType>(t.getType()); });
  if (it == argList.end())
    return;
  auto context = *it;
  argList.erase(it);

  auto strings = llvm::map_to_vector(argList, [](Value &v){ return std::string(""); });

  // insert new types and values
  auto funcOp = initOp->getParentOfType<FuncOp>();
  auto &table = analysis.getContextTable(funcOp);
  // TODO(zhiyuang): switch to a initOp-driven method
  // auto &table = analysis.getContextTable(initOp);
  if (!noContext)
    for (auto &pair : table) {
      auto valueType = builder.getType<ContextRefType>(pair.second.second);
      auto ref = builder.create<ContextRefOp>(initOp.getLoc(), valueType,
                                              pair.first(), context);
      auto load =
          builder.create<LoadOp>(initOp.getLoc(), pair.second.second, ref);
      argList.push_back(load.getResult());
      strings.push_back(pair.first().str());
    }

  // build
  auto typeList =
      llvm::map_to_vector(argList, [](Value &v) { return v.getType(); });
  auto newType = builder.getType<StructType>(true, typeList, type.getName());
  auto newInitOp = builder.create<InitOp>(initOp.getLoc(), newType, argList);

  auto stringList = llvm::map_to_vector(strings, [](auto &s) { return StringRef(s); });
  newInitOp->setAttr("context_names", builder.getStrArrayAttr(stringList));

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

void ContextToArgumentPass::runOnFunction(FuncOp funcOp, ContextBufferizationAnalysis &analysis,
                   HandlerDependencyAnalysis &dependency) {
  if (funcOp.isExtern())
    return;

  // per-function reference map
  std::map<StringRef, ContextRefOp> topRefs;
  if (dependency.hasPredecessor(funcOp))
    insertContextToArguments(funcOp, analysis);
  insertContextRefs(funcOp, topRefs, analysis);

  // rewrite all return with context read
  funcOp.walk([&](InitOp initOp) {
    auto type = initOp.getType().dyn_cast<StructType>();
    if (!type || !type.getIsEvent())
      return;

    bool noContext = !keepLastContext.getValue() && dependency.lookupHandler(initOp).isExtern();
    rewriteEventInit(initOp, analysis, dependency, noContext);
  });

  // rewrite all return with context read
  funcOp.walk([&](ContextRefOp refOp) {
    refOp->setAttr("transferToValue",
                   BoolAttr::get(refOp->getContext(), true));
    // replace them with top level ops, as all context refs could be global
    auto topRefOp = topRefs[refOp.getName()];
    if (topRefOp != refOp)
      refOp.getResult().replaceAllUsesWith(topRefOp.getResult());
  });

  // apply per-function pass
  OpPassManager pm;
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createMem2Reg());
  if (failed(runPipeline(pm, funcOp)))
    return signalPassFailure();

  removeContextArgument(funcOp);
}

// Conversion Pass
void ContextToArgumentPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  auto &analysis = getAnalysis<ContextBufferizationAnalysis>();
  auto &depency = getAnalysis<HandlerDependencyAnalysis>();

  moduleOp.walk([&](FuncOp funcOp) {
    runOnFunction(funcOp, analysis, depency);
  });
}

} // namespace ep2
} // namespace mlir
