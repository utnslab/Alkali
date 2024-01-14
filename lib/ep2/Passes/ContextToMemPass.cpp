#include "mlir/IR/BuiltinDialect.h"

#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace ep2 {

namespace {

void restoreContext(FuncOp funcOp) {
  if (funcOp.isController() || funcOp.isExtern())
    return;

  // insert all buffers from function arugments
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&funcOp.getBody().front());

  DenseMap<StringRef, int> contextName;
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    auto attr = funcOp.getArgAttrOfType<StringAttr>(i, "ep2.context_name");
    if (attr)
      contextName.try_emplace(attr.getValue(), i);
  }

  // insert a function arg
  funcOp.insertArgument(0, builder.getType<ContextType>(), {}, funcOp.getLoc());
  auto ctxArg = funcOp.getArgument(0);

  // remove all context args, if any
  DenseMap<Value, Value> contextRefs;
  DenseMap<StringRef, std::pair<Value, Value>> nameToRefs;
  for (auto &[name, i] : contextName) {
    auto ba = funcOp.getArgument(i + 1);
    auto refType = builder.getType<ContextRefType>(ba.getType());
    auto ref = builder.create<ContextRefOp>(funcOp.getLoc(), refType, name, ctxArg);
    contextRefs.try_emplace(ba, ref);
    nameToRefs.try_emplace(name, ba, ref);
  }

  // change the init op
  SmallVector<InitOp> toErase;
  funcOp->walk([&](InitOp initOp) {
    auto event = initOp.getType().dyn_cast<StructType>();
    if (!event || !event.getIsEvent())
      return;

    auto attr = initOp->getAttrOfType<ArrayAttr>("context_names");
    if (!attr)
      return;
    auto mapping = attr.getValue();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(initOp);
    SmallVector<Value> newOperands;
    for (unsigned i = 0; i < initOp.getNumOperands(); i++) {
      auto name = mapping[i].cast<StringAttr>();
      if (name == "")
        newOperands.push_back(initOp.getOperand(i));
      else {
        auto value = initOp.getOperand(i);
        auto it = nameToRefs.find(name);
        if (it == nameToRefs.end()) {
          // no context ref is created..
          auto refType = builder.getType<ContextRefType>(value.getType());
          auto ref = builder.create<ContextRefOp>(funcOp.getLoc(), refType, name, ctxArg);
          builder.create<StoreOp>(funcOp.getLoc(), ref, initOp.getOperand(i));
        } else {
          auto &[ba, ref] = it->second;
          if (value != ba) // we avoid load and store
            builder.create<StoreOp>(funcOp.getLoc(), ref, value);
        }
      }
    }

    // insert ctx
    auto insertIt = newOperands.begin();
    if (initOp.getOperand(0).getType().isa<AtomType>())
      insertIt++;
    newOperands.insert(insertIt, ctxArg);

    // get new type
    auto newTypes = llvm::map_to_vector(newOperands, [](Value &v) { return v.getType(); });
    auto newEvent = builder.getType<StructType>(true, newTypes, event.getName());

    auto newInitOp =
        builder.create<InitOp>(funcOp.getLoc(), newEvent, newOperands);
    initOp.replaceAllUsesWith(newInitOp.getResult());
    toErase.push_back(initOp);
  });
  for (auto &op : toErase)
    op.erase();

  // replace all reads
  for (auto &[ba, ref] : contextRefs) {
    auto load = builder.create<LoadOp>(funcOp.getLoc(), ba.getType(), ref);
    ba.replaceAllUsesWith(load);
  }

  BitVector argMask{funcOp.getNumArguments()};
  for (auto &[_, i] : contextName)
    argMask.set(i + 1);
  funcOp.eraseArguments(argMask);
}

} // namespace

void ContextToMemPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  OpBuilder builder(moduleOp);

  // restore context
  moduleOp->walk(restoreContext);

  OpPassManager pm;
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, moduleOp)))
    return signalPassFailure();
}

} // namespace ep2
} // namespace mlir