#include "mlir/IR/BuiltinDialect.h"

#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "ep2/dialect/Analysis/Passes.h"

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

// RAII guard to erase ops on destruction.
class EraseGuard {
  std::vector<Operation *> ops;
  EraseGuard() {}
  ~EraseGuard() {
    for (auto op : ops)
      op->erase();
  }
  void add(Operation *op) { ops.push_back(op); }
};

int getEP2TypeSize(Type type) {
  return llvm::TypeSwitch<Type, int>(type)
      .Case<StructType>([](StructType type) {
        int size = 0;
        for (auto field : type.getElementTypes())
          size += getEP2TypeSize(field);
        return size;
      })
      .Case<IntegerType>([](IntegerType type) { return type.getWidth(); })
      .Default([](auto type) {
        llvm_unreachable("unknown type");
        return 0;
      });
}

std::string renameSub(StringRef name, int index) {
  return name.str() + "_sub_" + std::to_string(index);
}

struct StructMap {
    StructType type;
    unsigned offset;
};

struct SplitMap {
    std::map<int, int> fieldToIndex;
    std::map<int, int> indexToNewIndex;
    SmallVector<StructMap> newTypes;
    size_t getSize() { return newTypes.size(); }
};

struct SplitMapContext {
  SplitMap map;
  SmallVector<Value> values;
  SplitMapContext(SplitMap map): map(map) {}

  Type indexToType(int index) {
    return map.newTypes[map.fieldToIndex[index]].type;
  }
  Value indexToValue(int index) {
    return values[map.fieldToIndex[index]];
  }
  void updateValueByIndex(int index, Value value) {
      values[map.fieldToIndex[index]] = value;
  }
  int indexToNewIndex(int index) {
    return map.indexToNewIndex[index];
  }
};

// update the operand list for init op
void updateValueList(SmallVector<Value> &values,
                     DenseMap<Value, SplitMapContext> &valueMapping, SmallVector<std::string> *names = nullptr) {
  auto size = values.size();
  for (size_t i = 0; i < size; ++i) {
    auto value = values[i];
    auto it = valueMapping.find(value);
    if (it != valueMapping.end()) {
      auto &ctx = it->second;
      // insert to the end
      for (auto [subIndex, subValue] : llvm::enumerate(ctx.values)) {
        values.push_back(subValue);
        if (names)
          names->push_back(renameSub(names->operator[](i), subIndex));
      }
    }
  }
}

void updateBlockSignature(Block *block, DenseMap<int, SplitMap> &argMapping,
                          DenseMap<Value, SplitMapContext> &valueMapping) {
  auto numArgs = block->getNumArguments();
  auto curNewIndex = numArgs;
  for (unsigned i = 0; i < numArgs; ++i) {
    auto arg = block->getArgument(i);
    auto it = argMapping.find(i);
    if (it != argMapping.end()) {
      auto &newTypes = it->second.newTypes;
      for (unsigned j = 0; j < newTypes.size(); ++j) {
        auto argIndex = curNewIndex++;
        block->insertArgument(argIndex, newTypes[j].type,
                              block->front().getLoc());
        auto [it2, _] = valueMapping.try_emplace(arg, it->second);
        it2->second.values.push_back(block->getArgument(argIndex));
      }
    }
  }
}

void updateFunctionSignature(FuncOp funcOp,
                          DenseMap<int, SplitMap> &argMapping,
                          DenseMap<Value, SplitMapContext> &valueMapping) {
  auto numArgs = funcOp.getNumArguments();
  auto curNewIndex = numArgs;
  for (unsigned i = 0; i < numArgs; ++i) {
    auto arg = funcOp.getArgument(i);
    auto it = argMapping.find(i);
    if (it != argMapping.end()) {
      auto &newTypes = it->second.newTypes;
      for (unsigned j = 0; j < newTypes.size(); ++j) {
        auto argIndex = curNewIndex++;

        OpBuilder builder(funcOp);
        auto newName = renameSub(funcOp.getArgAttrOfType<StringAttr>(i, "ep2.context_name"), j);
        auto named = builder.getNamedAttr("ep2.context_name", builder.getStringAttr(newName));
        funcOp.insertArgument(argIndex, newTypes[j].type, builder.getDictionaryAttr({named}), funcOp.getLoc());

        auto [it2, _] = valueMapping.try_emplace(arg, it->second);
        it2->second.values.push_back(funcOp.getArgument(argIndex));
      }
    }
  }
}

} // local namespace

void DeadFieldEliminatePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  OpBuilder builder(getOperation());
  auto &fieldAnalysis = getAnalysis<ExtractFiledAnalysis>();
  auto &result = fieldAnalysis.globalState;

  // construct a solution and mapping
  llvm::DenseMap<Value, SplitMap> sourceMap;
  for (auto &[value, usage] : result.usage) {
    if (auto op = value.getDefiningOp<ExtractOffsetOp>()) {
      auto [it, _] = sourceMap.try_emplace(value);
      auto &map = it->second;

      SmallVector<Type> fields;
      unsigned offset = 0;

      for (auto [i, field] : llvm::enumerate(usage.type.getElementTypes())) {
        if (usage.used[i]) {
          // we collect the fields
          map.fieldToIndex.try_emplace(i, map.newTypes.size());
          map.indexToNewIndex.try_emplace(i, fields.size());
          fields.push_back(field);
        }

        if (!usage.used[i] && !fields.empty()) {
          // or if we have some value collected..
          // TODO(zhiyuang): after loop
          std::string newName =  usage.type.getName().str() + "_sub_" + std::to_string(map.newTypes.size());
          auto structType = builder.getType<StructType>(
              false, fields, newName);
              
          map.newTypes.push_back({structType, offset});
          fields.clear();
        }
        offset += getEP2TypeSize(field);
      }
    }
  }

  DenseMap<Value, SmallVector<Value>> replaceMap;
  // replace the definition
  for (auto &[value, targets] : sourceMap) {
    auto extractOp = value.getDefiningOp<ExtractOffsetOp>();
    assert(extractOp && "must be extract op");

    auto [it, _] = replaceMap.try_emplace(value);
    auto &values = it->second;

    builder.setInsertionPoint(extractOp);
    for (auto sub : targets.newTypes) {
      auto subValue = builder.create<ExtractOffsetOp>(extractOp.getLoc(), sub.type,
                                      extractOp.getBuffer(),
                                      extractOp.getOffset() + sub.offset);
      values.push_back(subValue);
    }
  }

  // Do per-function translation
  for (auto funcOp : moduleOp.getOps<FuncOp>()) {

    auto &entryBlock = funcOp.getBody().front();

    // do per-block update
    for (auto &block : funcOp.getBody()) {
      auto &context = fieldAnalysis.blockContext[&block];

      DenseMap<Value, SplitMapContext> valueMapping;

      // find if any source value declared in the context
      for (auto &[source, map] : sourceMap) {
        if (context.query(source)) {
          auto [it, _] = valueMapping.try_emplace(source, map);
          for (auto value : replaceMap[source])
            it->second.values.push_back(value);
        }
      }
      // find if any block argument need mapping change
      DenseMap<int, SplitMap> argMapping;
      for (auto [i, arg] : llvm::enumerate(block.getArguments())) {
        if (auto usage = context.query(arg)) {
          auto it = sourceMap.find(usage->source);
          if (it != sourceMap.end()) {
            auto &info = it->second;
            argMapping.try_emplace(i, info);
          }
        }
      }

      // insert new argument and update mapping
      if (&block == &entryBlock)
        updateFunctionSignature(funcOp, argMapping, valueMapping);
      else
        updateBlockSignature(&block, argMapping, valueMapping);

      // per-op updating
      for (auto &op : block) {
        builder.setInsertionPoint(&op);
        // access op
        if (auto accessOp = dyn_cast<StructAccessOp>(op)) {
          auto it = valueMapping.find(accessOp.getInput());
          if (it != valueMapping.end()) {
            auto &ctx = it->second;
            // dispatch to value by index
            auto newStruct = ctx.indexToValue(accessOp.getIndex());
            auto newIndex = ctx.indexToNewIndex(accessOp.getIndex());
            auto newAccess = builder.create<StructAccessOp>(accessOp.getLoc(), newStruct,
                                           newIndex);
            accessOp.replaceAllUsesWith(newAccess.getResult());
            // TODO(check and repalce!)
          }
        }

        // update op
        if (auto accessOp = dyn_cast<StructUpdateOp>(op)) {
          auto it = valueMapping.find(accessOp.getInput());
          if (it != valueMapping.end()) {
            auto &ctx = it->second;
            // dispatch to value by index
            auto newStruct = ctx.indexToValue(accessOp.getIndex());
            auto newIndex = ctx.indexToNewIndex(accessOp.getIndex());
            auto newOp = builder.create<StructUpdateOp>(accessOp.getLoc(), accessOp.getType(), newStruct,
                                           newIndex, accessOp.getNewValue());
            auto [it2, _] = valueMapping.try_emplace(accessOp.getResult(), ctx);
            it2->second.updateValueByIndex(accessOp.getIndex(), newOp);
            // TODO(check and repalce!)
          }
        }

        // TODO(finish the rest of ops)
        // init op
        if (auto initOp = dyn_cast<InitOp>(op)) {
          std::string contextAttr = "context_names";
          // TODO: is a contexted return?
          auto event = initOp.getType().dyn_cast<StructType>();
          if (event && event.getIsEvent() && initOp->hasAttr(contextAttr)) {
            auto operands = llvm::to_vector(initOp.getOperands());
            auto names = llvm::map_to_vector(
                initOp->getAttrOfType<ArrayAttr>(contextAttr)
                    .getAsValueRange<StringAttr>(),
                [](StringRef name) { return name.str(); });
            updateValueList(operands, valueMapping, &names);

            auto nameRefs = llvm::map_to_vector(
                names, [](std::string &name) { return StringRef(name); });
            auto newOp = builder.create<InitOp>(initOp.getLoc(), event.getName(), operands);
            newOp->setAttr("context_names", builder.getStrArrayAttr(nameRefs));

            initOp.replaceAllUsesWith(newOp.getResult());
          }
        }

        // branchOp
        if (auto branchOp = dyn_cast<cf::CondBranchOp>(op)) {
          // TODO(zhiyuang): branchOp
          // We do not have branchOp. do not implement for now
        }

      } // for op in block
    }
  }

  // at last, schedule cse to remove the replaced ops
  OpPassManager pm;
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, moduleOp)))
    return signalPassFailure();

}

} // namespace ep2
} // namespace mlir


