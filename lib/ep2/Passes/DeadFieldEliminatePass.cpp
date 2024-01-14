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

struct StructMap {
    StructType type;
    unsigned size;
    unsigned offset;
};

struct SplitMap {
    std::map<int, int> fieldToIndex;
    SmallVector<StructMap> newTypes;
    SmallVector<Value> newValues;
};

};

void DeadFieldEliminatePass::runOnOperation() {
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

      for (auto [i, field] : llvm::enumerate(usage.type.getElementTypes())) {
        if (usage.used[i]) {
          // we collect the fields
          map.fieldToIndex.try_emplace(i, map.newTypes.size());
          fields.push_back(field);
        } else if (!fields.empty()) {
          // or if we have some value collected..
          // TODO(zhiyuang): fixname
          auto structType = builder.getType<StructType>(
              false, fields,
              usage.type.getName() + std::to_string(map.newTypes.size()));
          map.newTypes.push_back(structType);
          fields.clear();
        }
      }
    }
  }

  // replace the values on the original ops and usages
}

} // namespace ep2
} // namespace mlir


