
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace ep2 {

ExtrEmitOffsetAnalysis::ExtrEmitOffsetAnalysis(Operation* module, AnalysisManager& am) {
  auto calcSize = [&](mlir::Type ty){
    if (isa<mlir::IntegerType>(ty)) {
      assert(ty.isIntOrFloat());
      return ty.getIntOrFloatBitWidth()/8;
    } else if (isa<ep2::StructType>(ty)) {
      unsigned size = 0;
      for (auto ety : cast<ep2::StructType>(ty).getElementTypes()) {
        assert(ety.isIntOrFloat());
        size += ety.getIntOrFloatBitWidth()/8;
      }
      return size;
    }
  };

  module->walk([&](FuncOp fop){
    unsigned extrOffset = 0;
    unsigned emitOffset = 0;

    // assume they walk in order of insertion. definetly no guarantees... :(
    fop->walk<WalkOrder::PreOrder>([&](ExtractOp op){
      mlir::Type type = op->getResultTypes()[0];
      std::pair<unsigned, int> pr = {extrOffset, calcSize(type)};
      extrOffsets.emplace(op, pr);
      extrOffset += pr.second;
    });
    fop->walk<WalkOrder::PreOrder>([&](EmitOp op){
      mlir::Type ty = op->getOperand(1).getType();
      if (isa<ep2::StructType>(ty)) {
        std::pair<unsigned, int> pr = {emitOffset, calcSize(ty)};
        emitOffsets.emplace(op, pr);
        emitOffset += pr.second;
      } else if (isa<ep2::BufferType>(ty)) {
        std::pair<unsigned, int> pr = {emitOffset, -extrOffset};
        emitOffsets.emplace(op, pr);
        emitOffset += pr.second;
      } else {
        assert(false && "Unsupported type");
      }
    });
  });
}

}
}
