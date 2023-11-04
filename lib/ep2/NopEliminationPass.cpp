#include "mlir/IR/BuiltinDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace mlir {
namespace ep2 {

void NopEliminationPass::runOnOperation() {
    auto module = getOperation();
    // Fill
}

std::unique_ptr<Pass> createNopEliminationPass() {
    return std::make_unique<NopEliminationPass>();
}

}
}


