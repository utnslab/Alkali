
#include "mlir/IR/BuiltinDialect.h"

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

/*  Optimizations:
1) Cache table lookup value in a register, not memory
2) Verify no cls_workqueue or memory ops in critical region
3) Lower no_swap_begin/no_swap_end to netronome intrinsics
*/

void LowerNoctxswapPass::runOnOperation() {
}

} // namespace ep2
} // namespace mlir
