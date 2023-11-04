#include "mlir/IR/BuiltinDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace mlir {
namespace ep2 {

namespace {

// Conversion Patterns

// struct ContextRefOpLowering : public OpConversionPattern<ContextRefOp> {
//     using OpConversionPattern<ContextRefOp>::OpConversionPattern;

//     LogicalResult matchAndRewrite(ContextRefOp op, OpAdaptor adaptor,
//                                   ConversionPatternRewriter &rewriter) const final {
//         auto loc = op.getLoc();
//         auto context = op.getContext();

//         // create a call to 
//         auto contextTy = LLVM::LLVMType::getInt8PtrTy(context);
//         auto contextVal = rewriter.create<LLVM::AddressOfOp>(loc, contextTy, op.context());
//         rewriter.replaceOp(op, contextVal);

//         return success();
//     }
// };

// struct ReturnRefOpLowering : public OpConversionPattern<ReturnOp> {
//     using OpConversionPattern<ReturnOp>::OpConversionPattern;

//     LogicalResult matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
//                                   ConversionPatternRewriter &rewriter) const final {
//         auto loc = op.getLoc();
//         auto context = op.getContext();

//         // craete a struct and push it to a function.
//         rewriter.create<LLVM::CallOp>(loc, TypeRange{}, "_ep2_rt_enqueue");
//         rewriter.eraseOp(op);

//         return success();
//     }

// };

} // hold the patterns

// Conversion Pass

void LowerToLLVMPass::runOnOperation() {
    auto module = getOperation();
    auto &context = getContext();

    ConversionTarget target(getContext());

    // legal Ops
    target.addLegalDialect<BuiltinDialect,
                           LLVM::LLVMDialect, memref::MemRefDialect>();

    target.addIllegalDialect<EP2Dialect>();

    // install rewrite patterns
    RewritePatternSet patterns(&getContext());

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}

}
}