#ifndef EP2_PASSES_LIFTUTILS_H
#define EP2_PASSES_LIFTUTILS_H

#include "ep2/dialect/Dialect.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinOps.h"
#include <optional>

namespace mlir {
namespace ep2 {

std::optional<Type> stripMemRefType(OpBuilder &builder, Type type);
Type liftLLVMType(OpBuilder &builder, Type type);

void functionSplitter(ep2::FuncOp funcOp, llvm::DenseSet<Operation *> &sinkOps, llvm::DenseSet<Value> &sinkArgs);
std::pair<ep2::InitOp, ep2::ReturnOp> createGenerate(OpBuilder &builder, mlir::Location loc, StringRef name, ArrayRef<Value> values);

} // namespace ep2
} // namespace mlir


#endif // EP2_PASSES_LIFTUTILS_H