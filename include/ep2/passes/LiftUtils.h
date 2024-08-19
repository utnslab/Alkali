#ifndef EP2_PASSES_LIFTUTILS_H
#define EP2_PASSES_LIFTUTILS_H

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinOps.h"
#include <optional>

namespace mlir {
namespace ep2 {

std::optional<Type> stripMemRefType(OpBuilder &builder, Type type);
Type liftLLVMType(OpBuilder &builder, Type type);

} // namespace ep2
} // namespace mlir


#endif // EP2_PASSES_LIFTUTILS_H