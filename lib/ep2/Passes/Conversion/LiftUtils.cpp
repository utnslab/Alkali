#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "ep2/passes/LiftUtils.h"
#include <optional>

namespace mlir {
namespace ep2 {

static constexpr std::string_view kAnnoStructName{"_anno_struct_"};

// helper function to type convert from flat llvm struct to ep2 struct 
Type liftLLVMType(OpBuilder &builder, Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<LLVM::LLVMPointerType>([&](LLVM::LLVMPointerType ptrType) {
        llvm_unreachable("LLVM pointer type not supported");
        return ptrType;
      })
      .Case<LLVM::LLVMStructType>([&](LLVM::LLVMStructType structType) {
        // TODO(zhiyuang): we should handle size annotations
        auto liftedTypes = llvm::map_to_vector(
            structType.getBody(), [&](Type type) { return liftLLVMType(builder, type); });
        return builder.getType<ep2::StructType>(false, liftedTypes, kAnnoStructName);
      })
      .Default([&](Type type) { return type; });
}

std::optional<ep2::TableType> tryParseTable(OpBuilder& builder, LLVM::LLVMStructType type) {
  auto types = type.getBody();

  if (types.size() < 4)
    return std::nullopt;
  
  auto magicType = types[0].dyn_cast<LLVM::LLVMArrayType>();
  if (!(magicType && magicType.getNumElements() == 9321))
    return std::nullopt;
  
  auto keyType = liftLLVMType(builder, types[1]);
  auto valueType = liftLLVMType(builder, types[2]);

  auto sizeType = types[3].dyn_cast<LLVM::LLVMArrayType>();
  if (!(sizeType && sizeType.getElementType().isa<IntegerType>()))
    return std::nullopt;
  auto size = sizeType.getNumElements();

  // TODO(zhiyuang): further parse other signatures
  return builder.getType<ep2::TableType>(keyType, valueType, size);
}

std::optional<Type> stripMemRefType(OpBuilder &builder, Type type) {
  auto memRefType = type.dyn_cast<MemRefType>();
  if (!memRefType)
    return std::nullopt;
  auto elementType = memRefType.getElementType();
  auto dims = memRefType.getShape();
  
  // this is a pointer type.
  if (dims.size() == 1 && dims[0] == mlir::ShapedType::kDynamic) {
    if (auto structType = dyn_cast<LLVM::LLVMStructType>(elementType)) {
      // we got a pointer to struct ...
      // we only handle the struct with our type signature

      auto tableType = tryParseTable(builder, structType);
      // TODO(zhiyuang): for non table type, do we want to parse them as a struct?
      if (!tableType.has_value())
        llvm_unreachable("Unsupported struct type");

      return tableType;
    }
  } else if (dims.size() == 2 && dims[0] == 1) {
    // This is a specialized struct type
    // like memref<1x6xi32>
    llvm::SmallVector<Type> flatArr{(size_t)dims[1], elementType};
    return builder.getType<ep2::StructType>(false, flatArr, kAnnoStructName);
  } else if (dims.size() == 1 && dims[0] == 1) {
    // it's a nested scalar..
    if (isa<MemRefType>(elementType))
      return stripMemRefType(builder, elementType);
    // This is a scalar value
    return elementType;
  }

  return std::nullopt;
}

} // namespace ep2
} // namespace mlir
