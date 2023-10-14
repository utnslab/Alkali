//===- Dialect.cpp - ep2 IR Dialect registration in MLIR ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect for the ep2 IR: custom type parsing and
// operation verification.
//
//===----------------------------------------------------------------------===//

#include "ep2/dialect/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::ep2;

#include "ep2/dialect/EP2OpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// ep2InlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with ep2
/// operations.
struct ep2InlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// All call operations within ep2 can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations within ep2 can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // All functions within ep2 can be inlined.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator(ep2.return) by replacing it with a new
  /// operation as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    // Only "ep2.return" needs to be handled here.
    auto returnOp = cast<ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};

//===----------------------------------------------------------------------===//
// ep2 Operations
//===----------------------------------------------------------------------===//

/// A generalized parser for binary operations. This parses the different forms
/// of 'printBinaryOp' below.
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                               result.operands))
      return mlir::failure();
    result.addTypes(funcType.getResults());
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  if (parser.resolveOperands(operands, type, result.operands))
    return mlir::failure();
  result.addTypes(type);
  return mlir::success();
}

/// A generalized printer for binary operations. It prints in two different
/// forms depending on if all of the types match.
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
  printer << " " << op->getOperands();
  printer.printOptionalAttrDict(op->getAttrs());
  printer << " : ";

  // If all of the types are the same, print the type directly.
  Type resultType = *op->result_type_begin();
  if (llvm::all_of(op->getOperandTypes(),
                   [=](Type type) { return type == resultType; })) {
    printer << resultType;
    return;
  }

  // Otherwise, print a functional type.
  printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       int width, unsigned long value) {
  auto dataAttribute = builder.getI64IntegerAttr(value);
  ConstantOp::build(builder, state, builder.getI64Type(), dataAttribute);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}

/// Verify that the given attribute value is valid for the given type.
static mlir::LogicalResult verifyConstantForType(mlir::Type type,
                                                 mlir::Attribute opaqueValue,
                                                 mlir::Operation *op) {
  if (llvm::isa<mlir::TensorType>(type)) {
    // Check that the value is an elements attribute.
    auto attrValue = llvm::dyn_cast<mlir::DenseFPElementsAttr>(opaqueValue);
    if (!attrValue)
      return op->emitError("constant of TensorType must be initialized by "
                           "a DenseFPElementsAttr, got ")
             << opaqueValue;

    // If the return type of the constant is not an unranked tensor, the shape
    // must match the shape of the attribute holding the data.
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(type);
    if (!resultType)
      return success();

    // Check that the rank of the attribute type matches the rank of the
    // constant result type.
    auto attrType = llvm::cast<mlir::RankedTensorType>(attrValue.getType());
    if (attrType.getRank() != resultType.getRank()) {
      return op->emitOpError("return type must match the one of the attached "
                             "value attribute: ")
             << attrType.getRank() << " != " << resultType.getRank();
    }

    // Check that each of the dimensions match between the two types.
    for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
      if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
        return op->emitOpError(
                   "return type shape mismatches its attribute at dimension ")
               << dim << ": " << attrType.getShape()[dim]
               << " != " << resultType.getShape()[dim];
      }
    }
    return mlir::success();
  }
  auto resultType = llvm::cast<StructType>(type);
  llvm::ArrayRef<mlir::Type> resultElementTypes = resultType.getElementTypes();

  // Verify that the initializer is an Array.
  auto attrValue = llvm::dyn_cast<ArrayAttr>(opaqueValue);
  if (!attrValue || attrValue.getValue().size() != resultElementTypes.size())
    return op->emitError("constant of StructType must be initialized by an "
                         "ArrayAttr with the same number of elements, got ")
           << opaqueValue;

  // Check that each of the elements are valid.
  llvm::ArrayRef<mlir::Attribute> attrElementValues = attrValue.getValue();
  for (const auto it : llvm::zip(resultElementTypes, attrElementValues))
    if (failed(verifyConstantForType(std::get<0>(it), std::get<1>(it), op)))
      return mlir::failure();
  return mlir::success();
}

/// Verifier for the constant operation. This corresponds to the `::verify(...)`
/// in the op definition.

mlir::LogicalResult StructConstantOp::verify() {
  return verifyConstantForType(getResult().getType(), getValue(), *this);
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the AddOp, this is required by the shape inference
/// interface.

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided utility method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

void CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable CallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range CallOp::getArgOperands() { return getInputs(); }

/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
MutableOperandRange CallOp::getArgOperandsMutable() {
  return getInputsMutable();
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter &p) { printBinaryOp(p, *this); }

/// Infer the output shape of the MulOp, this is required by the shape inference
/// interface.

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult ReturnOp::verify() {
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = cast<FuncOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand())
    return mlir::success();

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType || llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resultType))
    return mlir::success();

  return emitError() << "type of return operand (" << inputType
                     << ") doesn't match function result type (" << resultType
                     << ")";
}

//===----------------------------------------------------------------------===//
// StructAccessOp
//===----------------------------------------------------------------------===//

void StructAccessOp::build(mlir::OpBuilder &b, mlir::OperationState &state,
                           mlir::Value input, size_t index) {
  // Extract the result type from the input type.
  StructType structTy = llvm::cast<StructType>(input.getType());
  assert(index < structTy.getNumElementTypes());
  mlir::Type resultType = structTy.getElementTypes()[index];

  // Call into the auto-generated build method.
  build(b, state, resultType, input, b.getI64IntegerAttr(index));
}

mlir::LogicalResult StructAccessOp::verify() {
  StructType structTy = llvm::cast<StructType>(getInput().getType());
  size_t indexValue = getIndex();
  if (indexValue >= structTy.getNumElementTypes())
    return emitOpError()
           << "index should be within the range of the input struct type";
  mlir::Type resultType = getResult().getType();
  if (resultType != structTy.getElementTypes()[indexValue])
    return emitOpError() << "must have the same result type as the struct "
                            "element referred to by the index";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ep2 Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace ep2 {
namespace detail {
/// This class represents the internal storage of the ep2 `StructType`.
struct StructTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /// A constructor for the type storage instance.
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<mlir::Type> elementTypes;
};
} // namespace detail
} // namespace ep2
} // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "ep2/dialect/EP2OpsTypes.cpp.inc"


//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ep2/dialect/EP2Ops.cpp.inc"

/// Create an instance of a `StructType` with the given element types. There
/// *must* be at least one element type.
// StructType StructType::get(llvm::ArrayRef<mlir::Type> elementTypes) {
//   assert(!elementTypes.empty() && "expected at least 1 element type");

//   // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
//   // of this type. The first parameter is the context to unique in. The
//   // parameters after the context are forwarded to the storage instance.
//   mlir::MLIRContext *ctx = elementTypes.front().getContext();
//   return Base::get(ctx, elementTypes);
// }

//===----------------------------------------------------------------------===//
// ep2Dialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void EP2Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ep2/dialect/EP2Ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "ep2/dialect/EP2OpsTypes.cpp.inc"
      >();
  // addInterfaces<ep2InlinerInterface>();
  // addTypes<StructType>();
}

mlir::Operation *EP2Dialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  if (llvm::isa<StructType>(type))
    return builder.create<StructConstantOp>(loc, type,
                                            llvm::cast<mlir::ArrayAttr>(value));
  return builder.create<ConstantOp>(loc, type,
                                    llvm::cast<mlir::IntegerAttr>(value));
}
