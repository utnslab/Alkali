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

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/CommonFolders.h"

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

mlir::Value updateValue(mlir::PatternRewriter &rewriter, mlir::Value oldValue,
                        mlir::Type type) {
  // we could only update integer value
  if (!isa<IntegerType>(oldValue.getType()) || !isa<IntegerType>(type))
    return oldValue;
  if (oldValue.getType().getIntOrFloatBitWidth() ==
      type.getIntOrFloatBitWidth())
    return oldValue;
  return rewriter.create<ep2::BitCastOp>(oldValue.getLoc(), type, oldValue);
}

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
// BitCastOp
//===----------------------------------------------------------------------===//

bool BitCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  if (isa<ep2::AnyType>(inputs[0]) || isa<ep2::AnyType>(outputs[0]))
    return true;
  if (isa<IntegerType>(inputs[0]) && isa<IntegerType>(outputs[0]))
    return true;
  if (inputs[0] == outputs[0])
    return true;
  return false;
}

OpFoldResult BitCastOp::fold(BitCastOp::FoldAdaptor adaptor) {
  // cast on identical
  if (getInput().getType() == getType())
    return getInput();
  if (getType().isa<ep2::AnyType>())
    return getInput();
  if (isa<IntegerType>(getInput().getType()) && isa<IntegerType>(getType())) {
    auto width = getType().dyn_cast<IntegerType>().getWidth();
    return constFoldCastOp<IntegerAttr, IntegerAttr, APInt, APInt, void>(
        adaptor.getOperands(), getType(), [&](const APInt &value, bool status) {
          return APInt(width, value.getLimitedValue());
        });
  }
}

// BitCast based type inference
class BitCastTypeInfer : public OpRewritePattern<BitCastOp> {
  using OpRewritePattern<BitCastOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BitCastOp op,
                                PatternRewriter &rewriter) const override {
    // fold input
    if (op.getType().isa<ep2::AnyType>()) {
      rewriter.replaceOp(op, ValueRange{op.getInput()});
      return success();
    }

    // fold output
    auto inType = op.getInput().getType();
    if (!inType.isa<ep2::AnyType>())
      return rewriter.notifyMatchFailure(op, "input type must be AnyType");
    
    op.getInput().setType(op.getType());
    op.replaceAllUsesWith(op.getInput());
    rewriter.eraseOp(op);
    return success();
  }
};

void BitCastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<BitCastTypeInfer>(context);
}


//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Build a constant operation.
/// The builder is passed as an argument, so is the state that this method is
/// expected to fill in order to build the operation.

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

OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
  return adaptor.getValue();
}

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

struct GloablImportDedupPattern : public OpRewritePattern<FuncOp> {
  using OpRewritePattern<FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    llvm::DenseMap<llvm::StringRef, GlobalImportOp> imports;
    llvm::SmallVector<GlobalImportOp> toErase;

    rewriter.startRootUpdate(op);
    for (auto &op : op.getOps()) {
      if (auto importOp = dyn_cast<GlobalImportOp>(op)) {
        auto [it, isInsert] = imports.try_emplace(importOp.getName(), importOp);
        if (!isInsert) {
          rewriter.replaceAllUsesWith(importOp, it->second);
          toErase.push_back(importOp);
          changed = true;
        }
      }
    }

    for (auto op : toErase)
      rewriter.eraseOp(op);

    if (changed) {
      rewriter.finalizeRootUpdate(op);
      return success();
    } else {
      rewriter.cancelRootUpdate(op);
      return failure();
    }
  }
};


void FuncOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<GloablImportDedupPattern>(context);
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
  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Emit Op and Extract Op Type Inference - through cast op infer
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// StructAccessOp && StructUpdateOp
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

// mlir::LogicalResult StructAccessOp::verify() {
//   StructType structTy = llvm::cast<StructType>(getInput().getType());
//   size_t indexValue = getIndex();
//   if (indexValue >= structTy.getNumElementTypes())
//     return emitOpError()
//            << "index should be within the range of the input struct type";
//   mlir::Type resultType = getResult().getType();
//   if (resultType != structTy.getElementTypes()[indexValue])
//     return emitOpError() << "must have the same result type as the struct "
//                             "element referred to by the index";
//   return mlir::success();
// }

class StructUpdateTypeInfer : public OpRewritePattern<StructUpdateOp> {
  using OpRewritePattern<StructUpdateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(StructUpdateOp op,
                                PatternRewriter &rewriter) const override {
    auto type = op.getInput().getType().getElementTypes()[op.getIndex()];
    auto newValue = updateValue(rewriter, op.getNewValue(), type);
    if (newValue == op.getNewValue())
      return failure();
    rewriter.replaceOpWithNewOp<StructUpdateOp>(op, op.getType(), op.getInput(), op.getIndex(),
                                                newValue);
    return success();
  }
};

void StructUpdateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<StructUpdateTypeInfer>(context);
}

//===----------------------------------------------------------------------===//
// TableLookup & Update
//===----------------------------------------------------------------------===//

class UpdateTypeInfer : public OpRewritePattern<UpdateOp> {
  using OpRewritePattern<UpdateOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(UpdateOp op,
                                PatternRewriter &rewriter) const override {
    auto tableType = op.getTable().getType();
    auto newKey = updateValue(rewriter, op.getKey(), tableType.getKeyType());
    auto newValue = updateValue(rewriter, op.getValue(), tableType.getValueType());
    if (newKey == op.getKey() && newValue == op.getValue())
      return failure();
    rewriter.replaceOpWithNewOp<UpdateOp>(op, op.getTable(), newKey, newValue);
    return success();
  }
};

class LookupTypeInfer : public OpRewritePattern<LookupOp> {
  using OpRewritePattern<LookupOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LookupOp op,
                                PatternRewriter &rewriter) const override {
    auto tableType = op.getTable().getType();
    auto newKey = updateValue(rewriter, op.getKey(), tableType.getKeyType());
    if (newKey == op.getKey())
      return failure();
    rewriter.replaceOpWithNewOp<LookupOp>(op, op.getType(), op.getTable(), newKey);
    return success();
  }
};

void LookupOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<LookupTypeInfer>(context);
}

void UpdateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<UpdateTypeInfer>(context);
}



//===----------------------------------------------------------------------===//
// ep2 Types
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TableGen'd attr and type definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "ep2/dialect/EP2OpsAttrDefs.cpp.inc"

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
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ep2/dialect/EP2OpsAttrDefs.cpp.inc"
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

LogicalResult EP2Dialect::verifyRegionArgAttribute(Operation *op,
                                                   unsigned regionIndex,
                                                   unsigned argIndex,
                                                   NamedAttribute attribute) {
  // currently we allow all success()
  return success();
}
