//===- Dialect.h - Dialect definition for the Toy IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the Toy language.
// See docs/Tutorials/Toy/Ch-2.md for more information.
//
//===----------------------------------------------------------------------===//

#ifndef EP2_DIALECT_H_
#define EP2_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace ep2 {
namespace detail {
struct StructTypeStorage;
} // namespace detail
} // namespace toy
} // namespace mlir

/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "ep2/dialect/EP2OpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// EP2 Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "ep2/dialect/EP2OpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// EP2 Attrs
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "ep2/dialect/EP2OpsAttrDefs.h.inc"

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "ep2/dialect/EP2Ops.h.inc"

#endif // EP2_DIALECT_H_
