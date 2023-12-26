//===- Ops.td - EP2 dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the EP2 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EP2_OPS
#define EP2_OPS

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def EP2_Dialect : Dialect {
  let name = "ep2";
  let cppNamespace = "::mlir::ep2";

  // We set this bit to generate a declaration of the `materializeConstant`
  // method so that we can materialize constants for our toy operations.
  let hasConstantMaterializer = 1;

  // We set this bit to generate the declarations for the dialect's type parsing
  // and printing hooks.
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let dependentDialects = [
    "::mlir::scf::SCFDialect",
  ];

  // for adding attrs
  let hasRegionArgAttrVerify = 1;
}

// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class EP2_Op<string mnemonic, list<Trait> traits = []> :
    Op<EP2_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

// It's common to define a base classes for attributes in the same dialect. This
// removes the need to pass in the dialect for each attribute, and can also be used
// to define a few fields ahead of time.
class EP2_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<EP2_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// Here is a simple definition of an "integer" attribute, with a type and value parameter.
def ScopeAttr : EP2_Attr<"Scope", "scope"> {
  let summary = "An Attribute containing the scope of given operator";
  let description = [{
    Given the accessing handlers and the partition key
  }];

  let parameters = (ins ArrayRefParameter<"std::string">:$handlers, OptionalParameter<"std::string">:$partitionKey);

  let assemblyFormat = "`<` $handlers `>` `[` $partitionKey `]`";
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

class EP2_BaseType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<EP2_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def EP2_AnyType : EP2_BaseType<"Any", "any"> {
  let summary = "Any type";
  let description = [{
    Any type, represents an unknow type, but allows to be 
    any valid EP2 type.
    Will be eliminated by type inference passes.

    Invalid when converting to lower level dialects.
  }];
}

// Provide a definition for the EP2 StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef. We use `DialectType`
// to demarcate the StructType as belonging to the EP2 dialect.
def EP2_StructType : EP2_BaseType<"Struct", "struct"> {
  let summary = "EP2 Struct Type";
  let description = [{
    "Struct type for EP2"
  }];

  let parameters = (ins AttrParameter<"bool", "isEvent">:$isEvent, ArrayRefParameter<"::mlir::Type", "member types">:$elementTypes, StringRefParameter<>:$name);

  let assemblyFormat = "`<` $name `:` struct($isEvent, $elementTypes)`>`";
  let extraClassDeclaration = [{
    size_t getNumElementTypes() { return getElementTypes().size(); }
  }];
}

def EP2_BufferType: EP2_BaseType<"Buffer", "buf"> {
  let summary = "Buffer type";
  let description = [{
    This is a buffer that contians a pointer to packet data
  }];
}

def EP2_TableType: EP2_BaseType<"Table", "table"> {
  let summary = "Table type";
  let description = [{
    This is a table with a key value look up interface
  }];

  let parameters = (ins "Type":$keyType, "Type":$valueType,  AttrParameter<"int", "size">:$size );
  let assemblyFormat = "`<` $keyType `,` $valueType `,` $size `>` ";
}

def EP2_AtomType: EP2_BaseType<"Atom", "atom"> {
  let summary = "Atom type. Atom with same name are unique";
  let description = [{
    Atoms are symbols that are unique within a dialect.
  }];
}

def EP2_ValueType : AnyTypeOf<[EP2_AnyType, AnyInteger, EP2_StructType, EP2_AtomType]>;

def EP2_ContextType: EP2_BaseType<"Context", "context"> {
  let summary = "Context Type, allows dynamic access to context";
  let description = [{
    All read/write to context will be translated to a load and store
  }];
}

// References
def EP2_ContextRefType: EP2_BaseType<"ContextRef", "conref"> {
  let summary = "Context Type, allows dynamic access to context";
  let description = [{
    All read/write to context will be translated to a load and store
  }];

  let parameters = (ins "Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
}

// Type collections
// Provide a definition of the types that are used within the EP2 dialect.
def EP2_Type : AnyTypeOf<[AnyInteger, EP2_AnyType, EP2_StructType, EP2_BufferType, EP2_TableType,
                          EP2_AtomType, EP2_ContextType, EP2_ContextRefType]>;

def EP2_Integer : AnyTypeOf<[AnyInteger, EP2_AnyType]>;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// BitCast Op
//===----------------------------------------------------------------------===//

def BitCastOp : EP2_Op<"bitcast", [Pure, DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "bitcast operation";
  let description = [{
    The "bitcast" operation converts a value from one type to another.
  }];

  // The bitcast operation takes a single input and returns a single result.
  let arguments = (ins EP2_Type:$input);
  let results = (outs EP2_Type:$output);

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// We define a toy operation by inheriting from our base 'EP2_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def ConstantOp : EP2_Op<"constant",
    [ConstantLike, Pure]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = toy.constant 2 : i64
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins AnyAttrOf<[APIntAttr,StrAttr]>:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs AnyTypeOf<[AnyInteger,EP2_AtomType]>:$output);

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "IntegerAttr":$value), [{
      build($_builder, $_state, $_builder.getI64Type(), value);
    }]>,

    OpBuilder<(ins "StringRef":$atom), [{
      build($_builder, $_state, $_builder.getType<AtomType>(),
            $_builder.getStringAttr(atom));
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "unsigned":$width, "int64_t":$value), [{
      build($_builder, $_state, $_builder.getIntegerType(width),
            $_builder.getIntegerAttr($_builder.getIntegerType(width), value));
    }]>
  ];

  // Indicate that additional verification for this operation is necessary.
  // let hasVerifier = 1;

  // Set the folder bit so that we can implement constant folders.
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// BinaryOps
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : EP2_Op<"add",
    [Pure]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins EP2_Integer:$lhs, EP2_Integer:$rhs);
  let results = (outs EP2_Integer);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def SubOp : EP2_Op<"sub",
    [Pure]> {
  let summary = "element-wise sub operation";
  let description = [{
    The "sub" operation performs element-wise sub between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins EP2_Integer:$lhs, EP2_Integer:$rhs);
  let results = (outs EP2_Integer);
}

//===----------------------------------------------------------------------===//
// Compare Ops
//===----------------------------------------------------------------------===//

def CmpOp : EP2_Op<"cmp",
    [Pure]> {
  let summary = "element-wise compare operation";
  let description = [{
    The "cmp" operation performs compare between values
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins I16Attr: $predicate, EP2_Integer:$lhs, EP2_Integer:$rhs);
  let results = (outs AnyI1);

  let builders = [
    OpBuilder<(ins "char":$pred, "Value":$lhs, "Value":$rhs),
    [{ build($_builder, $_state, $_builder.getI1Type(), $_builder.getI16IntegerAttr(pred), lhs, rhs); }]>
  ];
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

// Add bool to differentiate handlers/controllers.
def FuncOp : EP2_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove, CallableOpInterface
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "toy.func" operation represents a user defined function. These are
    callable SSA-region operations that contain toy computations.

    Example:

    ```mlir
    toy.func @main() {
      %0 = toy.constant dense<5.500000e+00> : tensor<f64>
      %1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      toy.print %1 : tensor<2x2xf64>
      toy.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // CallableOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }

    /// Returns if the function is a handler
    bool isHandler() {
      return getOperation()->getAttrOfType<StringAttr>("type").getValue() == "handler";
    }

    bool isExtern() {
      auto boolAttr = getOperation()->getAttrOfType<BoolAttr>("extern");
      return boolAttr ? boolAttr.getValue() : false;
    }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

def CallOp : EP2_Op<"call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = toy.generate @my_func(%1, %3)
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<EP2_Type>:$inputs);

  // The generic call operation returns a single value of TensorType or
  // StructType.
  let results = (outs EP2_Type);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

//===----------------------------------------------------------------------===//
// ContextRefOp
//===----------------------------------------------------------------------===//

def ContextRefOp : EP2_Op<"context_ref", [Pure,
  DeclareOpInterfaceMethods<PromotableAllocationOpInterface>,
]> {
  let summary = "Get a reference from context";
  let description = [{
    This operator get a reference from context. Similar to struct member access,
    this is a reference operation. The result could be load and stored.
    Or a "lvalue", in a c++ context.
  }];

  let arguments = (ins StrAttr:$name, EP2_ContextType:$context);
  let results = (outs EP2_ContextRefType:$value);
}

def ReRefOp : EP2_Op<"reref", [
  DeclareOpInterfaceMethods<PromotableAllocationOpInterface>,
]> {
  let summary = "Reexport an value as a reference";
  let description = [{
    Reexport a value to a reference, and export to the allocation interface. This is an intermidiate operator, should not be used after the transformation
  }];

  let arguments = (ins EP2_Type:$input);
  let results = (outs EP2_Type:$output);
}

def DeRefOp : EP2_Op<"deref", [
  DeclareOpInterfaceMethods<PromotableMemOpInterface>,
]> {
  let summary = "Load from a reref buffer";
  let description = [{
    Extract from buffer to a variable or struct. Return a new buffer.
  }];

  let arguments = (ins EP2_Type:$input);
  let results = (outs EP2_Type:$output);
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

def ExtractOp : EP2_Op<"extract", [
  DeclareOpInterfaceMethods<PromotableMemOpInterface>,
]> {
  let summary = "extract from buffer to variable";
  let description = [{
    Extract from buffer to a variable or struct. The buffer will be modified;
  }];

  let arguments = (ins EP2_BufferType:$buffer);
  let results = (outs EP2_Type:$output);
  // let hasVerifier = 1;
}

def ExtractValueOp : EP2_Op<"extract_value", [Pure]> {
  let summary = "extract from buffer of value type to variable";
  let description = [{
    Extract from buffer to a variable or struct. Return a new buffer.
  }];

  let arguments = (ins EP2_BufferType:$buffer);
  let results = (outs EP2_BufferType, EP2_Type:$output);
}

//===----------------------------------------------------------------------===//
// EmitOp
//===----------------------------------------------------------------------===//

def EmitOp : EP2_Op<"emit", [
  DeclareOpInterfaceMethods<PromotableMemOpInterface>,
]> {
  let summary = "emit op to buffer";
  let description = [{
    Emit from a variable or struct to a buffer. The buffer will be modified;
    The semantic is to prepend the header to the buffer

  }];

  let arguments = (ins EP2_BufferType:$buffer, EP2_Type:$value);
  // let hasVerifier = 1;
}

def EmitValueOp : EP2_Op<"emit_value", [Pure]> {
  let summary = "emit op to buffer";
  let description = [{
    Emit from a variable or struct to a buffer. Return the new buffer
    The semantic is to prepend the header to the buffer
  }];

  let arguments = (ins EP2_BufferType:$buffer, EP2_Type:$value);
  let results = (outs EP2_BufferType);
}

//===----------------------------------------------------------------------===//
// Table Related Operations
// LookupOp
//===----------------------------------------------------------------------===//

def LookupOp : EP2_Op<"lookup", [Pure]> {
  let summary = "lookup op for table";
  let description = [{
    Lookup from a table using key, return the value.
  }];

  let arguments = (ins EP2_TableType:$table, EP2_Type:$key);
  let results = (outs EP2_Type:$value);

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// UpdateOp
//===----------------------------------------------------------------------===//

def UpdateOp : EP2_Op<"update"> {
  let summary = "update op for table";
  let description = [{
    update a table using key and value.
  }];

  let arguments = (ins EP2_TableType:$table, EP2_Type:$key, EP2_Type:$value);
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Reference Related Operations
// LoadOp & StoreOp & Nop
//===----------------------------------------------------------------------===//

def LoadOp : EP2_Op<"load", [Pure,
  DeclareOpInterfaceMethods<PromotableMemOpInterface>,
]> {
  let summary = "Store variable to Ref";
  let description = [{
    Load an variable to a Ref. The variable could be a struct or a buffer.
  }];

  let arguments = (ins EP2_ContextRefType:$ref);
  let results = (outs EP2_Type:$output);
}

def StoreOp : EP2_Op<"store", [
  DeclareOpInterfaceMethods<PromotableMemOpInterface>,
]> {
  let summary = "Store variable to Ref";
  let description = [{
    Store an variable to a Ref. The variable could be a struct or a buffer.
  }];

  let arguments = (ins EP2_ContextRefType:$output, EP2_Type:$value);
}

def NopOp : EP2_Op<"nop", [Pure]> {
  let summary = "No operation";
  let description = [{
    No operation. Make type checker happy in many cases.
  }];

  let results = (outs NoneType:$output);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, $_builder.getNoneType()); }]>
  ];
}

/// Optimization Ops

def SinkOp : EP2_Op<"sink"> {
  let summary = "Value sink to hold a value and prevent it from being optimized away";
  let description = [{
    Take a list of values. Hold (use) the values.
  }];

  let arguments = (ins Variadic<EP2_Type>:$args);
}

def GuardOp : EP2_Op<"guard", [NoTerminator]> {
  let summary = "Guard a non-pure operation by its predicators";
  let description = [{
    Guard a value by a list of predicates
    If the predicate is false, the value "not be calculated".
    As it do not change anything other than value itself, it is pure.
  }];

  let arguments = (ins Variadic<EP2_Type>:$preds, BoolArrayAttr:$predAttrs);
  let regions = (region SizedRegion<1>:$body);

  // TODO(zhiyuang): verifier: the number of preds and predAttrs should match
  // region verifier should check the number (just one block one op)
}

//===----------------------------------------------------------------------===//
// InitOp
//===----------------------------------------------------------------------===//

def InitOp : EP2_Op<"init", [Pure]> {
  let summary = "Default Initialize an EP2 Value Type";
  let description = [{
    Perform a default initialization of an EP2 Value Type.
    For Value Types, it will perform zero-initilaization.
  }];

  let arguments = (ins Variadic<EP2_Type>:$args);
  let results = (outs EP2_Type:$output);

  let builders = [
    OpBuilder<(ins "Type":$type),
    [{ build($_builder, $_state, type, {}); }]>
  ];
  // let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : EP2_Op<"mul",
    [Pure]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins EP2_Integer:$lhs, EP2_Integer:$rhs);
  let results = (outs EP2_Integer);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def TerminateOp : EP2_Op<"terminate", [Pure, HasParent<"FuncOp">, Terminator]> {
  let summary = "End the current function execution";
  let description = [{
    Note that this function will not return value
  }];
}

def ReturnOp : EP2_Op<"return"> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<EP2_Type>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StructAccessOp & StructUpdateOp
//===----------------------------------------------------------------------===//

def StructAccessOp : EP2_Op<"struct_access", [Pure]> {
  let summary = "struct access";
  let description = [{
    Access the Nth element of a value returning a struct type.
  }];

  let arguments = (ins EP2_StructType:$input, I64Attr:$index);
  let results = (outs EP2_Type:$output);

  let assemblyFormat = [{
    $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
  }];

  // Allow building a StructAccessOp with just a struct value and an index.
  let builders = [
    OpBuilder<(ins "Value":$input, "size_t":$index)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
  // Set the folder bit so that we can fold constant accesses.
  // let hasFolder = 1;
}

def StructUpdateOp : EP2_Op<"struct_update", [Pure]> {
  let summary = "struct update";
  let description = [{
    update an element in a struct, like LLVM gep
  }];

  let arguments = (ins EP2_StructType:$input, I64Attr:$index, EP2_Type:$newValue);
  let results = (outs EP2_StructType:$output);

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// StructConstantOp
//===----------------------------------------------------------------------===//

def StructConstantOp : EP2_Op<"struct_constant", [ConstantLike, Pure]> {
  let summary = "struct constant";
  let description = [{
    Constant operation turns a literal struct value into an SSA value. The data
    is attached to the operation as an attribute. The struct constant is encoded
    as an array of other constant values. For example:

    ```mlir
      %0 = toy.struct_constant [
        dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
      ] : !toy.struct<tensor<*xf64>>
    ```
  }];

  let arguments = (ins ArrayAttr:$value);
  let results = (outs EP2_StructType:$output);

  let assemblyFormat = "$value attr-dict `:` type($output)";

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
  // let hasFolder = 1;
}

#endif // TOY_OPS
