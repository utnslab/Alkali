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

}

// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class EP2_Op<string mnemonic, list<Trait> traits = []> :
    Op<EP2_Dialect, mnemonic, traits>;


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

  let parameters = (ins ArrayRefParameter<"::mlir::Type", "member types">:$elementTypes, StringRefParameter<>:$name);

  // default storage class detail
  let genStorageClass = 0;
  let storageClass = "StructTypeStorage";

  let assemblyFormat = "`<` $name `:` struct($elementTypes)`>`";
  let extraClassDeclaration = [{
    size_t getNumElementTypes() { return getElementTypes().size(); }
  }];

  let extraClassDefinition = [{
    // define the get element types method, should be after invocation
    ArrayRef<Type> $cppClass::getElementTypes() const {
      return getImpl()->elementTypes;
    }
    StringRef $cppClass::getName() const {
      return getImpl()->name;
    }
  }];
}

def EP2_BufferType: EP2_BaseType<"Buffer", "buf"> {
  let summary = "Buffer type";
  let description = [{
    This is a buffer that contians a pointer to packet data
  }];
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

def EP2_ContextRefType: EP2_BaseType<"ContextRef", "conref"> {
  let summary = "Context Type, allows dynamic access to context";
  let description = [{
    All read/write to context will be translated to a load and store
  }];

  let parameters = (ins "Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
}

// Provide a definition of the types that are used within the EP2 dialect.
def EP2_Type : AnyTypeOf<[AnyInteger, EP2_StructType, EP2_BufferType,
                          EP2_AtomType, EP2_ContextType, EP2_ContextRefType]>;


//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// We define a toy operation by inheriting from our base 'EP2_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def ConstantOp : EP2_Op<"constant",
    [Pure]> {
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
  let arguments = (ins AnyAttrOf<[I64Attr,StrAttr]>:$value);

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
    OpBuilder<(ins "int":$width, "unsigned long":$value)>
  ];

  // Indicate that additional verification for this operation is necessary.
  // let hasVerifier = 1;

  // Set the folder bit so that we can implement constant folders.
  // let hasFolder = 1;
}

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

  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : EP2_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
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
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
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

def ContextRefOp : EP2_Op<"context_ref", [Pure]> {
  let summary = "Get a reference from context";
  let description = [{
    This operator get a reference from context. Similar to struct member access,
    this is a reference operation. The result could be load and stored.
    Or a "lvalue", in a c++ context.
  }];

  let arguments = (ins StrAttr:$name, EP2_ContextType:$context);
  let results = (outs EP2_ContextRefType:$value);

  // let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

def ExtractOp : EP2_Op<"extract", [Pure]> {
  let summary = "extract from buffer to variable";
  let description = [{
    Extract from buffer to a variable or struct. The buffer will be modified;
  }];

  let arguments = (ins EP2_BufferType:$buffer);
  let results = (outs EP2_Type:$output);
  // let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LoadOp & StoreOp & Nop
//===----------------------------------------------------------------------===//

def StoreOp : EP2_Op<"store"> {
  let summary = "Store variable to Ref";
  let description = [{
    Store an variable to a Ref. The variable could be a struct or a buffer.
  }];

  let arguments = (ins EP2_ContextRefType:$output, EP2_Type:$value);
  // let hasVerifier = 1;
}

def NopOp : EP2_Op<"nop"> {
  let summary = "No operation";
  let description = [{
    No operation. Make type checker happy in many cases.
  }];

  let results = (outs NoneType:$output);
  // let hasVerifier = 1;
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

  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs F64Tensor);

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

def ReturnOp : EP2_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
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
// StructAccessOp
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
