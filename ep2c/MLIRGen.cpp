//===- MLIRGen.cpp - MLIR Generation from a EP2 AST -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the ep2 language.
//
//===----------------------------------------------------------------------===//

#include "ep2/dialect/MLIRGen.h"
#include "ep2/lang/AST.h"
#include "ep2/dialect/Dialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>
#include <optional>

using namespace mlir::ep2;
using namespace ep2;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the ep2 AST.
///
/// This will emit operations that are specific to the ep2 language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {
    // init processes. Load pre-defined functions.
    auto I64Type = builder.getI64Type();
    // Queue function
    functionMap["Queue"] = builder.create<FuncOp>(builder.getUnknownLoc(),
      "Queue",
      builder.getFunctionType({I64Type, I64Type, I64Type}, {I64Type})
    );
    // print function
    functionMap["print"] = builder.create<FuncOp>(builder.getUnknownLoc(),
      "print",
      builder.getFunctionType({}, {})
    );
  }

  /// Public API: convert the AST for a ep2 module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // Create a global scope to hold global variables
    SymbolTableScopeT globalScope(symbolTable);

    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    this->modAST = &moduleAST;

    std::map<std::string, ScopeAttr> scopes;

    for (auto &record : moduleAST) {
      mlir::Operation *op = nullptr;
      if (FunctionAST *funcAST = llvm::dyn_cast<FunctionAST>(record.get())) {
        mlir::ep2::FuncOp func = mlirGen(*funcAST);
        op = func;

        if (!func)
          return nullptr;
        functionMap.insert({func.getName(), func});
      } else if (StructAST *str = llvm::dyn_cast<StructAST>(record.get())) {
        if (failed(mlirGen(*str)))
          return nullptr;

      } else if (ScopeAST *scope = llvm::dyn_cast<ScopeAST>(record.get())) {
        auto scopeAttr = builder.getAttr<ScopeAttr>(
            scope->getHandlers(), scope->getPartitionKey().str());
        scopes.insert_or_assign(scope->getName().str(), scopeAttr);

      } else if (GlobalAST *globalAST = llvm::dyn_cast<GlobalAST>(record.get())) {
        auto &decl = globalAST->getDecl();

        builder.setInsertionPointToEnd(theModule.getBody());
        auto initOp = builder.create<InitOp>(
            loc(decl.loc()), getVarType(decl.getType(), decl.loc()));
        op = initOp;
        // ok to just insert at top most level, as a global ref
        if (declare(decl, initOp).failed()) {
          emitError(loc(decl.loc())) << "error: global variable with name `"
                                     << decl.getName() << "' already exists";
          return nullptr;
        }
      } else {
        llvm_unreachable("unknown record type");
      }

      if (op)
        for (auto &[k, v]: record->getAttributes()) {
          if (v.empty())
            op->setAttr(k, builder.getBoolAttr(true));
          else if (k == "sync") { // process scope attribute
            auto scopeAttr = scopes.find(v);
            if (scopeAttr == scopes.end()) {
              emitError(op->getLoc()) << "error: scope with name `"
                                            << v << "' does not exist";
              return nullptr;
            }
            op->setAttr("scope", scopeAttr->second);
          } else
            op->setAttr(k, builder.getStringAttr(v));
        }
    }

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the ep2 operations.
    // if (failed(mlir::verify(theModule))) {
    //   theModule.emitError("module verification error");
    //   return nullptr;
    // }

    return theModule;
  }

private:
  /// A "module" matches a ep2 source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, std::pair<mlir::Value, VarDeclExprAST *>>
      symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<StringRef,
                                 std::pair<mlir::Value, VarDeclExprAST *>>;

  /// TODO(zhiyuang): nested if else?
  using UpdateTableT = std::map<StringRef, mlir::Value>;
  std::unique_ptr<UpdateTableT> updateTable = nullptr;

  /// A mapping for the functions that have been code generated to MLIR.
  ModuleAST* modAST;

  /// A mapping for the functions that have been code generated to MLIR.
  llvm::StringMap<mlir::ep2::FuncOp> functionMap;

  /// A mapping for named struct types to the underlying MLIR type and the
  /// original AST node.
  llvm::StringMap<std::pair<mlir::Type, StructAST *>> structMap;

  /// Helper conversion for a ep2 AST location to an MLIR location.
  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(VarDeclExprAST &var, mlir::Value value) {
    if (symbolTable.count(var.getName()))
      return mlir::failure();
    symbolTable.insert(var.getName(), {value, &var});
    return mlir::success();
  }

  // Update the symbol table with the new value
  void update(ExprAST &targetAst, mlir::Value value) {
      if (targetAst.getKind() == ExprAST::Expr_Path) {
        auto &path = cast<PathExprAST>(targetAst);

        // could be a variable or a struct access
        auto varName = path.getPath()[0]->getName();
        auto [_, decl] = symbolTable.lookup(varName);
        symbolTable.insert(varName, {value, decl});
        if (updateTable)
          updateTable->insert_or_assign(varName, value);
      } else if (targetAst.getKind() == ExprAST::Expr_Var) {
        auto &var = cast<VariableExprAST>(targetAst);
        auto varName = var.getName();
        auto [_, decl] = symbolTable.lookup(varName);
        symbolTable.insert(varName, {value, decl});
        if (updateTable)
          updateTable->insert_or_assign(varName, value);
      }
  }

  /// ===========
  ///  mlirGen functions. Per-node code gen
  /// ===========
  /// Create an MLIR type for the given struct.
  mlir::LogicalResult mlirGen(StructAST &str) {
    if (structMap.count(str.getName()))
      return emitError(loc(str.loc())) << "error: struct type with name `"
                                       << str.getName() << "' already exists";

    auto variables = str.getVariables();
    std::vector<mlir::Type> elementTypes;
    elementTypes.reserve(variables.size());
    for (auto &variable : variables) {
      if (variable->getInitVal())
        return emitError(loc(variable->loc()))
               << "error: variables within a struct definition must not have "
                  "initializers";
      // TODO: why check shape here?

      mlir::Type type = getVarType(variable->getType(), variable->loc());
      if (!type)
        return mlir::failure();
      elementTypes.push_back(type);
    }

    structMap.try_emplace(str.getName(),
      mlir::ep2::StructType::get(builder.getContext(), str.isEvent(), elementTypes, str.getName().str()),
      &str);
    return mlir::success();
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided ep2 AST prototype.
  mlir::ep2::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    llvm::SmallVector<mlir::Type, 4> argTypes;
    argTypes.reserve(proto.getArgs().size());
    for (auto &arg : proto.getArgs()) {
      mlir::Type type = getVarType(arg->getType(), arg->loc());
      if (!type)
        return nullptr;
      argTypes.push_back(type);
    }
    auto funcType = builder.getFunctionType(argTypes, std::nullopt);
    auto funcOp = builder.create<mlir::ep2::FuncOp>(location, proto.getMangledName(),
                                             funcType);
    
    // Set Attrs
    // TODO: change attrs to fields
    funcOp->setAttr("event", builder.getStringAttr(proto.getName()));
    if (proto.getAtom())
      funcOp->setAttr("atom", builder.getStringAttr(*proto.getAtom()));
    funcOp->setAttr("type", builder.getStringAttr(proto.getFunctionTypeName()));

    return funcOp;
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::ep2::FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    SymbolTableScopeT varScope(symbolTable);

    

    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    mlir::ep2::FuncOp function = mlirGen(*funcAST.getProto());
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    mlir::Block &entryBlock = function.front();
    auto protoArgs = funcAST.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(*std::get<0>(nameValue), std::get<1>(nameValue))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    TerminateOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<TerminateOp>(entryBlock.back());
    if (!returnOp) {
      builder.create<TerminateOp>(loc(funcAST.getProto()->loc()));
    }

    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main")
      function.setPrivate();

    return function;
  }

  StructAST *getStructForName(StringRef name) {
    llvm::StringRef structName;
    auto varIt = symbolTable.lookup(name);
    if (!varIt.first)
      return nullptr;
    structName = varIt.second->getType().name;
    if (structName.empty())
      return nullptr;

    // If the struct name was valid, check for an entry in the struct map.
    auto structIt = structMap.find(structName);
    if (structIt == structMap.end())
      return nullptr;
    return structIt->second.second;
  }

  /// Return the numeric member index of the given struct access expression.
  std::optional<size_t> getMemberIndex(StructAST* structAST, llvm::StringRef name) {
    auto structVars = structAST->getVariables();
    const auto *it = llvm::find_if(structVars, [&](auto &var) {
      return var->getName() == name;
    });
    if (it == structVars.end())
      return std::nullopt;
    return it - structVars.begin();
  }

  mlir::Value toRValue(mlir::Value value,
                       std::optional<mlir::Type> ltype = std::nullopt) {
    auto op = value.getDefiningOp();
    if (op == nullptr)
      return value;

    if (auto contextRefOp = dyn_cast<ContextRefOp>(op)) {
      // convert contextRef to Load
      auto resultType = ltype.value_or(contextRefOp.getType().getValueType());
      return builder.create<LoadOp>(contextRefOp->getLoc(), resultType, value);
    }

    return value;
  }

  void setConstantType(mlir::Value value, mlir::Type type) {
    if (isa<AnyType>(type))
      return;

    auto op = value.getDefiningOp();
    if (op == nullptr)
      return;

    if (auto constOp = dyn_cast<ConstantOp>(op)) {
      // assign to a constant
      if (isa<mlir::IntegerType>(constOp.getType())) {
        assert(isa<mlir::IntegerType>(type) && "Assign a integer constant to a non-integer");
        value.setType(type);
      }
    }
  }

  /// Emit a binary operation
  mlir::Value mlirGen(BinaryExprAST &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //

    // Otherwise, this is a normal binary op.
    mlir::Value lhs = mlirGen(*binop.getLHS());
    if (!lhs)
      return nullptr;
    if (binop.getOp() != '=')
      lhs = toRValue(lhs);

    mlir::Value rhs = mlirGen(*binop.getRHS());
    if (!rhs)
      return nullptr;
    // For assignment, change this to RValue (with reference to lvalue type)
    rhs = toRValue(rhs, lhs.getType());

    auto location = loc(binop.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+', '-' and '*'.
    switch (binop.getOp()) {
    case '+':
      return builder.create<AddOp>(location, lhs.getType(), lhs, rhs);
    case '-':
      return builder.create<SubOp>(location, lhs.getType(), lhs, rhs);
    case '*':
      return builder.create<MulOp>(location, lhs, rhs);
    case '=':
      // TODO(zhiyuang): Here we do not bring type info into IR, need to do type
      // check here

      if (auto refOp = dyn_cast<mlir::ep2::ContextRefOp>(lhs.getDefiningOp())) {
        builder.create<StoreOp>(location, lhs, rhs);

        setConstantType(rhs, refOp.getType().getValueType());
        return builder.create<NopOp>(location, builder.getNoneType());
      } else if (auto accessOp = dyn_cast<StructAccessOp>(lhs.getDefiningOp())) {
        auto structType = accessOp.getInput().getType();
        auto value = builder.create<StructUpdateOp>(location,
          structType, accessOp.getInput(), accessOp.getIndexAttr(), rhs);

        auto leftType = structType.getElementTypes()[accessOp.getIndex()];
        setConstantType(rhs, leftType);
        update(*binop.getLHS(), value);
        return value;
      } else {
        // assign to a variable, update ssa value
        setConstantType(rhs, lhs.getType());
        update(*binop.getLHS(), rhs);
        return rhs;
      }

      // TODO(zhiyuang): check this. Is there any other lvalue type?
      emitError(location, "Assignment: unknown lvalue type");
      lhs.getDefiningOp()->dump();
      return nullptr;
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(VariableExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()).first)
      return variable;

      emitError(loc(expr.loc()), "error: unknown variable '")
          << expr.getName() << "'";
      return nullptr;
  }

  /// Emit a return operation. This will return failure if any generation fails.
  mlir::LogicalResult mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (ret.getExpr().has_value()) {
      if (!(expr = mlirGen(**ret.getExpr())))
        return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    builder.create<ReturnOp>(location,
                             expr ? ArrayRef(expr) : ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// Emit a constant for a literal/constant array. It will be emitted as a
  /// flattened array of data in an Attribute attached to a `ep2.constant`
  /// operation. See documentation on [Attributes](LangRef.md#attributes) for
  /// more details. Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in MLIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "ep2.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::IntegerAttr getConstantAttr(NumberExprAST &lit) {
    // The type of this attribute is tensor of 64-bit floating-point with no
    // shape.
    return builder.getI64IntegerAttr(lit.getValue());
  }
  /// Emit a constant for a struct literal. It will be emitted as an array of
  /// other literals in an Attribute attached to a `ep2.struct_constant`
  /// operation. This function returns the generated constant, along with the
  /// corresponding struct type.
  std::pair<mlir::ArrayAttr, mlir::Type>
  getConstantAttr(StructLiteralExprAST &lit) {
    std::vector<mlir::Attribute> attrElements;
    std::vector<mlir::Type> typeElements;

    for (auto &var : lit.getValues()) {
      if (auto *number = llvm::dyn_cast<NumberExprAST>(var.get())) {
        attrElements.push_back(getConstantAttr(*number));
        typeElements.push_back(builder.getType<AnyType>());
      } else {
        auto *structLit = llvm::cast<StructLiteralExprAST>(var.get());
        auto attrTypePair = getConstantAttr(*structLit);
        attrElements.push_back(attrTypePair.first);
        typeElements.push_back(attrTypePair.second);
      }
    }
    mlir::ArrayAttr dataAttr = builder.getArrayAttr(attrElements);
    mlir::Type dataType = StructType::get(builder.getContext(), false, typeElements, "");
    return std::make_pair(dataAttr, dataType);
  }

  /// Emit a struct literal. It will be emitted as an array of
  /// other literals in an Attribute attached to a `ep2.struct_constant`
  /// operation.
  mlir::Value mlirGen(StructLiteralExprAST &lit) {
    mlir::ArrayAttr dataAttr;
    mlir::Type dataType;
    std::tie(dataAttr, dataType) = getConstantAttr(lit);

    // Build the MLIR op `ep2.struct_constant`. This invokes the
    // `StructConstantOp::build` method.
    return builder.create<StructConstantOp>(loc(lit.loc()), dataType, dataAttr);
  }

  /// Emit a call expression. It emits specific operations for the `transpose`
  /// builtin. Other identifiers are assumed to be user-defined functions.
  mlir::Value mlirGen(CallExprAST &call) {
    auto location = loc(call.loc());
    auto caller = getPath(*call.getCallee(), true);

    auto &path = call.getCallee()->getPath();
    std::string callee = std::string(path.back()->getName());

    // Codegen the operands first.
    SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }

    // generate struct access
    if (callee == "extract") {
      if (!caller || operands.size() != 1) {
        emitError(location) << "callop: invalid extract";
        return nullptr;
      }
      auto &target = operands[0];
      auto extractOp = builder.create<ExtractOp>(location, target.getType(), caller);
      // if its a value, update the SSA value
      auto &targetAst = *call.getArgs()[0];
      // single layer 
      update(targetAst, extractOp);
      // TODO(zhiyuang): else add access
      return extractOp;
    } else if (callee == "emit") { // generate struct write
      if (!caller || operands.size() != 1) {
        emitError(location) << "callop: invalid emit";
        return nullptr;
      }
      // TODO: update variable or add assignment
      auto &target = operands[0];
      builder.create<EmitOp>(location, caller, target);
      return builder.create<NopOp>(location);
    } else if(callee == "lookup"){
      if (!caller || !isa<TableType>(caller.getType()) || operands.size() != 1) {
        emitError(location) << "callop: invalid lookup";
        return nullptr;
      }
      auto table = dyn_cast<TableType>(caller.getType());
      return builder.create<LookupOp>(location, table.getValueType(), caller, operands[0]);
    } else if(callee == "update"){
      if (!caller || operands.size() != 2) {
        emitError(location) << "callop: invalid update";
        return nullptr;
      }
      // TODO: update variable or add assignment
      auto &key = operands[0];
      auto &value = operands[1];
      builder.create<UpdateOp>(location, caller, key, value);
      return builder.create<NopOp>(location);
    }
    // Otherwise this is a call to a user-defined function. Calls to
    // user-defined functions are mapped to a custom call that takes the callee
    // name as an attribute.
    auto calledFuncIt = functionMap.find(callee);
    if (calledFuncIt == functionMap.end()) {
      emitError(location) << "no defined function found for '" << callee << "'";
      return nullptr;
    }
    mlir::ep2::FuncOp calledFunc = calledFuncIt->second;
    return builder.create<CallOp>(
        location, calledFunc.getFunctionType().getResult(0),
        mlir::SymbolRefAttr::get(builder.getContext(), callee), operands);
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value mlirGen(NumberExprAST &num) {
    auto attr = builder.getI64IntegerAttr(num.getValue());
    return builder.create<ConstantOp>(loc(num.loc()), attr);
  }

  // return either a struct access op or a variable op
  mlir::Value getPath(PathExprAST &path, bool limit = false) {
    auto location = loc(path.loc());
    if (path.getPathLength() == 0) {
      emitError(location) << "Not a function";
      return nullptr;
    }

    int end = limit ? path.getPathLength() - 1 : path.getPathLength();
    mlir::Value value = nullptr;
    for (int i = 0; i < end; i++) {
      auto &curVarExpr = path.getPath()[i];

      if (!value) { // only the first call could be value
        auto newValue = mlirGen(*curVarExpr);
        if (!newValue) return nullptr;
        value = newValue;
      } else {
        // If it is a context..
        if (isa<ContextType>(value.getType())) {
          auto internalType = builder.getType<ContextRefType>(getVarType());
          value = builder.create<ContextRefOp>(location, internalType, curVarExpr->getName(), value);
          continue;
        }
        // else, c++ dot access
        // TODO(zhiyuang): multi-level struct access
        // here we lookup only by local symbol table, need to also be on fields
        auto &varExpr = path.getPath()[i-1];
        auto structExpr = getStructForName(varExpr->getName());
        if (structExpr == nullptr) {
          emitError(location) << "dot applied to a non-struct: " << varExpr->getName();
          return nullptr;
        }
        auto index = getMemberIndex(structExpr, curVarExpr->getName());

        if (!index) {
          emitError(location) << "invalid access into struct expression";
          return nullptr;
        }

        value = builder.create<StructAccessOp>(location, value, *index);
      }
    }

    return value;
  }

  mlir::Value mlirGen(PathExprAST &path) {
    return getPath(path);
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case ep2::ExprAST::Expr_BinOp:
      return mlirGen(cast<BinaryExprAST>(expr));
    case ep2::ExprAST::Expr_Var:
      return mlirGen(cast<VariableExprAST>(expr));
    case ep2::ExprAST::Expr_AtomLiteral:
      return mlirGen(cast<AtomLiteralExprAST>(expr));
    case ep2::ExprAST::Expr_StructLiteral:
      return mlirGen(cast<StructLiteralExprAST>(expr));
    case ep2::ExprAST::Expr_Call:
      return mlirGen(cast<CallExprAST>(expr));
    case ep2::ExprAST::Expr_Num:
      return mlirGen(cast<NumberExprAST>(expr));
    case ep2::ExprAST::Expr_Path:
      return mlirGen(cast<PathExprAST>(expr));
    case ep2::ExprAST::Expr_Init:
      return mlirGen(cast<InitExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value mlirGen(VarDeclExprAST &vardecl) {
    auto *init = vardecl.getInitVal();
    auto location = loc(vardecl.loc());
    mlir::Value value;
    if (!init) {
      // build default initalization
      mlir::Type type = getVarType(vardecl.getType(), vardecl.loc());
      if (!type) {
        emitError(location)
            << "do not get a type for default initialization";
        return nullptr;
      }

      value = builder.create<InitOp>(location, type);
    } else {
      value = mlirGen(*init);
    }

    if (!value)
      return nullptr;

    // Handle the case where we are initializing a struct value.
    VarType varType = vardecl.getType();
    if (!varType.name.empty()) {
      // Check that the initializer type is the same as the variable
      // declaration.
      mlir::Type type = getVarType(varType, vardecl.loc());
      if (!type)
        return nullptr;
      if (type != value.getType()) {
        emitError(loc(vardecl.loc()))
            << "struct type of initializer is different than the variable "
               "declaration. Got "
            << value.getType() << ", but expected " << type;
        return nullptr;
      }
    }

    // Register the value in the symbol table.
    if (failed(declare(vardecl, value)))
      return nullptr;
    return value;
  }

  mlir::Value mlirGen(InitExprAST &init) {
    auto location = loc(init.loc());
    VarType varType = init.getType();
    mlir::Type type = getVarType(varType, init.loc());
    if (!type) {
      emitError(location)
          << "do not get a type for default initialization";
      return nullptr;
    }

    auto values = llvm::map_to_vector(
        init.getInitVals(), [&](auto &expr) { return mlirGen(*expr); });
    return builder.create<InitOp>(location, type, values);
  }

  mlir::Value mlirGen(AtomLiteralExprAST &atom) {
    return builder.create<ConstantOp>(loc(atom.loc()), atom.getAtom());
  }

  mlir::LogicalResult mlirGen(IfElseExprAST &ifelse, bool rootScope) {
    auto cond = mlirGen(*ifelse.getCond());
    if (!cond)
      return mlir::failure();

    mlir::Region thenRegion, elseRegion;
    auto ip = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&thenRegion.emplaceBlock());
    // build if block
    updateTable = std::make_unique<UpdateTableT>();
    if (mlirGen(*ifelse.getThenExprs(), /* newScope= */ false).failed())
      return mlir::failure();
    auto thenTable = std::move(updateTable);

    // build else block
    builder.setInsertionPointToStart(&elseRegion.emplaceBlock());
    updateTable = std::make_unique<UpdateTableT>();
    if (ifelse.hasElse()) {
      if (mlirGen(*ifelse.getElseExprs(), /* newScope= */ false).failed())
        return mlir::failure();
    }
    auto elseTable = std::move(updateTable);

    // both table exists here
    auto merge = [&, this](UpdateTableT *from, UpdateTableT *to) {
      for (auto &[k, v] : *from) {
        if (!to->count(k)) {
          auto value = symbolTable.lookup(k).first;
          to->insert({k, value});
        }
      }
    };
    merge(thenTable.get(), elseTable.get());
    merge(elseTable.get(), thenTable.get());

    auto keys = llvm::map_to_vector(*thenTable, [](auto &pair) { return pair.first; });

    // create a new if op
    bool hasElse = ifelse.hasElse() || keys.size() != 0;
    auto types = llvm::map_to_vector(
        keys, [&](auto &key) { return symbolTable.lookup(key).first.getType(); });

    builder.restoreInsertionPoint(ip);
    auto newIfOp = builder.create<mlir::scf::IfOp>(loc(ifelse.loc()), types, cond, hasElse);
    newIfOp.getThenRegion().takeBody(thenRegion);
    if (ifelse.hasElse())
      newIfOp.getElseRegion().takeBody(elseRegion);

    // TODO(zhiyuang): empty else block?
    builder.setInsertionPointToEnd(newIfOp.thenBlock());
    auto thenYields = llvm::map_to_vector(
        keys, [&](auto &key) { return thenTable->at(key); });
    builder.create<mlir::scf::YieldOp>(loc(ifelse.loc()), thenYields);
    if (hasElse) {
      builder.setInsertionPointToEnd(newIfOp.elseBlock());
      auto elseYields = llvm::map_to_vector(
          keys, [&](auto &key) { return elseTable->at(key); });
      builder.create<mlir::scf::YieldOp>(loc(ifelse.loc()), elseYields);
    }

    // update the values
    // TODO(zhiyuang): check if its at top level
    for (size_t i = 0; i < keys.size(); i++) {
      auto [_, decl] = symbolTable.lookup(keys[i]);
      symbolTable.insert(keys[i], {newIfOp.getResult(i), decl});
    }
    builder.setInsertionPointAfter(newIfOp);

    return mlir::success();
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  mlir::LogicalResult mlirGen(ExprASTList &blockAST, bool newScope = true) {
    // construct this any way; new scope just prevent variable declaration
    SymbolTableScopeT varScope(symbolTable);

    for (auto &expr : blockAST) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!newScope) {
          emitError(loc(vardecl->loc()))
              << "error: variable declaration not allowed in this context";
          return mlir::failure();
        }
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      } else if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
        return mlirGen(*ret);
      else if (auto *ifelse = dyn_cast<IfElseExprAST>(expr.get())) {
        if (mlirGen(*ifelse, newScope).failed())
          return mlir::failure();
        continue;
      }

      // Generic expression dispatch codegen.
      if (!mlirGen(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getVarType() {
    return builder.getType<AnyType>();
  }

  /// Build an MLIR type from a ep2 AST variable type (forward to the generic
  /// getType above for non-struct types).
  mlir::Type getVarType(const VarType &type, const Location &location) {
    if (!type.name.empty()) {
      // Handle builtin types
      if (type.name == "long")
        return builder.getI64Type();
      if (type.name == "int")
        return builder.getI32Type();
      else if (type.name == "atom")
        return builder.getType<AtomType>();
      else if (type.name == "context")
        return builder.getType<ContextType>();
      else if (type.name == "buf")
        return builder.getType<BufferType>();
      else if (type.name == "bits") {
        if (!type.checkParam({VarTemplateParam::VarTemplateParam_Integer})) {
          emitError(loc(location)) << "error: not valid parameter for bits'"
                                   << type.params.size() << "'";
          return nullptr;
        }
        return builder.getIntegerType(type.params[0].value);
      } else if (type.name == "table") {
        if (!type.checkParam({VarTemplateParam::VarTemplateParam_Type,
                              VarTemplateParam::VarTemplateParam_Type,
                              VarTemplateParam::VarTemplateParam_Integer})) {
          emitError(loc(location))
              << "error: Wrong Type parameter for table'" << type.params.size() << "'";
          return nullptr;
        }
        // // TODO(zhiyuang): correct location
        auto kt = getVarType(*type.params[0].type, location);
        auto vt = getVarType(*type.params[1].type, location);
        return builder.getType<TableType>(kt, vt, type.params[2].value);
      }

      auto it = structMap.find(type.name);
      if (it == structMap.end()) {
        emitError(loc(location))
            << "error: unknown struct type '" << type.name << "'";
        return nullptr;
      }
      return it->second.first;
    }

    return builder.getType<AnyType>();
  }
};

} // namespace

namespace ep2 {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace ep2
