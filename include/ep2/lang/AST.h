//===- AST.h - Node definition for the Toy AST ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST for the Toy language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure where each node
// references its children using std::unique_ptr<>.
//
//===----------------------------------------------------------------------===//

#ifndef EP2_AST_H
#define EP2_AST_H

#include "ep2/lang/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include <utility>
#include <vector>
#include <optional>
#include <string>

namespace ep2 {

/// A variable type with either name or shape information.
struct VarType {
  std::string name;
  std::vector<int64_t> shape;
};

/// Base class for all expression nodes.
class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_StructLiteral,
    Expr_AtomLiteral,
    Expr_Var,
    Expr_Path, // Var connected by dots
    Expr_BinOp,
    Expr_Call,
    Expr_Init
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(std::move(location)) {}
  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const ExprASTKind kind;
  Location location;
};

/// A block-list of expressions.
using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  unsigned long val;

public:
  NumberExprAST(Location loc, unsigned long val)
      : ExprAST(Expr_Num, std::move(loc)), val(val) {}

  unsigned long getValue() { return val; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
};

/// Expression class for a literal value.
class LiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;
  std::vector<int64_t> dims;

public:
  LiteralExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> values,
                 std::vector<int64_t> dims)
      : ExprAST(Expr_Literal, std::move(loc)), values(std::move(values)),
        dims(std::move(dims)) {}

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }
  llvm::ArrayRef<int64_t> getDims() { return dims; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }
};

/// @brief Atom Literal
class AtomLiteralExprAST : public ExprAST {
  std::string atom;

public:
  AtomLiteralExprAST(Location loc, llvm::StringRef atom)
      : ExprAST(Expr_AtomLiteral, std::move(loc)), atom(atom) {}

  llvm::StringRef getAtom() { return atom; }
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_AtomLiteral; }
};

/// Expression class for a literal struct value.
class StructLiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;

public:
  StructLiteralExprAST(Location loc,
                       std::vector<std::unique_ptr<ExprAST>> values)
      : ExprAST(Expr_StructLiteral, std::move(loc)), values(std::move(values)) {
  }

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) {
    return c->getKind() == Expr_StructLiteral;
  }
};

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string name;

public:
  VariableExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Var, std::move(loc)), name(name) {}

  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Var; }
};

class PathExprAST : public ExprAST {
  std::vector<std::unique_ptr<VariableExprAST>> path;

public:
  PathExprAST(Location loc, std::unique_ptr<VariableExprAST> name)
      : ExprAST(Expr_Path, std::move(loc)), path{} {
    path.push_back(std::move(name));
  }

  std::vector<std::unique_ptr<VariableExprAST>> &getPath() { return path; }
  size_t getPathLength() { return path.size(); }
  void append(std::unique_ptr<VariableExprAST> next) {
    path.push_back(std::move(next));
  }
  std::string print() {
    std::string pathStr = "Path: ";
    for (size_t i = 0 ; i < path.size(); i++) {
      pathStr += path[i]->getName();
      if (i != path.size() - 1) {
        pathStr += ".";
      }
    }
    return pathStr;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Path; }
};

/// Expression class for defining a variable.
class VarDeclExprAST : public ExprAST {
  std::string name;
  VarType type;
  std::unique_ptr<ExprAST> initVal;

public:
  VarDeclExprAST(Location loc, llvm::StringRef name, VarType type,
                 std::unique_ptr<ExprAST> initVal = nullptr)
      : ExprAST(Expr_VarDecl, std::move(loc)), name(name),
        type(std::move(type)), initVal(std::move(initVal)) {}

  llvm::StringRef getName() { return name; }
  ExprAST *getInitVal() { return initVal.get(); }
  const VarType &getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
};

class InitExprAST : public ExprAST {
  VarType type;
  std::vector<std::unique_ptr<ExprAST>> initVals;

public:
  InitExprAST(Location loc, VarType type,
                 std::vector<std::unique_ptr<ExprAST>> initVals)
      : ExprAST(Expr_Init, std::move(loc)),
        type(std::move(type)), initVals(std::move(initVals)) {}

  std::vector<std::unique_ptr<ExprAST>> &getInitVals() { return initVals; }
  const VarType &getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Init; }
};

/// Expression class for a return operator.
class ReturnExprAST : public ExprAST {
  std::optional<std::unique_ptr<ExprAST>> expr;

public:
  ReturnExprAST(Location loc, std::optional<std::unique_ptr<ExprAST>> expr)
      : ExprAST(Expr_Return, std::move(loc)), expr(std::move(expr)) {}

  std::optional<ExprAST *> getExpr() {
    if (expr.has_value())
      return expr->get();
    return std::nullopt;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Return; }
};

/// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;

public:
  char getOp() { return op; }
  ExprAST *getLHS() { return lhs.get(); }
  ExprAST *getRHS() { return rhs.get(); }

  BinaryExprAST(Location loc, char op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
      : ExprAST(Expr_BinOp, std::move(loc)), op(op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_BinOp; }
};

/// Expression class for function calls.
class CallExprAST : public ExprAST {
  std::unique_ptr<PathExprAST> callee;
  std::vector<std::unique_ptr<ExprAST>> args;

public:
  CallExprAST(Location loc, std::unique_ptr<PathExprAST> callee,
              std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(Expr_Call, std::move(loc)), callee(std::move(callee)),
        args(std::move(args)) {}

  std::unique_ptr<PathExprAST> &getCallee() { return callee; }
  std::string printCallee() { return callee->print(); }
  llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return args; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Call; }
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
class PrototypeAST {
 public:
  enum FunctionType {
    Function_Controller,
    Function_Handler
  };

 private:
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VarDeclExprAST>> args;
  std::optional<std::string> atom;
  FunctionType functionType;

  std::string mangledName{};

public:
  PrototypeAST(Location location, const std::string &name,
               std::vector<std::unique_ptr<VarDeclExprAST>> args, std::optional<std::string> atom, FunctionType type) 
      : location(std::move(location)), name(name), args(std::move(args)), atom(atom), functionType(type) {}

  const Location &loc() { return location; }
  std::string getFunctionTypeName() const {
    if (functionType == Function_Controller)
      return "controller";
    else
      return "handler";
  }
  std::optional<std::string> getAtom() const { return atom; }
  llvm::StringRef getName() const { return name; }
  llvm::StringRef getMangledName() {
    mangledName = "__" + getFunctionTypeName() + "_" + name;
    if (atom)
      mangledName = mangledName + "_" + atom.value();
    return mangledName;
  }
  llvm::ArrayRef<std::unique_ptr<VarDeclExprAST>> getArgs() { return args; }
};

/// This class represents a top level record in a module.
class RecordAST {
public:
  enum RecordASTKind {
    Record_Function,
    Record_Struct,
    Record_Event,
    Record_Handler,
    Record_Controller
  };

  RecordAST(RecordASTKind kind) : kind(kind) {}
  virtual ~RecordAST() = default;

  RecordASTKind getKind() const { return kind; }

private:
  const RecordASTKind kind;
};

/// This class represents a function definition itself.
class FunctionAST : public RecordAST {
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<ExprASTList> body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> proto,
              std::unique_ptr<ExprASTList> body)
      : RecordAST(Record_Function), proto(std::move(proto)),
        body(std::move(body)) {}
  PrototypeAST *getProto() { return proto.get(); }
  ExprASTList *getBody() { return body.get(); }

  /// LLVM style RTTI
  static bool classof(const RecordAST *r) {
    return r->getKind() == Record_Function;
  }
};

/// This class represents a struct definition.
class StructAST : public RecordAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VarDeclExprAST>> variables;

public:
  StructAST(Location location, const std::string &name,
            std::vector<std::unique_ptr<VarDeclExprAST>> variables)
      : RecordAST(Record_Struct), location(std::move(location)), name(name),
        variables(std::move(variables)) {}

  const Location &loc() { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::unique_ptr<VarDeclExprAST>> getVariables() {
    return variables;
  }

  /// LLVM style RTTI
  static bool classof(const RecordAST *r) {
    return r->getKind() == Record_Struct;
  }
};

/// @brief This class represents a event definition
class EventAST : public RecordAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VarDeclExprAST>> variables;

public:
  EventAST(Location location, const std::string &name,
            std::vector<std::unique_ptr<VarDeclExprAST>> variables)
      : RecordAST(Record_Struct), location(std::move(location)), name(name),
        variables(std::move(variables)) {}

  const Location &loc() { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::unique_ptr<VarDeclExprAST>> getVariables() {
    return variables;
  }

  /// LLVM style RTTI
  static bool classof(const RecordAST *r) {
    return r->getKind() == Record_Event;
  }
};

/// This class represents a list of functions to be processed together
class ModuleAST {
  std::vector<std::unique_ptr<RecordAST>> records;

public:
  ModuleAST(std::vector<std::unique_ptr<RecordAST>> records)
      : records(std::move(records)) {}

  auto begin() { return records.begin(); }
  auto end() { return records.end(); }
};

void dump(ModuleAST &);

} // namespace toy

#endif // EP2_AST_H
