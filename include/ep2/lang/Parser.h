//===- Parser.h - Toy Language Parser -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parser for the Toy language. It processes the Token
// provided by the Lexer and returns an AST.
//
//===----------------------------------------------------------------------===//

#ifndef EP2_PARSER_H
#define EP2_PARSER_H

#include "ep2/lang/AST.h"
#include "ep2/lang/Lexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <utility>
#include <vector>
#include <optional>

namespace ep2 {

/// This is a simple recursive parser for the Toy language. It produces a well
/// formed AST from a stream of Token supplied by the Lexer. No semantic checks
/// or symbol resolution is performed. For example, variables are referenced by
/// string and the code could reference an undeclared variable and the parsing
/// succeeds.
class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &lexer) : lexer(lexer) {}

  /// Parse a full Module. A module is a list of function definitions.
  std::unique_ptr<ModuleAST> parseModule() {
    lexer.getNextToken(); // prime the lexer

    // Parse functions and structs one at a time and accumulate in this vector.
    // module ::= funcion | struct | event | handler | controller
    std::vector<std::unique_ptr<RecordAST>> records;
    std::unordered_map<std::string, VarType> contextFields;

    while (true) {
      std::unique_ptr<RecordAST> record;
      auto tok = lexer.getCurToken();
      switch (tok) {
      case tok_eof:
        break;
      // function types
      case tok_def:        // FALL THROUGH
      case tok_handler:    // FALL THROUGH
      case tok_controller:
        record = parseDefinition(tok);
        break;
      // Struct types
      case tok_struct: {
        record = parseStruct(tok);
        StructAST* str = static_cast<StructAST*>(record.get());
        if (str->getName() == "context") {
          for (const auto& field : str->getVariables()) {
            contextFields.emplace(field->getName().str(), field->getType());
          }
          continue;
        }
        break;
      }
      case tok_event:
        record = parseStruct(tok);
        break;
      // modifiers 
      case tok_extern:
        // TODO: add modifier capture
        lexer.consume(tok_extern);
        continue;
      default:
        return parseError<ModuleAST>("'def' or 'struct'",
                                     "when parsing top level module records");
      }
      if (!record)
        break;
      records.push_back(std::move(record));
    }

    // If we didn't reach EOF, there was an error during parsing
    if (lexer.getCurToken() != tok_eof)
      return parseError<ModuleAST>("nothing", "at end of module");

    return std::make_unique<ModuleAST>(std::move(records), std::move(contextFields));
  }

private:
  Lexer &lexer;

  /// Parse a return statement.
  /// return :== return ; | return expr ;
  std::unique_ptr<ReturnExprAST> parseReturn(Token token) {
    auto loc = lexer.getLastLocation();
    lexer.consume(token);

    // return takes an optional argument
    std::optional<std::unique_ptr<ExprAST>> expr;

    if (lexer.getCurToken() != ';') {
      expr = parseExpression();
      if (!expr)
        return nullptr;
    }
    return std::make_unique<ReturnExprAST>(std::move(loc), std::move(expr));
  }

  /// Parse a literal number.
  /// numberexpr ::= number
  std::unique_ptr<ExprAST> parseNumberExpr() {
    auto loc = lexer.getLastLocation();
    auto result =
        std::make_unique<NumberExprAST>(std::move(loc), lexer.getValue());
    lexer.consume(tok_number);
    return std::move(result);
  }

  /// Parse an atom expression. Atom is a unique identifier
  /// atom ::= ':' id
  std::unique_ptr<AtomLiteralExprAST> parseAtom() {
    auto loc = lexer.getLastLocation();
    if (lexer.getCurToken() != tok_colon)
      return parseError<AtomLiteralExprAST>(":", "in atom literal");

    lexer.consume(tok_colon);
    if (lexer.getCurToken() != tok_identifier)
      return parseError<AtomLiteralExprAST>("identifier", "in atom literal");

    auto result =
        std::make_unique<AtomLiteralExprAST>(std::move(loc), lexer.getId());
    lexer.consume(tok_identifier);
    return std::move(result);
  }

  /// Parse a literal array expression.
  /// tensorLiteral ::= [ literalList ] | number
  /// literalList ::= tensorLiteral | tensorLiteral, literalList
  std::unique_ptr<ExprAST> parseTensorLiteralExpr() {
    auto loc = lexer.getLastLocation();
    lexer.consume(Token('['));

    // Hold the list of values at this nesting level.
    std::vector<std::unique_ptr<ExprAST>> values;
    // Hold the dimensions for all the nesting inside this level.
    std::vector<int64_t> dims;
    do {
      // We can have either another nested array or a number literal.
      if (lexer.getCurToken() == '[') {
        values.push_back(parseTensorLiteralExpr());
        if (!values.back())
          return nullptr; // parse error in the nested array.
      } else {
        if (lexer.getCurToken() != tok_number)
          return parseError<ExprAST>("<num> or [", "in literal expression");
        values.push_back(parseNumberExpr());
      }

      // End of this list on ']'
      if (lexer.getCurToken() == ']')
        break;

      // Elements are separated by a comma.
      if (lexer.getCurToken() != ',')
        return parseError<ExprAST>("] or ,", "in literal expression");

      lexer.getNextToken(); // eat ,
    } while (true);
    if (values.empty())
      return parseError<ExprAST>("<something>", "to fill literal expression");
    lexer.getNextToken(); // eat ]

    /// Fill in the dimensions now. First the current nesting level:
    dims.push_back(values.size());

    /// If there is any nested array, process all of them and ensure that
    /// dimensions are uniform.
    if (llvm::any_of(values, [](std::unique_ptr<ExprAST> &expr) {
          return llvm::isa<LiteralExprAST>(expr.get());
        })) {
      auto *firstLiteral = llvm::dyn_cast<LiteralExprAST>(values.front().get());
      if (!firstLiteral)
        return parseError<ExprAST>("uniform well-nested dimensions",
                                   "inside literal expression");

      // Append the nested dimensions to the current level
      auto firstDims = firstLiteral->getDims();
      dims.insert(dims.end(), firstDims.begin(), firstDims.end());

      // Sanity check that shape is uniform across all elements of the list.
      for (auto &expr : values) {
        auto *exprLiteral = llvm::cast<LiteralExprAST>(expr.get());
        if (!exprLiteral)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
        if (exprLiteral->getDims() != firstDims)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
      }
    }
    return std::make_unique<LiteralExprAST>(std::move(loc), std::move(values),
                                            std::move(dims));
  }

  /// Parse a literal struct expression.
  /// structLiteral ::= { (structLiteral | tensorLiteral)+ }
  std::unique_ptr<ExprAST> parseStructLiteralExpr() {
    auto loc = lexer.getLastLocation();
    lexer.consume(Token('{'));

    // Hold the list of values.
    std::vector<std::unique_ptr<ExprAST>> values;
    do {
      // We can have either another nested array or a number literal.
      if (lexer.getCurToken() == '[') {
        values.push_back(parseTensorLiteralExpr());
        if (!values.back())
          return nullptr;
      } else if (lexer.getCurToken() == tok_number) {
        values.push_back(parseNumberExpr());
        if (!values.back())
          return nullptr;
      } else {
        if (lexer.getCurToken() != '{')
          return parseError<ExprAST>("{, [, or number",
                                     "in struct literal expression");
        values.push_back(parseStructLiteralExpr());
      }

      // End of this list on '}'
      if (lexer.getCurToken() == '}')
        break;

      // Elements are separated by a comma.
      if (lexer.getCurToken() != ',')
        return parseError<ExprAST>("} or ,", "in struct literal expression");

      lexer.getNextToken(); // eat ,
    } while (true);
    if (values.empty())
      return parseError<ExprAST>("<something>",
                                 "to fill struct literal expression");
    lexer.getNextToken(); // eat }

    return std::make_unique<StructLiteralExprAST>(std::move(loc),
                                                  std::move(values));
  }

  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<ExprAST> parseParenExpr() {
    lexer.getNextToken(); // eat (.
    auto v = parseExpression();
    if (!v)
      return nullptr;

    if (lexer.getCurToken() != ')')
      return parseError<ExprAST>(")", "to close expression with parentheses");
    lexer.consume(Token(')'));
    return v;
  }

  /// Parse a call expression.
  std::unique_ptr<ExprAST> parseCallExpr(std::unique_ptr<PathExprAST> path,
                                         const Location &loc) {
    lexer.consume(Token('('));
    std::vector<std::unique_ptr<ExprAST>> args;
    if (lexer.getCurToken() != ')') {
      while (true) {
        if (auto arg = parseExpression())
          args.push_back(std::move(arg));
        else
          return nullptr;

        if (lexer.getCurToken() == ')')
          break;

        if (lexer.getCurToken() != ',')
          return parseError<ExprAST>(", or )", "in argument list");
        lexer.getNextToken();
      }
    }
    lexer.consume(Token(')'));

    // It can be a builtin call to print
    // TODO: extract here?

    // Call to a user-defined function
    return std::make_unique<CallExprAST>(loc, std::move(path),
                                         std::move(args));
  }

  /// identifierexpr
  ///   ::= identifier
  ///   ::= identifierexpr '.'identifier 
  ///   ::= identifier '(' expression ')'
  std::unique_ptr<ExprAST> parseIdentifierExpr() {
    std::string name = std::string(lexer.getId());
    auto variable = std::make_unique<VariableExprAST>(
      lexer.getLastLocation(), name);

    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat identifier.

    auto path = std::make_unique<PathExprAST>(
      lexer.getLastLocation(), std::move(variable));

    // Parse dot expressions
    while (lexer.getCurToken() == '.') {
      lexer.consume(tok_dot);
      if (lexer.getCurToken() != tok_identifier)
        return parseError<ExprAST>("identifier", "parse struct access");
      path->append(std::make_unique<VariableExprAST>(
              lexer.getLastLocation(), lexer.getId()));
      lexer.consume(tok_identifier);
    }

    std::unique_ptr<AtomLiteralExprAST> atom;
    if (lexer.getCurToken() == ':')
      atom = parseAtom();

    // This is a function call.
    // TODO: fix name to path
    if (lexer.getCurToken() == '(')
      return parseCallExpr(std::move(path), lexer.getLastLocation());

    // Parse construction
    if (lexer.getCurToken() == '{') {
      lexer.consume(tok_bracket_open);

      if (path->getPath().size() != 1)
        return parseError<ExprAST>("name error", "parse init");

      std::vector<std::unique_ptr<ExprAST>> values;
      if (atom)
        values.push_back(std::move(atom));

      do {
        auto expr = parseExpression();
        if (expr == nullptr)
          return parseError<ExprAST>("expr", "Parsing Initiliazaiton");
        values.push_back(std::move(expr));
        
        if (lexer.getCurToken() == tok_bracket_close) {
          lexer.consume(tok_bracket_close);
          break;
        }

        if (lexer.getCurToken() != ',')
          return parseError<ExprAST>("comma", "parsing init list");
        lexer.consume(Token(','));
      } while (true);

      // complex expression for name
      return std::make_unique<InitExprAST>(
        lexer.getLastLocation(), VarType{name}, std::move(values));
    }

    return path;
  }

  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= tensorliteral
  std::unique_ptr<ExprAST> parsePrimary() {
    switch (lexer.getCurToken()) {
    default:
      llvm::errs() << "unknown token '" << lexer.getCurToken()
                   << "' when expecting an expression\n";
      return nullptr;
    case tok_identifier:
      return parseIdentifierExpr();
    case tok_colon:
      return parseAtom();
    case tok_number:
      return parseNumberExpr();
    case '(':
      return parseParenExpr();
    case '{':
      return parseStructLiteralExpr();
    case ';':
      return nullptr;
    case '}':
      return nullptr;
    }
  }

  /// Recursively parse the right hand side of a binary expression, the ExprPrec
  /// argument indicates the precedence of the current binary operator.
  ///
  /// binoprhs ::= ('+' primary)*
  std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                         std::unique_ptr<ExprAST> lhs) {
    // If this is a binop, find its precedence.
    while (true) {
      int tokPrec = getTokPrecedence();

      // If this is a binop that binds at least as tightly as the current binop,
      // consume it, otherwise we are done.
      if (tokPrec < exprPrec)
        return lhs;

      // Okay, we know this is a binop.
      int binOp = lexer.getCurToken();
      lexer.consume(Token(binOp));
      auto loc = lexer.getLastLocation();

      // Parse the primary expression after the binary operator.
      auto rhs = parsePrimary();
      if (!rhs)
        return parseError<ExprAST>("expression", "to complete binary operator");

      // If BinOp binds less tightly with rhs than the operator after rhs, let
      // the pending operator take rhs as its lhs.
      int nextPrec = getTokPrecedence();
      if (tokPrec < nextPrec) {
        rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
        if (!rhs)
          return nullptr;
      }

      // Merge lhs/RHS.
      lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp,
                                            std::move(lhs), std::move(rhs));
    }
  }

  /// expression::= primary binop rhs
  std::unique_ptr<ExprAST> parseExpression() {
    auto lhs = parsePrimary();
    if (!lhs)
      return nullptr;

    return parseBinOpRHS(0, std::move(lhs));
  }

  /// type ::= < shape_list >
  /// shape_list ::= num | num , shape_list
  std::unique_ptr<VarType> parseType() {
    if (lexer.getCurToken() != '<')
      return parseError<VarType>("<", "to begin type");
    lexer.getNextToken(); // eat <

    auto type = std::make_unique<VarType>();

    while (lexer.getCurToken() == tok_number) {
      type->shape.push_back(lexer.getValue());
      lexer.getNextToken();
      if (lexer.getCurToken() == ',')
        lexer.getNextToken();
    }

    if (lexer.getCurToken() != '>')
      return parseError<VarType>(">", "to end type");
    lexer.getNextToken(); // eat >
    return type;
  }

  /// Parse a typed variable declaration.
  std::unique_ptr<VarDeclExprAST>
  parseTypedDeclaration(llvm::StringRef typeName, bool requiresInitializer,
                        const Location &loc) {
    // Parse the variable name.
    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("name", "in variable declaration");
    std::string id(lexer.getId());
    lexer.getNextToken(); // eat id

    // Parse the initializer.
    std::unique_ptr<ExprAST> expr;
    if (requiresInitializer) {
      if (lexer.getCurToken() != '=')
        return parseError<VarDeclExprAST>("initializer",
                                          "in variable declaration");
      lexer.consume(Token('='));
      expr = parseExpression();
    }

    VarType type;
    type.name = std::string(typeName);
    return std::make_unique<VarDeclExprAST>(loc, std::move(id), std::move(type),
                                            std::move(expr));
  }


  /// Parse a typed variable declaration, with var length specified
  std::unique_ptr<VarDeclExprAST>
  parseTypedDeclaration(llvm::StringRef typeName, int64_t length, 
                        bool requiresInitializer, const Location &loc) {
    // Parse the variable name.
    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("name", "in variable declaration");
    std::string id(lexer.getId());
    lexer.getNextToken(); // eat id

    // Parse the initializer.
    std::unique_ptr<ExprAST> expr;
    if (requiresInitializer) {
      if (lexer.getCurToken() != '=')
        return parseError<VarDeclExprAST>("initializer",
                                          "in variable declaration");
      lexer.consume(Token('='));
      expr = parseExpression();
    }

    VarType type;
    type.name = std::string(typeName);
    type.length = length;
    return std::make_unique<VarDeclExprAST>(loc, std::move(id), std::move(type),
                                            std::move(expr));
  }

  /// Parse a variable declaration, for either a tensor value or a struct value,
  /// with an optionally required initializer.
  /// decl ::= var identifier [ type ] (= expr)? 
  /// decl ::= bits<num> identifier  (= expr)?
  /// decl ::= identifier identifier (= expr)?
  std::unique_ptr<VarDeclExprAST> parseDeclaration(bool requiresInitializer) {
    // Check to see if this is a 'var' declaration.
    if (lexer.getCurToken() == tok_var)
      return parseVarDeclaration(requiresInitializer);
    
    if(lexer.getCurToken() == tok_table)
      return parseTableDeclaration(requiresInitializer);
    
    if(lexer.getCurToken() == tok_identifier && lexer.getId() == "bits")
      return parseBitsDeclaration(requiresInitializer);

    // Parse the type name.
    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("type name", "in variable declaration");
    auto loc = lexer.getLastLocation();
    std::string typeName(lexer.getId());
    lexer.getNextToken(); // eat id

    // Parse the rest of the declaration.
    return parseTypedDeclaration(typeName, requiresInitializer, loc);
  }

  /// Parse a bits declaration, for either a tensor value or a struct value,
  /// with an optionally required initializer.
  /// decl ::= bits<num> identifier  (= expr)?
  std::unique_ptr<VarDeclExprAST>
  parseBitsDeclaration(bool requiresInitializer) {
    if (lexer.getId() != "bits")
      return parseError<VarDeclExprAST>("bits", "to begin declaration");
    
    auto loc = lexer.getLastLocation();
    std::string typeName(lexer.getId());
    lexer.getNextToken(); // eat bits

    if(lexer.getCurToken() != tok_angle_bracket_open)
      return parseError<VarDeclExprAST>("<",
                                        "after 'bits' declaration");
    loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat <       

    if (lexer.getCurToken() != tok_number)
      return parseError<VarDeclExprAST>("num", "after 'bits<' declaration"); 
    uint64_t bits_length = lexer.getValue();
    loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat num 

    if(lexer.getCurToken() != tok_angle_bracket_close)
      return parseError<VarDeclExprAST>(">",
                                        "after 'bits<num' declaration");
    loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat > 

    // Parse the rest of the declaration.
    return parseTypedDeclaration(typeName, bits_length, requiresInitializer, loc);

  }
  
  /// Parse a table declaration
  /// decl ::= table<key_type, value_type> identifier  (= expr)?
  std::unique_ptr<VarDeclExprAST>
  parseTableDeclaration(bool requiresInitializer) {
    auto loc = lexer.getLastLocation();
    std::string typeName("table");
    lexer.getNextToken(); // eat table

    if(lexer.getCurToken() != tok_angle_bracket_open)
      return parseError<VarDeclExprAST>("<",
                                        "after 'table' declaration");
    loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat <    

    // Parse: decl+
    std::vector<std::unique_ptr<VarDeclExprAST>> decls;
    while (lexer.getCurToken() != '>') {
      auto decl = parseDeclaration(/*requiresInitializer=*/false);
      if (!decl)
        return nullptr;
      decls.push_back(std::move(decl));

      if (lexer.getCurToken() != ';')
        return parseError<VarDeclExprAST>(";",
                                     "after variable in struct definition");
      lexer.consume(Token(';'));
    }

    if(lexer.getCurToken() != tok_angle_bracket_close)
      return parseError<VarDeclExprAST>(">",
                                        "after declaration");
    loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat >

    if(lexer.getCurToken() != tok_parenthese_open)
      return parseError<VarDeclExprAST>("(",
                                        "after 'table' declaration");
    loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat (    

    if (lexer.getCurToken() != tok_number)
      return parseError<VarDeclExprAST>("num", "after 'table' declaration"); 
    uint64_t table_length = lexer.getValue();
    loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat num 

    if(lexer.getCurToken() != tok_parenthese_close)
      return parseError<VarDeclExprAST>("(",
                                        "after 'table' declaration");

    loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat ) 

    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("name", "in table declaration");
    std::string id(lexer.getId());
    lexer.getNextToken(); // eat id

    assert(!requiresInitializer);
    if(decls.size()!= 2)
      return parseError<VarDeclExprAST>("more than 2 types", "in table <> declaration");

    std::unique_ptr<ExprAST> expr;
    
    VarType type;
    type.name = std::string(typeName);
    type.length = table_length; // table size

    TableVarType table_type;
    table_type.key_type = decls[0]->getType();
    table_type.value_type = decls[1]->getType();

    auto tvar = std::make_unique<VarDeclExprAST>(loc, std::move(id), std::move(type),
                                            std::move(expr));
    tvar->setTableType(table_type);
    tvar->isTable(true);
    return tvar;
  }

  /// Parse a variable declaration, it starts with a `var` keyword followed by
  /// and identifier and an optional type (shape specification) before the
  /// optionally required initializer.
  /// decl ::= var identifier [ type ] (= expr)?
  std::unique_ptr<VarDeclExprAST>
  parseVarDeclaration(bool requiresInitializer) {
    if (lexer.getCurToken() != tok_var)
      return parseError<VarDeclExprAST>("var", "to begin declaration");
    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat var

    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("identified",
                                        "after 'var' declaration");
    std::string id(lexer.getId());
    lexer.getNextToken(); // eat id

    std::unique_ptr<VarType> type; // Type is optional, it can be inferred
    if (lexer.getCurToken() == '<') {
      type = parseType();
      if (!type)
        return nullptr;
    }
    if (!type)
      type = std::make_unique<VarType>();

    std::unique_ptr<ExprAST> expr;
    if (requiresInitializer) {
      lexer.consume(Token('='));
      expr = parseExpression();
    }
    return std::make_unique<VarDeclExprAST>(std::move(loc), std::move(id),
                                            std::move(*type), std::move(expr));
  }

  /// Parse a block: a list of expression separated by semicolons and wrapped in
  /// curly braces.
  ///
  /// block ::= { expression_list }
  /// expression_list ::= block_expr ; expression_list
  /// block_expr ::= decl | "return" | expr
  std::unique_ptr<ExprASTList> parseBlock() {
    if (lexer.getCurToken() != '{')
      return parseError<ExprASTList>("{", "to begin block");
    lexer.consume(Token('{'));

    auto exprList = std::make_unique<ExprASTList>();

    // Ignore empty expressions: swallow sequences of semicolons.
    while (lexer.getCurToken() == ';')
      lexer.consume(Token(';'));

    while (lexer.getCurToken() != '}' && lexer.getCurToken() != tok_eof) {
      if (lexer.getCurToken() == tok_var) {
        // Variable declaration
        auto varDecl = parseDeclaration(/*requiresInitializer=*/true);
        if (!varDecl)
          return nullptr;
        exprList->push_back(std::move(varDecl));
      } else if (lexer.getCurToken() == tok_return || lexer.getCurToken() == tok_generate) {
        // Return statement
        auto ret = parseReturn(lexer.getCurToken());
        if (!ret)
          return nullptr;
        exprList->push_back(std::move(ret));
      } else if (lexer.getCurToken() == tok_table){
        // table<> declaration
        auto tableDecl = parseDeclaration(/*requiresInitializer=*/false);
        if (!tableDecl)
          return nullptr;
        exprList->push_back(std::move(tableDecl));
      }
      else if (lexer.getCurToken() == tok_identifier && lexer.getId() == "bits" && lexer.peekNextToken() == tok_angle_bracket_open) {
        // bits<> declaration
        auto bitsDecl = parseDeclaration(/*requiresInitializer=*/false);
        if (!bitsDecl)
          return nullptr;
        exprList->push_back(std::move(bitsDecl));
      }
      else if (lexer.getCurToken() == tok_identifier && lexer.peekNextToken() == tok_identifier) {
        auto loc = lexer.getLastLocation();
        auto type = lexer.getId();
        lexer.consume(tok_identifier);
        auto decl = parseTypedDeclaration(type, false, loc);
        if (!decl)
          return nullptr;
        exprList->push_back(std::move(decl));
      } else {
        // General expression
        auto expr = parseExpression();
        if (!expr)
          return nullptr;
        exprList->push_back(std::move(expr));
      }
      // Ensure that elements are separated by a semicolon.
      if (lexer.getCurToken() != ';')
        return parseError<ExprASTList>(";", "after expression");

      // Ignore empty expressions: swallow sequences of semicolons.
      while (lexer.getCurToken() == ';')
        lexer.consume(Token(';'));
    }

    if (lexer.getCurToken() != '}')
      return parseError<ExprASTList>("}", "to close block");

    lexer.consume(Token('}'));
    return exprList;
  }

  /// prototype ::= def id '(' decl_list ')'
  /// decl_list ::= identifier | identifier, decl_list
  std::unique_ptr<PrototypeAST> parsePrototype(Token token) {
    auto loc = lexer.getLastLocation();

    if (lexer.getCurToken() != token)
      return parseError<PrototypeAST>("def|event|controller", "in prototype");

    PrototypeAST::FunctionType type;
    if (token == tok_controller)
      type = PrototypeAST::Function_Controller;
    else 
      type = PrototypeAST::Function_Handler;

    lexer.consume(token);

    if (lexer.getCurToken() != tok_identifier)
      return parseError<PrototypeAST>("function name", "in prototype");

    // TODO: same name for controller and handler leads to conflict in MLIR code.
    std::string fnName(lexer.getId());
    lexer.consume(tok_identifier);
    
    std::optional<std::string> atom{};
    if (token == tok_handler && lexer.getCurToken() == tok_colon) {
      auto atomLit = parseAtom();
      atom = atomLit->getAtom();
    }

    if (lexer.getCurToken() != '(')
      return parseError<PrototypeAST>("(", "in prototype");
    lexer.consume(Token('('));

    std::vector<std::unique_ptr<VarDeclExprAST>> args;
    if (lexer.getCurToken() != ')') {
      do {
        VarType type;
        std::string name;

        // Parse either the name of the variable, or its type.
        std::string nameOrType(lexer.getId());
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_identifier);

        // If the next token is an identifier, we just parsed the type.
        if (lexer.getCurToken() == tok_identifier) {
          type.name = std::move(nameOrType);

          // Parse the name.
          name = std::string(lexer.getId());
          lexer.consume(tok_identifier);
        } else {
          // Otherwise, we just parsed the name.
          name = std::move(nameOrType);
        }

        args.push_back(
            std::make_unique<VarDeclExprAST>(std::move(loc), name, type));
        if (lexer.getCurToken() != ',')
          break;
        lexer.consume(Token(','));
        if (lexer.getCurToken() != tok_identifier)
          return parseError<PrototypeAST>(
              "identifier", "after ',' in function parameter list");
      } while (true);
    }
    if (lexer.getCurToken() != ')')
      return parseError<PrototypeAST>(")", "to end function prototype");

    // success.
    lexer.consume(Token(')'));
    return std::make_unique<PrototypeAST>(std::move(loc), fnName,
                                          std::move(args), atom, type);
  }

  /// Parse a function definition, we expect a prototype initiated with the
  /// `def|handler|controller` keyword, followed by a block containing a list of expressions.
  ///
  /// definition ::= prototype block
  std::unique_ptr<FunctionAST> parseDefinition(Token token) {
    auto proto = parsePrototype(token);
    if (!proto)
      return nullptr;

    if (auto block = parseBlock())
      return std::make_unique<FunctionAST>(std::move(proto), std::move(block));
    return nullptr;
  }

  /// Parse a struct definition, we expect a struct initiated with the
  /// `struct` keyword, followed by a block containing a list of variable
  /// declarations.
  ///
  /// definition ::= `struct` identifier `{` decl+ `}`
  std::unique_ptr<RecordAST> parseStruct(Token token_type) {
    auto loc = lexer.getLastLocation();
    lexer.consume(token_type);
    if (lexer.getCurToken() != tok_identifier)
      return parseError<StructAST>("name", "in struct definition");
    std::string name(lexer.getId());
    lexer.consume(tok_identifier);

    // Parse: '{'
    if (lexer.getCurToken() != '{')
      return parseError<StructAST>("{", "in struct definition");
    lexer.consume(Token('{'));

    // Parse: decl+
    std::vector<std::unique_ptr<VarDeclExprAST>> decls;
    while (lexer.getCurToken() != '}') {
      auto decl = parseDeclaration(/*requiresInitializer=*/false);
      if (!decl)
        return nullptr;
      decls.push_back(std::move(decl));

      if (lexer.getCurToken() != ';')
        return parseError<StructAST>(";",
                                     "after variable in struct definition");
      lexer.consume(Token(';'));
    }

    // Parse: '}'
    lexer.consume(Token('}'));
    return std::make_unique<StructAST>(token_type == tok_event, loc, name, std::move(decls));
  }

  /// Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    if (!isascii(lexer.getCurToken()))
      return -1;

    // 1 is lowest precedence.
    switch (static_cast<char>(lexer.getCurToken())) {
    case '=':
      return 10;
    case '-':
      return 20;
    case '+':
      return 20;
    case '*':
      return 40;
    case '/':
      return 40;
    default:
      return -1;
    }
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }
};

} // namespace toy

#endif // EP2_PARSER_H
