
#ifndef _UTILS_H_
#define _UTILS_H_

#include "mlir/IR/BuiltinDialect.h"

#include "ep2/lang/Lexer.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
namespace ep2 {

static TableInfo getTableStr(ep2::TableType type) {
  auto getIntTypeStr = [](mlir::Type intTy) {
    std::string out;
    llvm::raw_string_ostream oss(out);
    intTy.print(oss);
    return oss.str();
  };

  mlir::Type keyTy = type.getKeyType();
  std::string keyStr = "";
  mlir::Type valTy = type.getValueType();
  std::string valStr = "";

  if (isa<mlir::IntegerType>(keyTy)) {
    keyStr = getIntTypeStr(keyTy);
  } else {
    assert(false && "Unsupported key type");
  }

  if (isa<ep2::StructType>(valTy)) {
    valStr = cast<ep2::StructType>(valTy).getName().str();
  } else if (isa<ep2::BufferType>(valTy)) {
    valStr = "__buf_t";
  } else if (isa<mlir::IntegerType>(valTy)) {
    valStr = getIntTypeStr(valTy);
  } else {
    assert(false && "Unsupported value type");
  }

  TableInfo info;
  info.tableType = "struct table_" + keyStr + "_" + valStr + "_" + std::to_string(type.getSize()) + "_t";
  info.keyType = keyStr;

  if (!isa<mlir::IntegerType>(valTy)) {
    valStr = "struct " + valStr;
  }
  info.valType = valStr;
  info.size = type.getSize();
  return info;
}

static const char* toStringDecl(MemType ty) {
  switch (ty) {
    case MemType::LMEM: return "LMEM";
    case MemType::CLS: return "CLS";
    case MemType::CTM: return "CTM";
    case MemType::IMEM: return "IMEM";
    case MemType::EMEM: return "EMEM";
    default: {
      assert(false && "Unsupported memtype");
      return nullptr;
    }
  }
}

static const char* toStringFunc(MemType ty) {
  switch (ty) {
    case MemType::LMEM: return "lmem";
    case MemType::CLS: return "cls";
    case MemType::CTM: return "ctm";
    case MemType::IMEM: return "mem";
    case MemType::EMEM: return "mem";
    default: {
      assert(false && "Unsupported memtype");
      return nullptr;
    }
  }
}

}
}

#endif
