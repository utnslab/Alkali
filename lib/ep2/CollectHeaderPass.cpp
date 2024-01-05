
#include "mlir/IR/BuiltinDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#include <cassert>

#include "Utils.h"

using namespace mlir;

namespace mlir {
namespace ep2 {

static mlir::Operation* getParentFunction(mlir::Operation* op) {
  while (!isa<FunctionOpInterface>(op)) {
    op = op->getParentOp();
    assert(op != nullptr);
  }
  return op;
}

CollectInfoAnalysis::CollectInfoAnalysis(Operation* module, AnalysisManager& am) {
  auto builder = OpBuilder(module);

  auto charType = builder.getI8Type();
  auto charPtrType = LLVM::LLVMPointerType::get(charType);
  auto u32Type = builder.getI32Type();
  auto bufType = LLVM::LLVMStructType::getIdentified(module->getContext(), "__buf_t");
  llvm::SmallVector<mlir::Type> sTypes = {charPtrType, u32Type};
  auto bRes = bufType.setBody(sTypes, true);
  assert(bRes.succeeded());

  mlir::LLVMTypeConverter llvmConv(module->getContext());
  auto makeLLVMStruct = [&](mlir::Type ty) {
    auto sTy = mlir::LLVM::LLVMStructType::getIdentified(module->getContext(), cast<ep2::StructType>(ty).getName());
    llvm::SmallVector<mlir::Type> sTypes;
    for (const auto& eTy : cast<ep2::StructType>(ty).getElementTypes()) {
      sTypes.push_back(llvmConv.convertType(eTy));
    }
    auto sRes = sTy.setBody(sTypes, true);
    assert(sRes.succeeded());
    return sTy;
  };

  std::vector<ep2::StructType> realStructTypes;
  auto appendToStructTypes = [&](ep2::StructType strTy) {
    if (strTy.getIsEvent()) return;
    for (ep2::StructType ty : realStructTypes) {
      if (ty == strTy) {
        return;
      }
    }
    realStructTypes.push_back(strTy);
  };

  module->walk([&](mlir::Operation* op){
    for (const auto& v : op->getOperands()) {
      if (isa<ep2::StructType>(v.getType())) {
        appendToStructTypes(cast<ep2::StructType>(v.getType()));
      }
    }
    for (const auto& v : op->getResults()) {
      if (isa<ep2::StructType>(v.getType())) {
        appendToStructTypes(cast<ep2::StructType>(v.getType()));
      }
    }
  });

  for (ep2::StructType ty : realStructTypes) {
    auto lty = mlir::LLVM::LLVMStructType::getIdentified(module->getContext(), ty.getName());
    llvm::SmallVector<mlir::Type> types;
    for (int i = 0; i<ty.getNumElementTypes(); ++i) {
      auto ety = ty.getElementTypes()[i];
      if (ety.isIntOrFloat()) {
        typeBitWidths.insert(ety.getIntOrFloatBitWidth());
      }
      types.push_back(llvmConv.convertType(ety));
    }
    lty.setBody(types, true);
    this->structDefs.emplace_back(ty.getName().str(), lty);
  }


  auto &contextAnalysis = am.getAnalysis<ContextBufferizationAnalysis>();
  std::string context_prefix = "context_chain_";
  std::string context_suffix = "_t";
  unsigned ctx_id = 1;
  for (const auto& ctx : contextAnalysis.contextTables) {
    std::string name = context_prefix + std::to_string(ctx_id) + context_suffix;
    auto contextTy = mlir::LLVM::LLVMStructType::getIdentified(module->getContext(), name);
    llvm::SmallVector<mlir::Type> contextTypes(ctx.size());
    for (const auto& field : ctx) {
      mlir::Type ty = field.second.second;
      if (isa<ep2::BufferType>(ty)) {
        contextTypes[field.second.first] = bufType;
      } else if (isa<ep2::StructType>(ty)) {
        LLVM::LLVMStructType sTy = makeLLVMStruct(ty);
        contextTypes[field.second.first] = sTy;
      } else {
        assert(ty.isSignlessInteger());
        typeBitWidths.insert(ty.getIntOrFloatBitWidth());
        contextTypes[field.second.first] = llvmConv.convertType(ty);
      }
    }
    contextTy.setBody(contextTypes, true);
    this->structDefs.emplace_back(name, contextTy);
    ctx_id += 1;
  }

  auto lowerStructAnalysis = am.getAnalysis<LowerStructAnalysis>();
  std::vector<mlir::LLVM::LLVMStructType> emittedStructs;

  module->walk([&](ep2::FuncOp funcOp){
    if (funcOp.isExtern()) {
      return;
    }

    ArrayRef<LLVM::LLVMStructType> wrapperTypes = lowerStructAnalysis.getWrapperTypes(funcOp);
    std::string eventName = funcOp->getAttr("event").cast<StringAttr>().getValue().str();

    for (const mlir::LLVM::LLVMStructType& cwrapType : wrapperTypes) {
      mlir::LLVM::LLVMStructType& wrapType = const_cast<mlir::LLVM::LLVMStructType&>(cwrapType);
      bool doInsert = true;
      for (auto& old : emittedStructs) {
        if (wrapType == old) {
          doInsert = false;
          break;
        }
      }
      if (doInsert) {
        std::string name = "event_param_" + wrapType.getName().str();

        llvm::SmallVector<mlir::Type> types;
        auto sty = mlir::LLVM::LLVMStructType::getIdentified(module->getContext(), name);
        unsigned ctr = 0;
        for (mlir::Type ty : wrapType.getBody()) {
          if (isa<ep2::BufferType>(ty)) {
            types.push_back(bufType);
          } else if (isa<ep2::StructType>(ty)) {
            types.push_back(makeLLVMStruct(ty));
          } else {
            if (ty.isIntOrFloat()) {
              typeBitWidths.insert(ty.getIntOrFloatBitWidth());
            }
            types.push_back(llvmConv.convertType(ty));
          }
        }
        auto res = sty.setBody(types, true);
        assert(res.succeeded());
        this->structDefs.emplace_back(name, sty);
        emittedStructs.push_back(wrapType);
      }
    }
  });

  // generate event queues
  auto getOpd = [&](mlir::Value opd){
    return cast<ep2::ConstantOp>(opd.getDefiningOp()).getValue().cast<IntegerAttr>().getValue().getSExtValue();
  };
  module->walk([&](ep2::FuncOp funcOp) {
    if (funcOp.isExtern()) {
      return;
    }
    if (funcOp->getAttr("type").cast<StringAttr>().getValue() == "controller" && !funcOp.isExtern()) {
      funcOp->walk([&](ep2::ConnectOp op) {
        assert(op.getMethod() == "Queue");
        std::vector<int> replicas;
        for (mlir::Value arg : op.getOuts()) {
          auto port = cast<ep2::ConstantOp>(arg.getDefiningOp()).getValue().cast<ep2::PortAttr>();
          replicas.push_back(port.getInstance());
        }

        // TODO do not hardcode queue size=100
        std::string eventName = funcOp->getAttr("event").cast<mlir::StringAttr>().getValue().str();
        QueueInfo qInfo(MemType::CLS, 100, replicas);
        this->eventQueues.emplace(eventName, qInfo); 
      });
    } else {
      std::string eventName = funcOp->getAttr("event").cast<StringAttr>().getValue().str();
      std::string stageName = funcOp->hasAttr("atom") ? funcOp->getAttr("atom").cast<StringAttr>().getValue().str() : funcOp->getAttr("event").cast<StringAttr>().getValue().str();
      this->eventAllocs[eventName + "_a_" + stageName] = {};
    }
  });

  HandlerDependencyAnalysis& hda = am.getAnalysis<HandlerDependencyAnalysis>();
  this->eventDeps = hda.eventDeps;

  LocalAllocAnalysis& laa = am.getAnalysis<LocalAllocAnalysis>();
  for (const auto& pr : laa.localAllocs) {
    mlir::Operation* funcOp = getParentFunction(pr.first->getParentOp());
    std::string eventName = funcOp->getAttr("event").cast<StringAttr>().getValue().str();
    std::string stageName = funcOp->hasAttr("atom") ? funcOp->getAttr("atom").cast<StringAttr>().getValue().str() : funcOp->getAttr("event").cast<StringAttr>().getValue().str();

    if (isa<ep2::StructType>(pr.first->getResult(0).getType())) {
      std::pair<std::string, std::string> prOut;
      prOut.first = cast<ep2::StructType>(pr.first->getResult(0).getType()).getName().str();
      prOut.second = pr.second;
      this->eventAllocs[eventName + "_a_" + stageName].push_back(prOut);
    }
  }

  // tables
  module->walk([&](ep2::InitOp initOp){
    if (isa<ep2::TableType>(initOp->getResult(0).getType())) {
      TableInfo ti = getTableStr(cast<ep2::TableType>(initOp->getResult(0).getType()));
      for (const TableInfo& t : this->tableInfos) {
        if (t.tableType == ti.tableType) {
          return;
        }
      }
      ti.tableId = laa.localAllocs[initOp];
      this->tableInfos.push_back(ti);
    }
  });
}

void CollectHeaderPass::runOnOperation() {
  getAnalysis<CollectInfoAnalysis>();
  markAnalysesPreserved<CollectInfoAnalysis>();
}

}
}
