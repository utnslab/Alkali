
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

/*
Collect all the information from EP2 IR that will not exist after lowering to
Netronome.
*/

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

  // get all the struct definitions to emit later
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

  // Make sure structDefs are unique.
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

  // Make context struct
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

  // Get event structs, add to our struct emission list.
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

  // Generate event queues
  auto getOpd = [&](mlir::Value opd){
    return cast<ep2::ConstantOp>(opd.getDefiningOp()).getValue().cast<IntegerAttr>().getValue().getSExtValue();
  };
  module->walk([&](ep2::FuncOp funcOp) {
    if (funcOp.isExtern()) {
      return;
    }
    if (funcOp->getAttr("type").cast<StringAttr>().getValue() == "controller" && !funcOp.isExtern()) {
      funcOp->walk([&](ep2::ConnectOp op) {
        assert(op.getMethod() == "Queue" || op.getMethod() == "PartitionByScope");
        std::string eventName = funcOp->getAttr("event").cast<mlir::StringAttr>().getValue().str();
        auto& qInfo = this->eventQueues[eventName];
        
        if (op.getMethod() == "Queue" && op.getParameters()) {
          qInfo.size = op.getParameters()->getValue()[0].cast<mlir::IntegerAttr>().getValue().getSExtValue();
        } else {
          qInfo.size = 256;
        }

        qInfo.memType = MemType::CLS;
        qInfo.replicas.push_back(cast<ep2::ConstantOp>(
          op.getOuts()[0].getDefiningOp()).getValue().cast<ep2::PortAttr>().getInstance());
      });
    } else {
      std::string eventName = funcOp->getAttr("event").cast<StringAttr>().getValue().str();
      std::string stageName = funcOp->hasAttr("atom") ? funcOp->getAttr("atom").cast<StringAttr>().getValue().str() : funcOp->getAttr("event").cast<StringAttr>().getValue().str();
      // Just a simple encoding to allow combining eventName + stageName
      this->eventAllocs[eventName + "_a_" + stageName + "_a_" + funcOp.getName().str()] = {};
    }
  });

  HandlerDependencyAnalysis& hda = am.getAnalysis<HandlerDependencyAnalysis>();
  this->eventDeps = hda.eventDeps;

  LocalAllocAnalysis& laa = am.getAnalysis<LocalAllocAnalysis>();
  for (const auto& pr : laa.localAllocs) {
    mlir::Operation* funcOp = getParentFunction(pr.first->getParentOp());
    std::string eventName = funcOp->getAttr("event").cast<StringAttr>().getValue().str();
    std::string stageName = funcOp->hasAttr("atom") ? funcOp->getAttr("atom").cast<StringAttr>().getValue().str() : funcOp->getAttr("event").cast<StringAttr>().getValue().str();

    mlir::Type ty = isa<ep2::StoreOp>(pr.first) ? pr.first->getOperand(1).getType() : pr.first->getResult(0).getType();

    if (isa<ep2::StructType>(ty) && !isa<ep2::StoreOp>(pr.first)) {
      std::pair<std::string, std::string> prOut;
      prOut.first = cast<ep2::StructType>(ty).getName().str();
      prOut.second = pr.second;
      this->eventAllocs[eventName + "_a_" + stageName + "_a_" + cast<ep2::FuncOp>(funcOp).getName().str()].push_back(prOut);
    }
  }

  // Get local tables (note these init's are replicated per handler, thus per ME).
  module->walk([&](ep2::InitOp initOp){
    if (isa<ep2::TableType>(initOp->getResult(0).getType())) {
      TableInfo ti = getTableStr(cast<ep2::TableType>(initOp->getResult(0).getType()));
      ti.isLocal = true;
      if (this->tableInfos.find(ti.tableType) == this->tableInfos.end()) {
        this->tableInfos[ti.tableType].first = ti;
      }
      this->tableInfos[ti.tableType].second.push_back(laa.localAllocs[initOp]);
    }
  });

  // Get global tables (allocated in CLS, assume never replicated).
  module->walk([&](ep2::GlobalOp op) {
    if (isa<ep2::TableType>(op.getOutput().getType())) {
      TableInfo ti = getTableStr(cast<ep2::TableType>(op.getOutput().getType()));
      ti.isLocal = false;
      if (this->tableInfos.find(ti.tableType) == this->tableInfos.end()) {
        this->tableInfos[ti.tableType].first = ti;
      }
      this->tableInfos[ti.tableType].second.push_back(op.getName().str());
    }
  });
}

void CollectHeaderPass::runOnOperation() {
  getAnalysis<CollectInfoAnalysis>();
  markAnalysesPreserved<CollectInfoAnalysis>();
}

}
}
