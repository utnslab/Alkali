
#include "mlir/IR/BuiltinDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

using namespace mlir;

namespace mlir {
namespace ep2 {

void CollectHeaderPass::runOnOperation() {
  auto module = getOperation();
  auto builder = OpBuilder(getOperation());

  auto &contextAnalysis = getAnalysis<ContextBufferizationAnalysis>();
  mlir::LLVMTypeConverter llvmConv(&getContext());
  std::string context_prefix = "context_chain_";
  std::string context_suffix = "_t";
  unsigned ctx_id = 1;
  for (const auto& ctx : contextAnalysis.contextTables) {
    std::string name = context_prefix + std::to_string(ctx_id) + context_suffix;
    auto contextTy = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), name);
    llvm::SmallVector<mlir::Type> contextTypes(ctx.size());
    for (const auto& field : ctx) {
      mlir::Type ty = field.second.second;
      assert(ty.isSignlessInteger());
      contextTypes[field.second.first] = llvmConv.convertType(ty);
    }
    contextTy.setBody(contextTypes, true);
    info->structDefs.emplace_back(name, contextTy);
    ctx_id += 1;
  }

  std::vector<ep2::StructType> realStructTypes;
  module->walk([&](StructAccessOp op){
    assert(isa<ep2::StructType>(op.getInput().getType()));
    ep2::StructType strTy = op.getInput().getType();
    if (strTy.getIsEvent()) return;

    for (ep2::StructType ty : realStructTypes) {
      if (ty == strTy) {
        return;
      }
    }
    realStructTypes.push_back(op.getInput().getType());
  });

  for (ep2::StructType ty : realStructTypes) {
    auto lty = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), ty.getName());
    llvm::SmallVector<mlir::Type> types;
    for (int i = 0; i<ty.getNumElementTypes(); ++i) {
      types.push_back(llvmConv.convertType(ty.getElementTypes()[i]));
    }
    lty.setBody(types, true);
    info->structDefs.emplace_back(ty.getName().str(), lty);
  }

  auto lowerStructAnalysis = getAnalysis<LowerStructAnalysis>();
  std::vector<mlir::LLVM::LLVMStructType> emittedStructs;

  module->walk([&](ep2::FuncOp funcOp){
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
        auto charType = builder.getI8Type();
        auto charPtrType = LLVM::LLVMPointerType::get(charType);
        std::string name = "event_param_" + wrapType.getName().str();

        llvm::SmallVector<mlir::Type> types;
        auto sty = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), name);
        unsigned ctr = 0;
        for (mlir::Type ty : wrapType.getBody()) {
          if (isa<ep2::BufferType>(ty)) {
            types.push_back(charPtrType);
          } else {
            types.push_back(llvmConv.convertType(ty));
          }
        }
        auto res = sty.setBody(types, true);
        assert(res.succeeded());
        info->structDefs.emplace_back(name, sty);
        emittedStructs.push_back(wrapType);
      }
    }
  });

  // generate event queues
  auto getOpd = [&](mlir::Value opd){
    return cast<ep2::ConstantOp>(opd.getDefiningOp()).getValue().cast<IntegerAttr>().getValue().getSExtValue();
  };
  module->walk([&](ep2::FuncOp funcOp) {
    if (funcOp->getAttr("type").cast<StringAttr>().getValue() == "controller") {
      funcOp->walk([&](ep2::CallOp op){
        assert(op.getInputs().size() == 3);
        assert(getOpd(op.getOperand(1)) == 1);
        assert(getOpd(op.getOperand(2)) == 1);

        std::string eventName = funcOp->getAttr("event").cast<mlir::StringAttr>().getValue().str();
        std::pair<MemType, int> pr = {MemType::CLS, getOpd(op.getOperand(0))};
        info->eventQueues.emplace(eventName, pr); 
      });
    } else {
      std::string eventName = funcOp->getAttr("event").cast<StringAttr>().getValue().str();
      std::string stageName = funcOp->hasAttr("atom") ? funcOp->getAttr("atom").cast<StringAttr>().getValue().str() : funcOp->getAttr("event").cast<StringAttr>().getValue().str();
      info->eventAllocs[eventName + "_a_" + stageName] = {};
    }
  });

  HandlerDependencyAnalysis& hda = getAnalysis<HandlerDependencyAnalysis>();
  info->eventDeps = hda.eventDeps;

  LocalAllocAnalysis& laa = getAnalysis<LocalAllocAnalysis>();
  for (const auto& pr : laa.localAllocs) {
    mlir::Operation* funcOp = pr.first->getParentOp();
    std::string eventName = funcOp->getAttr("event").cast<StringAttr>().getValue().str();
    std::string stageName = funcOp->hasAttr("atom") ? funcOp->getAttr("atom").cast<StringAttr>().getValue().str() : funcOp->getAttr("event").cast<StringAttr>().getValue().str();

    std::pair<std::string, std::string> prOut;
    prOut.first = cast<ep2::StructType>(cast<ep2::ExtractOp>(pr.first).getOutput().getType()).getName().str();
    prOut.second = pr.second;
    info->eventAllocs[eventName + "_a_" + stageName].push_back(prOut);
  }
}

}
}
