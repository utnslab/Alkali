#include "mlir/IR/BuiltinDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
namespace ep2{

namespace {

} // hold util functions

LowerStructAnalysis::LowerStructAnalysis(Operation *op) {
    const std::string prefix = "__wrapper";
    auto context = op->getContext();
    if (!isa<ModuleOp>(op)) {
        op->emitError("expected ModuleOp");
        return;
    }

    // register events
    op->walk([&](ep2::FuncOp funcOp) {
        if (funcOp->getAttr("type").cast<StringAttr>().getValue() != "handler")
            return WalkResult::advance();

        auto argTypes = funcOp.getArgumentTypes();
        while (argTypes.size() > 0 && (isa<ep2::AtomType>(argTypes.front()) || isa<ep2::ContextType>(argTypes.front()))) {
            argTypes = argTypes.slice(1);
        }

        // TODO: checks
        auto eventName = funcOp->getAttr("event").dyn_cast<StringAttr>().getValue().str();
        std::string atomName = "";
        if (funcOp->hasAttr("atom"))
            atomName = funcOp->getAttr("atom").dyn_cast<StringAttr>().getValue().str();

        auto inputType = LLVM::LLVMStructType::getIdentified(context, eventName);
        if (inputType.getBody().size() == 0)
            inputType.setBody(argTypes, false);
        handlerTypes.insert({eventName, inputType});
        ioTypes.insert({{eventName, atomName}, {inputType}});

        // get returns
        funcOp->walk([&](ep2::ReturnOp returnOp) {
            if (returnOp->getNumOperands() == 0)
                return WalkResult::advance();
            auto retStruct = returnOp->getOperandTypes().front().dyn_cast<ep2::StructType>();
            auto structTypes = retStruct.getElementTypes();
            while (structTypes.size() > 0 && (isa<ep2::AtomType>(structTypes.front()) || isa<ep2::ContextType>(structTypes.front())))
                structTypes = structTypes.slice(1);

            // converting the call target to a struct
            auto genType = LLVM::LLVMStructType::getIdentified(context, retStruct.getName());
            // TODO: type check and skip patterns
            if (genType.getBody().size() == 0)
                genType.setBody(structTypes, false);
            
            ioTypes[{eventName, atomName}].push_back(genType);
            return WalkResult::advance();
        });
    });
}

ArrayRef<LLVM::LLVMStructType> LowerStructAnalysis::getWrapperTypes(ep2::FuncOp funcOp) {
    auto eventName = funcOp->getAttr("event").dyn_cast<StringAttr>().getValue().str();
    std::string atomName = "";
    if (funcOp->hasAttr("atom"))
        atomName = funcOp->getAttr("atom").dyn_cast<StringAttr>().getValue().str();
    return ioTypes[{eventName, atomName}];
}

} // namespace mlir
} // namespace ep2
