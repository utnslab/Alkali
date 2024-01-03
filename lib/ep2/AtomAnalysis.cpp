
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/EquivalenceClasses.h"

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace ep2 {

AtomAnalysis::AtomAnalysis(Operation* module, AnalysisManager& am) {
  auto& m = this->atomToNum;
  size_t atomCtr = 0;
  module->walk([&](FuncOp op) {
    if (op->hasAttr("atom")) {
      llvm::StringRef k = op->getAttr("atom").cast<StringAttr>().getValue();
      llvm::StringRef event = op->getAttr("event").cast<StringAttr>().getValue();
      if (m.find(k) == m.end()) {
        m.try_emplace(k, std::pair<std::string, size_t>{event.str(), atomCtr++});
      }
    }
  });
}

}
}
