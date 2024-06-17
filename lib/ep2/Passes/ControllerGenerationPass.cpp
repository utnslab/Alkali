#include "mlir/IR/BuiltinDialect.h"

#include "ep2/dialect/Analysis/Passes.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

#include <optional>

namespace mlir {
namespace ep2 {

void ControllerGenerationPass::runOnOperation() {
  auto moduleOp = getOperation();
  OpBuilder builder(getOperation());
  auto &dependency = getAnalysis<HandlerDependencyAnalysis>();
  moduleOp.walk([&](FuncOp funcOp) {
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    bool if_need_generation = false;
    Value in_template, out_template;
    int in_instances, out_instances;

    in_instances = 1;
    out_instances = 1;

    mlir::scf::ValueVector ins, outs;

    if (funcOp.isController()) {
      funcOp->walk([&](ConstantOp constantOp) {
        if (auto portType = constantOp.getType().dyn_cast<ep2::PortType>()) {
          auto value = constantOp.getResult();
          auto portAttr = constantOp.getValue().cast<PortAttr>();

          HandlerDependencyAnalysis::HandlerFullName fullname{portAttr.getHandler().str(), portAttr.getAtom().str()};
          FuncOp handler = dependency.handlersMap[fullname];
          if(handler == nullptr)
          {
            printf("Error: handler %s not found in controller gen\n", portAttr.getHandler().str().c_str());
            assert(false);
          }

          if (portAttr.getInstance() == -1) {
            if (if_need_generation == false)
              if_need_generation = true;

            if (portType.getIn()) {
              if(handler->hasAttr("instances") == false && handler.isExtern())
                  in_instances = 1;
              else
                  in_instances = cast<mlir::ArrayAttr>(handler->getAttr("instances")).getValue().size();
              assert(in_instances > 0);
              for (int i = 0; i < in_instances; i++) {
                auto tmpportAttr = builder.getAttr<PortAttr>(
                    portAttr.getHandler(), portAttr.getAtom(), i);
                auto in_port = builder.create<ConstantOp>(
                    constantOp.getLoc(), portType, tmpportAttr);
                ins.push_back(in_port.getResult());
              }
            }

            if (portType.getOut())
              for (int i = 0; i < out_instances; i++) {
                if(handler->hasAttr("instances") == false && handler.isExtern())
                  out_instances = 1;
                else
                  out_instances = cast<mlir::ArrayAttr>(handler->getAttr("instances")).getValue().size();
                assert(out_instances > 0);
                auto tmpportAttr = builder.getAttr<PortAttr>(
                    portAttr.getHandler(), portAttr.getAtom(), i);
                auto out_port = builder.create<ConstantOp>(
                    constantOp.getLoc(), portType, tmpportAttr);
                outs.push_back(out_port.getResult());
              }
          }
        }
      });

      if (if_need_generation) {

        funcOp->walk([&](ConnectOp connectOp) {
            builder.create<ConnectOp>(connectOp.getLoc(), connectOp.getMethod(),ins, outs, connectOp.getParametersAttr());
            connectOp.erase();
        });

        funcOp->walk([&](ConstantOp constantOp) {
          if (auto portType = constantOp.getType().dyn_cast<ep2::PortType>()) {
            auto portAttr = constantOp.getValue().cast<PortAttr>();
            if (portAttr.getInstance() == -1) {
                constantOp.erase();
            }
          }
        });
        //     // first earse all connect
        //     funcOp->walk([&](ConnectOp connectOp) {
        //       connectOp.erase();
        //     });
        //     // then earse all constant
        //     funcOp->walk([&](ConstantOp constantOp) {
        //       constantOp.erase();
        //     });

        //     // then add generated constance template
        //     for (int i = 0; i < in_instances; i++) {

        //         auto in_port = builder.create<ConstantOp>(funcOp.getLoc(),
        //         in_template.getType(), in_template.getAttr());

        //     }
        // //   funcOp->walk(
        // //       [&](ConnectOp connectop) {
        // //     });
      }
    }
  });
}

} // namespace ep2
} // namespace mlir