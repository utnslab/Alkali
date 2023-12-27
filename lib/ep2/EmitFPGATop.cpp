#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"
#include "ep2/dialect/FPGAPasses.h"
#include "mlir/IR/BuiltinDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <fstream>
#include <iostream>
#include <list>
#include <string>

using namespace mlir;

namespace mlir {
namespace ep2 {

void EmitFPGAPass::emitTop() {
  std::ofstream file("__ep2top.sv");
    file << "module ep2top" << "#()\n";
    emitModuleParameter(file, extern_inouts);
    
    for (auto &[handler, edges] : handlerDependencyAnalysis->graph) {
        Operation *op = handler;
        auto funcOp = dyn_cast<FuncOp>(op);
        if(funcOp->hasAttr("extern"))
            continue;
        // currently we only emit connectivity for handler
        if (funcOp->getAttr("type").cast<StringAttr>().getValue().str() != "handler") {
            printf("Error: Currently only support handler type\n");
            assert(false);
        }
    
        // emit module for this func op
        std::list<struct module_port_config> ports;
        auto handler_edge = handler_edge_map[funcOp];
        
        // // CHECKING if logged handler_edge matches with handlerDependencyAnalysis's result; 
        if(handler_edge.size() -1 != edges.size()){
            printf("Error: handler_edge size does not match with handlerDependencyAnalysis's result: %ld- %ld\n", handler_edge.size() - 1, edges.size());
            funcOp.dump();
            handlerDependencyAnalysis->dump();
            assert(false);
        }
        printf("handler_edge size matches with handlerDependencyAnalysis's result: %ld- %ld\n", handler_edge.size() , edges.size());
        int eid = 0;
        // first emit edge wires.
        for(auto &e : handler_edge) {
           int wid = 0;
            // TODO: when supporting global table, type is not AXIS
            for(auto &w : e.event_wires) {
                if(e.direction == OUT){
                    auto target_func = dyn_cast<FuncOp>((Operation *)edges[e.id]);
                    if(!target_func){
                        funcOp.dump();
                        printf("Error:%d out port's target_func is null\n", e.id);
                        assert(false);
                    }
                    std::string out_wire_name;
                    if(e.if_extern)
                        out_wire_name = w.name;
                    else
                    {   
                        // 0 means target func's incoming event
                        out_wire_name = handler_edge_map[target_func][0].event_wires[wid].name;
                    }
                    struct module_port_config port_config = {AXIS, {out_wire_name}, "", w.name, .axis = w.axis};
                    ports.push_back(port_config);
                }
                else if (e.direction == IN){ // TODO: CHECK extern
                    struct module_port_config port_config = {AXIS, {w.name}, "", w.name, .axis = w.axis};
                    ports.push_back(port_config);
                    if(!e.if_extern)
                        emitwire(file, w);
                }

                wid++;
            }
            eid++;
        }

        // then emit handler's module defination
        std::list<struct module_param_config> params;
        emitModuleCall(file, funcOp.getName().str(), funcOp.getName().str(), ports, params);
        
        

    }
    file <<  "\nendmodule\n";
}


} // namespace ep2
} // namespace mlir
