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
    std::vector<struct inout_config> wires;
    for(auto &e : extern_inouts){
        wires.push_back(e.second);
    }
    emitModuleParameter(file, wires);
    handlerDependencyAnalysis->dump();

    mlir::DenseMap<std::tuple<int,  mlir::ep2::FuncOp, llvm::StringRef>, struct top_handler_inout_wires> handler_replcate_inwires, handler_replcates_outwires;

    module->walk([&](ep2::FuncOp funcOp) {
        auto functype = funcOp->getAttr("type").cast<StringAttr>().getValue().str();
        if (functype == "handler" && !funcOp.isExtern()) {
            auto replicate_size = funcOp->getAttr("instances").cast<ArrayAttr>().size();
            for(int replicate_index = 0; replicate_index < replicate_size; replicate_index++){
                std::list<struct module_port_config> ports;
                for(auto &e : handler_in_edge_map[funcOp]){
                    auto in_wires = e.event_wires;
                    int wid = 0;
                    for(auto &w : e.event_wires) {
                        in_wires[wid].name = w.name + "_r" + std::to_string(replicate_index);
                        struct module_port_config port_config = {AXIS, {in_wires[wid].name}, "", w.name, .axis = w.axis};
                        ports.push_back(port_config);
                        emitwire(file, in_wires[wid]);
                        wid ++;
                    }
                    handler_replcate_inwires[{replicate_index, funcOp, e.eventname}] = {e.if_extern, false, in_wires};
                    // printf("Info: handler_replcate_inwires %s, size: %ld\n", funcOp.getName().str().c_str(), in_wires.size());
                }

                for(auto &e : handler_out_edge_map[funcOp]){
                    auto out_wires = e.event_wires;
                    int wid = 0;
                    for(auto &w : e.event_wires) {
                        out_wires[wid].name = w.name + "_r" + std::to_string(replicate_index);
                        struct module_port_config port_config = {AXIS, {out_wires[wid].name}, "", w.name, .axis = w.axis};
                        ports.push_back(port_config);
                        emitwire(file, out_wires[wid]);
                        wid ++;
                    }
                    handler_replcates_outwires[{replicate_index, funcOp, e.eventname}] = {e.if_extern, false, out_wires};
                    printf("Info: handler_replcates_outwires %s, eventname %s, size: %ld\n", funcOp.getName().str().c_str(), e.eventname.c_str(), out_wires.size());
                }

                std::list<struct module_param_config> params;
                emitModuleCall(file, funcOp.getName().str(), assign_name(funcOp.getName().str()), ports, params);
            }
        }
    });

    std::vector<struct wire_assign_config> wire_assignments;
   module->walk([&](ep2::FuncOp funcOp) {
        auto functype = funcOp->getAttr("type").cast<StringAttr>().getValue().str();
        if (functype == "controller") {
            printf("--------%s----------\n", funcOp.getName().str().c_str());
            auto in_edges = ctrl_ins[funcOp];
            auto out_edges = ctrl_outs[funcOp];
            std::list<struct module_port_config> ports;
            for(auto &e : in_edges){
                auto event_wires = e.second.event_wires;
                int wid = 0;
                for(auto &w : event_wires) {
                    struct module_port_config port_config = {AXIS, {w.name}, "", w.name, .axis = w.axis};
                    if(!e.second.if_connect_to_extern)
                    {    
                        emitwire(file, w);
                        auto src_wires = handler_replcates_outwires[{e.second.replicate_index, e.second.funcop, e.second.eventname}].event_wires;
                        printf("src_wires size: %ld - %d\n", src_wires.size(), event_wires.size());
                        printf("src funcop name: %s, rid: %d\n", e.second.funcop.getName().str().c_str(), e.second.replicate_index);
                        auto src_handler_out_wire = src_wires[wid];
                        wire_assignments.push_back({-1, -1, -1, -1, src_handler_out_wire, w}); 

                        handler_replcates_outwires[{e.second.replicate_index, e.second.funcop, e.second.eventname}].if_connected = true;
                    }
                    ports.push_back(port_config);                    
                    wid ++;
                }
            }
            for(auto &e: out_edges){
                auto event_wires = e.second.event_wires;
                int wid = 0;
                for(auto &w : event_wires) {
                    struct module_port_config port_config = {AXIS, {w.name}, "", w.name, .axis = w.axis};
                    if(!e.second.if_connect_to_extern)
                    {    
                        emitwire(file, w);
                        auto dst_handler_in_wire = handler_replcate_inwires[{e.second.replicate_index, e.second.funcop, e.second.eventname}].event_wires[wid];
                        wire_assignments.push_back({-1, -1, -1, -1, w, dst_handler_in_wire}); 

                        handler_replcate_inwires[{e.second.replicate_index, e.second.funcop, e.second.eventname}].if_connected = true;
                    }
                    ports.push_back(port_config);
                    wid++;
                }
            }
            std::list<struct module_param_config> params;
            emitModuleCall(file, funcOp.getName().str(), assign_name(funcOp.getName().str()), ports, params);
        }
    });

    for(auto &p: handler_replcate_inwires){
        if(!p.second.if_connected){
            assert(p.second.if_extern);
            auto event_name = std::get<2>(p.first);
            int wid = 0;
            for(auto &w : p.second.event_wires){
                auto extern_arg_name = getExternArgName(event_name.str(), wid);
                auto extern_inout = extern_inouts[extern_arg_name];
                struct wire_config extern_in_w = {AXIS, extern_arg_name, "", false, -1, true, .axis = extern_inout.axis};
                wire_assignments.push_back({-1, -1, -1, -1, extern_in_w, w}); 
                wid++;
            }
        }
    }

    for(auto &p : handler_replcates_outwires){
        if(!p.second.if_connected){
            assert(p.second.if_extern);
            auto event_name = std::get<2>(p.first);
            int wid = 0;
            for(auto &w : p.second.event_wires){
                auto extern_arg_name = getExternArgName(event_name.str(), wid);
                auto extern_inout = extern_inouts[extern_arg_name];
                struct wire_config extern_out_w = {AXIS, extern_arg_name, "", false, -1, true, .axis = extern_inout.axis};
                wire_assignments.push_back({-1, -1, -1, -1, w, extern_out_w}); 
                wid++;
            }
        }
    }



    for(auto &e : wire_assignments){
        emitwireassign(file, e);
    }
    
    

    // module->walk([&](ep2::FuncOp funcOp) {
    //     auto functype = funcOp->getAttr("type").cast<StringAttr>().getValue().str();
    //     if (functype == "controller") {
    //         auto targetfunc = ctrl_func_to_target_handler[funcOp];
    //         auto in_edges = controller_in_edge_map[targetfunc];
    //         auto out_edges = controller_out_edge_map[targetfunc];
    //         std::list<struct module_port_config> ports;
    //         for(auto &e : in_edges){
    //             auto event_wires = e.event_wires;
    //             assert(!e.if_extern);
    //             for(auto &w : event_wires) {
    //                 struct module_port_config port_config = {AXIS, {w.name}, "", w.name, .axis = w.axis};
    //                 ports.push_back(port_config);
    //             }
    //         }
    //         for(auto &e: out_edges){
    //             auto event_wires = e.event_wires;
    //             assert(!e.if_extern);
    //             for(auto &w : event_wires) {
    //                 struct module_port_config port_config = {AXIS, {w.name}, "", w.name, .axis = w.axis};
    //                 ports.push_back(port_config);
    //             }
    //         }
    //         std::list<struct module_param_config> params;
    //         emitModuleCall(file, funcOp.getName().str(), assign_name(funcOp.getName().str()), ports, params);
    //     }
    // });
            

    // for (auto &[handler, edges] : handlerDependencyAnalysis->graph) {
    //     Operation *op = handler;
    //     auto funcOp = dyn_cast<FuncOp>(op);

    //     if(funcOp.isExtern())
    //         continue;
    //     // currently we only emit connectivity for handler
    //     if (funcOp->getAttr("type").cast<StringAttr>().getValue().str() != "handler") {
    //         printf("Error: Currently only support handler type\n");
    //         assert(false);
    //     }
    
    //     // emit module for this func op
        
    //     auto handler_in_edge = handler_in_edge_map[funcOp];
    //     auto handler_out_edge = handler_out_edge_map[funcOp];
        
    //     // // CHECKING if logged handler_edge matches with handlerDependencyAnalysis's result; 
    //     if(handler_out_edge.size() != edges.size()){
    //         printf("Error: handler_edge size does not match with handlerDependencyAnalysis's result: %ld- %ld\n", handler_out_edge.size(), edges.size());
    //         funcOp.dump();
    //         handlerDependencyAnalysis->dump();
    //         assert(false);
    //     }

    //     int eid = 0;
    //     auto replicate_size = funcOp->getAttr("instances").cast<ArrayAttr>().size();
    //     for(int replicate_index = 0; replicate_index < replicate_size; replicate_index++){
    //         std::list<struct module_port_config> ports;
    //         for(auto &e : handler_in_edge){
    //             auto event_wires = e.event_wires;
    //             bool if_extern = e.if_extern;
    //             int wid = 0;
    //             for(auto &w : event_wires) {
    //                 auto in_wire = w;
    //                 if(controller_out_edge_map.find(funcOp) != controller_out_edge_map.end())
    //                 {    
    //                     in_wire = controller_out_edge_map[funcOp][replicate_index].event_wires[wid];
    //                 }
    //                 struct module_port_config port_config = {AXIS, {in_wire.name}, "", w.name, .axis = w.axis};
    //                 ports.push_back(port_config);
    //                 if(!if_extern)
    //                     emitwire(file, in_wire);
    //                 wid ++;
    //             }
    //         }

    //         for(auto &e : handler_out_edge){
    //             auto event_wires = e.event_wires;
    //             bool if_extern = e.if_extern;
    //             auto next_func = dyn_cast<FuncOp>((Operation *)edges[e.id]);

    //             if(!next_func){
    //                 funcOp.dump();
    //                 printf("Error:%d out port's target_func is null\n", e.id);
    //                 assert(false);
    //             }

    //             if(controller_in_edge_map.find(next_func) != controller_in_edge_map.end()){
    //                 if_extern = false;
    //             }

    //             int wid = 0;
    //             for(auto &w : event_wires) {

    //                 std::string out_wire_name;
    //                 if(if_extern)
    //                     out_wire_name = w.name;
    //                 else if(controller_in_edge_map.find(next_func) != controller_in_edge_map.end()) // has controller
    //                 {
    //                     auto ctrl_wire = controller_in_edge_map[next_func].back().event_wires[wid];
    //                     out_wire_name = ctrl_wire.name;
    //                     emitwire(file, ctrl_wire); // emit conteroller's wire defination
    //                 }
    //                 else
    //                 {   
    //                     // 0 means target func's incoming event
    //                     auto ctrl_wire = handler_in_edge_map[next_func][0];
    //                     out_wire_name = ctrl_wire.event_wires[wid].name;
    //                 }
    //                 struct module_port_config port_config = {AXIS, {out_wire_name}, "", w.name, .axis = w.axis};
    //                 ports.push_back(port_config);
    //                 wid++;

    //             }

    //             // remove the last edge in controller_in_edge_map
    //             if(controller_in_edge_map.find(next_func) != controller_in_edge_map.end()){
    //                 controller_in_edge_map[next_func].pop_back();
    //             }
    //         }
            
    //         std::list<struct module_param_config> params;
    //         emitModuleCall(file, funcOp.getName().str(), assign_name(funcOp.getName().str()), ports, params);
    //     }

    // }

    


    
    file <<  "\nendmodule\n";
}


} // namespace ep2
} // namespace mlir
