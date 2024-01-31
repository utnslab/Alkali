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
        printf("Info: extern_inouts %s\n", e.first.c_str());
        wires.push_back(e.second);
    }
    emitModuleParameter(file, wires);
    handlerDependencyAnalysis->dump();

    mlir::DenseMap<std::tuple<int,  mlir::ep2::FuncOp, llvm::StringRef>, struct top_handler_inout_wires> handler_replcate_inwires, handler_replcates_outwires;

    for(auto global_table: global_tables){
        auto module_name = global_table.second.module_name;
        auto name = global_table.second.name;
        auto ports = global_table.second.ports;
        auto params = global_table.second.params;
        auto lookup_wires = global_table.second.lookup_wires;
        auto update_wires = global_table.second.update_wires;
        emitModuleCall(file, module_name, name, ports, params);
        for(auto &w : lookup_wires){
            emitwire(file, w);
        }
        for(auto &w : update_wires){
            emitwire(file, w);
        }
    }

    module->walk([&](ep2::FuncOp funcOp) {
        auto functype = funcOp->getAttr("type").cast<StringAttr>().getValue().str();
        if (functype == "handler" && !funcOp.isExtern()) {
            int replicate_size = 1;
            if(funcOp->hasAttr("instances"))
                replicate_size = funcOp->getAttr("instances").cast<ArrayAttr>().size();
            
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
                }

                for(auto &e : global_state_ports[funcOp]){
                    ports.push_back(e);
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
            if(!p.second.if_extern)
                printf("Warning: input event %s not connected\n", std::get<2>(p.first).str().c_str());
            assert(p.second.if_extern);
            auto event_name = std::get<2>(p.first);
            int wid = 0;
            for(auto &w : p.second.event_wires){
                auto extern_arg_name = getExternArgName(event_name.str(), wid);
                auto extern_inout = extern_inouts[extern_arg_name];
                struct wire_config extern_in_w = {AXIS, extern_arg_name, "", false, "", true, .axis = extern_inout.axis};
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
                struct wire_config extern_out_w = {AXIS, extern_arg_name, "", false, "", true, .axis = extern_inout.axis};
                wire_assignments.push_back({-1, -1, -1, -1, w, extern_out_w}); 
                wid++;
            }
        }
    }

    for(auto &e : wire_assignments){
        emitwireassign(file, e);
    }
    
    file <<  "\nendmodule\n";
}


} // namespace ep2
} // namespace mlir
