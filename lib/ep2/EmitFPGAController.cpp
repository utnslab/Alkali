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
#include <bit>

using namespace mlir;

namespace mlir {
namespace ep2 {
// struct inout_info {
//   FuncOp funcop;
//   int replicate_index;
// };

int event_wire_counts = 0;
// mlir::DenseMap<Value, struct inout_info> ins, outs;
FuncOp targetfunc = nullptr;
FuncOp srcfunc = nullptr;

void EmitFPGAPass::emitControllerInOut(std::ofstream &file, ep2::FuncOp funcOp) {
  for (auto &block : funcOp.getBody().getBlocks()) {
    std::string targetEvent = funcOp->getAttr("event").cast<StringAttr>().getValue().str();
    auto &dependency = getAnalysis<HandlerDependencyAnalysis>();

    funcOp->walk([&](ConstantOp constantOp) {
        if (auto portType = constantOp.getType().dyn_cast<ep2::PortType>()) {
            auto value = constantOp.getResult();
            auto portAttr = constantOp.getValue().cast<PortAttr>();
            HandlerDependencyAnalysis::HandlerFullName fullname{portAttr.getHandler().str(), portAttr.getAtom().str()};

            
            FuncOp handler = dependency.handlersMap[fullname];

            int instance = portAttr.getInstance();

            struct inout_info tmp = {handler, instance, targetEvent, {}, false};
            if (portType.getIn() && !portType.getOut())
            {    
                srcfunc = handler;
                // ins[value] = tmp;
                auto [it, _] = ctrl_ins.try_emplace(funcOp);
                it->second[value] = tmp;
            }
            else if (!portType.getIn() && portType.getOut())
            {    
                dependency.dump();
                targetfunc = handler;
                if(handler == nullptr)
                {    
                    printf("ERROR: handler is null\n");
                    assert(false);
                }
                
                // outs[value] = tmp;
                auto [it, _] = ctrl_outs.try_emplace(funcOp);
                it->second[value] = tmp;
                // ctrl_outs[funcOp][value] = tmp;
            }
        }
    });
    std::vector<struct inout_config> inout_wires;
    bool if_in_extern_event = !handlerDependencyAnalysis->hasPredecessor(targetfunc);
    bool if_out_extern_event = !handlerDependencyAnalysis->hasSuccessor(targetEvent);
    
    std::vector<struct wire_config>  event_wires;
    if(if_in_extern_event){
        event_wires = handler_in_edge_map[targetfunc][0].event_wires;
    }
    else if(if_out_extern_event){
        event_wires = handler_out_edge_map[srcfunc][0].event_wires;
    }
    else{
        event_wires = handler_in_edge_map[targetfunc][0].event_wires;
    }

    event_wire_counts = event_wires.size();

    int port_id = 0;

    for(auto &in : ctrl_ins[funcOp]){
        auto in_event_wires = event_wires;
        for (auto &wire : in_event_wires) {
            assert(wire.type == AXIS);
            if(!if_in_extern_event) // If extern, remain the oringinal name
                wire.name = "in_" + std::to_string(in.second.replicate_index) + '_' + wire.name;
            struct inout_config in_if = {IN, AXIS, wire.name, wire.debuginfo, wire.axis};
            inout_wires.push_back(in_if);
        }
        in.second.event_wires = in_event_wires;
        in.second.if_connect_to_extern = if_in_extern_event;
        // controller_in_edge_map[targetfunc].push_back({IN, port_id, if_in_extern_event, in_event_wires});
        port_id++;
    }

    for(auto &out : ctrl_outs[funcOp]){
        auto out_event_wires = event_wires;
        for (auto &wire : out_event_wires) {
            assert(wire.type == AXIS);
            if(!if_out_extern_event)  // If extern, remain the oringinal name
                wire.name = "out_" + std::to_string(out.second.replicate_index) + '_' + wire.name;
            struct inout_config out_if = {OUT, AXIS, wire.name, wire.debuginfo, wire.axis};
            inout_wires.push_back(out_if);
        }
        out.second.event_wires = out_event_wires;
        out.second.if_connect_to_extern = if_out_extern_event;
        // controller_out_edge_map[targetfunc].push_back({OUT, port_id, if_out_extern_event, out_event_wires});
        port_id++;
    }
    emitModuleParameter(file, inout_wires);

  }
}

void EmitFPGAPass::emitControllerMux(std::ofstream &file, ep2::ConnectOp conncetop){
    auto invs = conncetop.getIns();
    auto outvs = conncetop.getOuts();

    int in_count = invs.size();
    int out_count = outvs.size();

    assert((in_count >= 1) && (out_count >= 1));
    
    std::string mux_name = "ctrl_all_to_all" ;

    if(in_count == 1){
        mux_name = "ctrl_demux";
    }
    else if(out_count == 1){
        mux_name = "ctrl_mux";
    }

    std::vector<struct wire_config> selector_wires, dispatcher_wires;
    int select_data_width = ceil(log2(in_count));
    int dispatch_data_width = ceil(log2(out_count));
    struct axis_config selector_axis = {0, 0, select_data_width};
    struct axis_config dispatcher_axis = {0, 0, dispatch_data_width};

    if(in_count == 1 && out_count == 1){
        // in == out, direct connect
        struct bits_config bitsc = {event_wire_counts};
        auto barrier_name = assign_name("barriers");
        struct wire_config barrier_wire = {BIT, barrier_name, "barrier bits", false, "", true, {.bit = bitsc}};
        emitwire(file, barrier_wire);

        auto in_var_names = std::vector<std::string>();
        auto out_var_names = std::vector<std::string>();
        for(int i = 0; i < event_wire_counts ; i++){
            auto inwire = ctrl_ins[*cur_funcop][invs[0]].event_wires[i];
            auto outwire = ctrl_outs[*cur_funcop][outvs[0]].event_wires[i];
            std::list<struct module_port_config> ports;
            ports.push_back({AXIS, {inwire.name}, "src wire", "s_in", inwire.axis});
            ports.push_back({AXIS, {outwire.name}, "dst wire", "m_out", outwire.axis});
            std::list<struct module_param_config> params;
            params.push_back({"DATA_WIDTH", inwire.axis.data_width});
            params.push_back({"KEEP_ENABLE ", inwire.axis.if_keep});
            ports.push_back({BIT, {barrier_wire.name + "[" + std::to_string(i) +"]"}, "barrier bits", "barrier", {.bit =bitsc}});
            emitModuleCall(file, "barrier_queue", assign_name("barrier_queue"), ports, params);

            struct bits_config single_bit = {1};
            auto inc_string = "(" + inwire.name + "_tready && " + inwire.name + "_tvalid" + ")";
            auto inc_name = assign_name("inc");
            struct wire_config inc_wire = {BIT, inc_name, "increase counter", true, inc_string, true, {.bit =single_bit}};
            emitwire(file, inc_wire);
            in_var_names.push_back(inc_name);

            auto dec_string = "(" + outwire.name + "_tready && " + outwire.name + "_tvalid" + ")";
            auto dec_name = assign_name("dec");
            struct wire_config dec_wire = {BIT, dec_name, "decrease counter", true, dec_string, true, {.bit =single_bit}};
            emitonewire(file, dec_wire);
            out_var_names.push_back(dec_name);
        }
        std::reverse(in_var_names.begin(), in_var_names.end());
        std::reverse(out_var_names.begin(), out_var_names.end());
        std::list<struct module_port_config> barrier_ports;
        barrier_ports.push_back({BIT, in_var_names, "inc wire", "s_inc", {.bit =bitsc}});
        barrier_ports.push_back({BIT, out_var_names, "desc wire", "s_dec", {.bit =bitsc}});
        barrier_ports.push_back({BIT, {barrier_name}, "barriers", "ctrl_barrier", {.bit =bitsc}});
        std::list<struct module_param_config> params;
        params.push_back({"PORT_COUNT", event_wire_counts});

        emitModuleCall(file, "ctrl_barrier", assign_name("ctrl_barrier"), barrier_ports, params);

        return;
    }
    
    for(int i = 0; i < event_wire_counts ; i++){
        std::list<struct module_port_config> ports;
        auto in_var_names = std::vector<std::string>();
        auto out_var_names = std::vector<std::string>();
        // In out axis should be same
        auto axis = ctrl_ins[*cur_funcop][invs[0]].event_wires[i].axis;

        for(auto in : invs){
            auto in_id = ctrl_ins[*cur_funcop][in].replicate_index;
            auto event_wire = ctrl_ins[*cur_funcop][in].event_wires[i];
            in_var_names.push_back(event_wire.name);
        }
        ports.push_back({AXIS, in_var_names, "(de)mux in", "s_val_axis", axis});

        for(auto out : outvs){
            auto out_id = ctrl_outs[*cur_funcop][out].replicate_index;
            auto event_wire = ctrl_outs[*cur_funcop][out].event_wires[i];
            out_var_names.push_back(event_wire.name);
        }
        ports.push_back({AXIS, out_var_names, "(des)mux out", "m_val_axis", axis});

        if(mux_name == "ctrl_mux" || mux_name == "ctrl_all_to_all"){
            struct wire_config selector_wire = {AXIS, assign_name(mux_name + "_select"), "selector wire", false, "", true, selector_axis};
            selector_wires.push_back(selector_wire);
            ports.push_back({AXIS, {selector_wire.name}, "selector wire", "s_selector", selector_axis});
        }
        if(mux_name == "ctrl_demux" || mux_name == "ctrl_all_to_all"){
            struct wire_config dispatcher_wire = {AXIS, assign_name(mux_name + "_dispacth"), "dispatcher wire", false, "", true, dispatcher_axis};
            dispatcher_wires.push_back(dispatcher_wire);
            ports.push_back({AXIS, {dispatcher_wire.name}, "dispatcher wire", "s_dispatcher", dispatcher_axis});
        }
    
        std::list<struct module_param_config> params;
        if(mux_name == "ctrl_mux" || mux_name == "ctrl_all_to_all")
            params.push_back({"S_COUNT  ", in_count});
        if(mux_name == "ctrl_demux" || mux_name == "ctrl_all_to_all")
            params.push_back({"D_COUNT  ", out_count});
        params.push_back({"DATA_WIDTH", axis.data_width});
        params.push_back({"KEEP_ENABLE ", axis.if_keep});

        emitModuleCall(file, mux_name, assign_name(mux_name), ports, params);
    }

    std::vector<std::string> selector_var_names, dispatcher_var_names;
    for(auto &wire : selector_wires){
        emitwire(file, wire);
        selector_var_names.push_back(wire.name);
    }
    for(auto &wire : dispatcher_wires){
        emitwire(file, wire);
        dispatcher_var_names.push_back(wire.name);
    }
    // emit selector
    std::list<struct module_port_config> selector_ports, dispatcher_ports;
    if(mux_name == "ctrl_mux" || mux_name == "ctrl_all_to_all"){
        std::vector<std::string> selector_inc_names;
        struct axis_config axis = {0, 0, 1};
        for(auto in : invs){
            auto event_wire = ctrl_ins[*cur_funcop][in].event_wires[0];
            auto inc_string = "(" + event_wire.name + "_tready && " + event_wire.name + "_tvalid" + ")";
            auto inc_name = assign_name("inc");
            struct wire_config inc_wire = {AXIS, inc_name, "increase from all port's first arg", true, inc_string, true, axis};
            emitwire(file, inc_wire);
            selector_inc_names.push_back(inc_name);
        }
        selector_ports.push_back({AXIS, selector_inc_names, "increase from all port's first arg", "s_inc", axis});
        selector_ports.push_back({AXIS, selector_var_names, "selector wire", "m_selector", selector_axis});
        std::list<struct module_param_config> params;
        params.push_back({"S_COUNT", in_count});
        params.push_back({"SELECT_WIDTH", select_data_width});
        params.push_back({"REPLICATED_OUT_NUM", (int)selector_wires.size()});
        emitModuleCall(file, "ctrl_selector", "ctrl_selector", selector_ports, params);
    }
    if(mux_name == "ctrl_demux" || mux_name == "ctrl_all_to_all"){
        dispatcher_ports.push_back({AXIS, dispatcher_var_names, "dispatcher wire", "m_dispatcher", dispatcher_axis});
        std::list<struct module_param_config> params;
        params.push_back({"D_COUNT", out_count});
        params.push_back({"DISPATCH_WIDTH", dispatch_data_width});
        params.push_back({"REPLICATED_OUT_NUM", (int)dispatcher_wires.size()});
        emitModuleCall(file, "ctrl_dispatcher", "ctrl_dispatcher", dispatcher_ports, params);
    }

}

void EmitFPGAPass::emitController(ep2::FuncOp funcOp) {
    cur_funcop = &funcOp;
    auto handler_name = funcOp.getName().str();
    std::ofstream file(handler_name + ".sv");
    file << "module " << handler_name << "#()\n";
    emitControllerInOut(file, funcOp);
    
    funcOp->walk([&](ConnectOp connectop) {
        emitControllerMux(file, connectop);
    }
    );
    
    file << "\nendmodule\n";
  
//   // from m replicas to n replicas
//   std::vector<struct inout_config> inout_wires;
//   std::list<struct wire_config> mux_in_wires;
//   std::list<struct wire_config> demux_out_wires;
//   // TODO: should have a analysis struct:
//   // std::unordered_map<ep2::FuncOp, std::list<ep2::FuncOp>>
//   // controller_srcs; std::unordered_map<ep2::FuncOp,
//   // std::list<ep2::FuncOp>> controller_dsts;

//   // vvv
//   int src_count = 2;
//   int dst_count = 2;
//   mlir::Value tmped_event;
//   for (auto &temp : handlerInOutAnalysis->handler_returnop_list) {
//     if (temp.second.size() != 0) {
//       tmped_event = temp.second[0];
//     }
//   }
//   // ^^^

//   auto event_type = tmped_event.getType();

//   assert(isa<ep2::StructType>(event_type));
//   auto return_event_struct = cast<ep2::StructType, Type>(event_type);
//   // TODO: This need special care -- event size contains buf/context and itis
//   // not simply a struct..
//   int eventsize = 233;
//   bool if_stream = true; // event is a stream
//   struct axis_config event_axis = {1, 1, eventsize};

//   std::string portname, debuginfo;

//   for (int src_id = 0; src_id < src_count; src_id++) {
//     auto name = assign_name("inport");
//     debuginfo = "input event from src port " + std::to_string(src_id);
//     portname = name + "_" + std::to_string(src_id);
//     struct axis_config wire = {if_stream, if_stream, eventsize};
//     struct inout_config out_if = {IN, AXIS, portname, debuginfo, wire};
//     inout_wires.push_back(out_if);
//     // Collect input port wire information for further wiring
//     mux_in_wires.push_back(
//         {AXIS, portname, debuginfo + "wire", false, "", true, wire});
//   }

//   for (int dst_id = 0; dst_id < dst_count; dst_id++) {
//     auto name = assign_name("outport");
//     debuginfo = "output event for dst port " + std::to_string(dst_id);
//     portname = name + "_" + std::to_string(dst_id);
//     struct axis_config wire = {if_stream, if_stream, eventsize};
//     struct inout_config out_if = {OUT, AXIS, portname, debuginfo, wire};
//     inout_wires.push_back(out_if);
//     // Collect output port wire information for further wiring
//     demux_out_wires.push_back(
//         {AXIS, portname, debuginfo + "wire", false, "", true, wire});
//   }

//   emitModuleParameter(file, inout_wires);

//   bool if_enable_mux = (src_count > 1);
//   bool if_enable_demux = (dst_count > 1);
//   std::vector<std::string> fifo_invar_names;
//   if (if_enable_mux) {
//     //  emit mux's output wire defination
//     struct module_port_config mux_in_port, mux_out_port;
//     struct wire_config mux_out_wire;
//     auto mux_out_name = assign_name("mux_out");
//     mux_out_wire.name = mux_out_name;
//     mux_out_wire = {AXIS,  mux_out_name, "Mux output wire",
//                     false, "", true,         event_axis};
//     mux_out_port = {AXIS, {mux_out_name}, "mux output", "m_axis", event_axis};

//     emitwire(file, mux_out_wire);

//     // emit mux call
//     std::list<struct module_port_config> ports;
//     std::vector<std::string> var_names;
//     for (auto &w : mux_in_wires) {
//       var_names.push_back(w.name);
//     }
//     mux_in_port = {AXIS, var_names, "mux input", "s_axis", event_axis};

//     ports.push_back(mux_in_port);
//     ports.push_back(mux_out_port);
//     // MUX parameters
//     std::list<struct module_param_config> params;
//     params.push_back({"S_COUNT  ", src_count});
//     params.push_back({"DATA_WIDTH", eventsize});
//     params.push_back({"KEEP_ENABLE ", 1});
//     params.push_back({"USER_ENABLE ", 0});

//     emitModuleCall(file, "axis_arb_mux ", "axis_arb_mux", ports, params);

//     fifo_invar_names = {mux_out_name};
//   } else {
//     fifo_invar_names = {inout_wires.front().name};
//   }

//   // EMIT FIFO + disptacher
//   struct module_port_config fifo_in_port, fifo_out_port;
//   std::list<struct module_port_config> fifoports;
//   fifo_in_port = {AXIS, fifo_invar_names, "queue input", "s_axis", event_axis};

//   std::vector<std::string> fifo_out_var_names;
//   for (auto &w : demux_out_wires) {
//     fifo_out_var_names.push_back(w.name);
//   }
//   fifo_out_port = {AXIS, fifo_out_var_names, "queue output", "m_axis",
//                    event_axis};
//   fifoports.push_back(fifo_in_port);
//   fifoports.push_back(fifo_out_port);

//   // MUX parameters
//   std::list<struct module_param_config> params;
//   params.push_back({"D_COUNT  ", dst_count});
//   params.push_back({"DATA_WIDTH", eventsize});
//   params.push_back({"KEEP_ENABLE ", 1});
//   params.push_back({"USER_ENABLE ", 0});
//   params.push_back({"QUEUE_TYPE ", 0});
//   params.push_back({"QUEUE_SIZE ", 128});

//   emitModuleCall(file, "dispatch_queue", "queue", fifoports, params);
//   file << "\nendmodule\n";
}
}
} // namespace mlir