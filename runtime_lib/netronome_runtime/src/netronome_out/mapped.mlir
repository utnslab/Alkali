module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  %0 = "ep2.global"() <{name = "firewall_ip_table"}> {instances = ["cls_cu11", "cls_cu12"]} : () -> !ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
  %1 = "ep2.global"() <{name = "firewall_tcpport_table"}> {instances = ["cls_cu0", "cls_cu1", "cls_cu2"]} : () -> !ep2.table<i32, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
  %2 = "ep2.global"() <{name = "priority_table"}> {instances = ["cls_cu0", "cls_cu1", "cls_cu2"]} : () -> !ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>
  %3 = "ep2.global"() <{name = "err_tracker_table"}> {instances = ["cls_cu3", "cls_cu4", "cls_cu5"]} : () -> !ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
  %4 = "ep2.global"() <{name = "tcp_tracker_table"}> {instances = ["cls_cu6", "cls_cu9", "cls_cu10"]} : () -> !ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
  %5 = "ep2.global"() <{name = "lb_table"}> {instances = ["cls_cu13", "cls_cu14", "cls_cu15"]} : () -> !ep2.table<i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, 64>
  %6 = "ep2.global"() <{name = "ip_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
  %7 = "ep2.global"() <{name = "lb_fwd_table"}> : () -> !ep2.table<i32, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>, 64>
  ep2.func @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_DMA_WRITE_REQ_dma_write(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.buf) attributes {atom = "dma_write", event = "DMA_WRITE_REQ", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", generationIndex = 2 : i32, instances = ["cu0", "cu1", "cu2"], type = "handler"} {
    %8 = "ep2.constant"() <{value = "process_packet_1"}> : () -> !ep2.atom
    %9 = "ep2.global_import"() <{name = "firewall_tcpport_table"}> : () -> !ep2.table<i32, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
    %10 = "ep2.global_import"() <{name = "priority_table"}> : () -> !ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %11 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %12 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %13 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %14 = ep2.struct_access %12[6] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %15 = ep2.struct_access %12[7] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %16 = "ep2.add"(%14, %15) : (i32, i32) -> i32
    %17 = ep2.struct_access %13[0] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %18 = "ep2.add"(%16, %17) : (i32, i16) -> i32
    %19 = ep2.struct_access %13[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %20 = "ep2.add"(%18, %19) : (i32, i16) -> i32
    %21 = "ep2.lookup"(%9, %20) : (!ep2.table<i32, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>, i32) -> !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>
    %22 = "ep2.lookup"(%10, %20) : (!ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>
    %23 = ep2.struct_access %21[4] : <"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %24 = ep2.struct_access %22[0] : <"priority_entries_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %25 = "ep2.add"(%24, %24) : (i32, i32) -> i32
    %26 = ep2.struct_access %21[3] : <"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %27 = "ep2.context_ref"(%arg0) <{name = "context0"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>>
    "ep2.store"(%27, %13) : (!ep2.conref<!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>>, !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>) -> ()
    %28 = "ep2.context_ref"(%arg0) <{name = "context1"}> : (!ep2.context) -> !ep2.conref<i32>
    "ep2.store"(%28, %14) : (!ep2.conref<i32>, i32) -> ()
    %29 = "ep2.context_ref"(%arg0) <{name = "context2"}> : (!ep2.context) -> !ep2.conref<i32>
    "ep2.store"(%29, %20) : (!ep2.conref<i32>, i32) -> ()
    %30 = "ep2.context_ref"(%arg0) <{name = "context3"}> : (!ep2.context) -> !ep2.conref<i32>
    "ep2.store"(%30, %25) : (!ep2.conref<i32>, i32) -> ()
    %31 = "ep2.context_ref"(%arg0) <{name = "context4"}> : (!ep2.context) -> !ep2.conref<i16>
    "ep2.store"(%31, %17) : (!ep2.conref<i16>, i16) -> ()
    %32 = "ep2.init"(%8, %arg0, %23, %26, %arg1) : (!ep2.atom, !ep2.context, i32, i32, !ep2.buf) -> !ep2.struct<"NET_RECV_1" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, i32, i32, !ep2.buf>
    ep2.return %32 : !ep2.struct<"NET_RECV_1" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, i32, i32, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__controller_NET_RECV_1() attributes {event = "NET_RECV_1", type = "controller"} {
    %8 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 0>}> : () -> !ep2.port<true, false>
    %9 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 1>}> : () -> !ep2.port<true, false>
    %10 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 2>}> : () -> !ep2.port<true, false>
    %11 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_1" : "process_packet_1", 0>}> : () -> !ep2.port<false, true>
    %12 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_1" : "process_packet_1", 1>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%8, %9, %10, %11, %12) <{method = "Queue", operandSegmentSizes = array<i32: 3, 2>, parameters = [32]}> : (!ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<false, true>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_NET_RECV_1_process_packet_1(%arg0: !ep2.context, %arg1: i32, %arg2: i32, %arg3: !ep2.buf) attributes {atom = "process_packet_1", event = "NET_RECV_1", generationIndex = 24 : i32, instances = ["cu11", "cu12"], type = "handler"} {
    %8 = "ep2.constant"() <{value = "process_packet_2"}> : () -> !ep2.atom
    %9 = "ep2.context_ref"(%arg0) <{name = "context0"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>>
    %10 = "ep2.context_ref"(%arg0) <{name = "context2"}> : (!ep2.context) -> !ep2.conref<i32>
    %11 = "ep2.load"(%9) : (!ep2.conref<!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>>) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %12 = "ep2.load"(%10) : (!ep2.conref<i32>) -> i32
    %13 = "ep2.global_import"() <{name = "firewall_ip_table"}> : () -> !ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
    %14 = "ep2.lookup"(%13, %12) : (!ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>, i32) -> !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>
    %15 = ep2.struct_access %14[4] : <"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %16 = "ep2.add"(%arg1, %15) : (i32, i32) -> i32
    %17 = ep2.struct_access %14[3] : <"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %18 = "ep2.add"(%17, %arg2) : (i32, i32) -> i32
    %19 = ep2.struct_access %11[2] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i32
    %20 = ep2.struct_access %11[7] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %21 = "ep2.context_ref"(%arg0) <{name = "context5"}> : (!ep2.context) -> !ep2.conref<i32>
    "ep2.store"(%21, %18) : (!ep2.conref<i32>, i32) -> ()
    %22 = "ep2.context_ref"(%arg0) <{name = "context6"}> : (!ep2.context) -> !ep2.conref<i16>
    "ep2.store"(%22, %20) : (!ep2.conref<i16>, i16) -> ()
    %23 = "ep2.context_ref"(%arg0) <{name = "context7"}> : (!ep2.context) -> !ep2.conref<i32>
    "ep2.store"(%23, %16) : (!ep2.conref<i32>, i32) -> ()
    %24 = "ep2.context_ref"(%arg0) <{name = "context8"}> : (!ep2.context) -> !ep2.conref<i32>
    "ep2.store"(%24, %19) : (!ep2.conref<i32>, i32) -> ()
    %25 = "ep2.init"(%8, %arg0, %arg3) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_RECV_2" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    ep2.return %25 : !ep2.struct<"NET_RECV_2" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__controller_NET_RECV_2() attributes {event = "NET_RECV_2", type = "controller"} {
    %8 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_1" : "process_packet_1", 0>}> : () -> !ep2.port<true, false>
    %9 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_1" : "process_packet_1", 1>}> : () -> !ep2.port<true, false>
    %10 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_2" : "process_packet_2", 0>}> : () -> !ep2.port<false, true>
    %11 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_2" : "process_packet_2", 1>}> : () -> !ep2.port<false, true>
    %12 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_2" : "process_packet_2", 2>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%8, %9, %10, %11, %12) <{method = "Queue", operandSegmentSizes = array<i32: 2, 3>, parameters = [32]}> : (!ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<false, true>, !ep2.port<false, true>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_NET_RECV_2_process_packet_2(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet_2", event = "NET_RECV_2", generationIndex = 25 : i32, instances = ["cu6", "cu9", "cu10"], type = "handler"} {
    %8 = "ep2.constant"() <{value = "process_packet_3"}> : () -> !ep2.atom
    %9 = "ep2.context_ref"(%arg0) <{name = "context7"}> : (!ep2.context) -> !ep2.conref<i32>
    %10 = "ep2.context_ref"(%arg0) <{name = "context1"}> : (!ep2.context) -> !ep2.conref<i32>
    %11 = "ep2.load"(%10) : (!ep2.conref<i32>) -> i32
    %12 = "ep2.load"(%9) : (!ep2.conref<i32>) -> i32
    %13 = "ep2.global_import"() <{name = "tcp_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
    %14 = "ep2.lookup"(%13, %11) : (!ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>, i32) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %15 = ep2.struct_access %14[0] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %16 = "ep2.sub"(%15, %12) : (i32, i32) -> i32
    %17 = "ep2.struct_update"(%14, %16) <{index = 0 : i64}> : (!ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    "ep2.update"(%13, %11, %17) : (!ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>, i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>) -> ()
    %18 = "ep2.context_ref"(%arg0) <{name = "context9"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>>
    "ep2.store"(%18, %17) : (!ep2.conref<!ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>>, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>) -> ()
    %19 = "ep2.init"(%8, %arg0, %arg1) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_RECV_3" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    ep2.return %19 : !ep2.struct<"NET_RECV_3" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__controller_NET_RECV_3() attributes {event = "NET_RECV_3", type = "controller"} {
    %8 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_2" : "process_packet_2", 0>}> : () -> !ep2.port<true, false>
    %9 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_2" : "process_packet_2", 1>}> : () -> !ep2.port<true, false>
    %10 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_2" : "process_packet_2", 2>}> : () -> !ep2.port<true, false>
    %11 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_3" : "process_packet_3", 0>}> : () -> !ep2.port<false, true>
    %12 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_3" : "process_packet_3", 1>}> : () -> !ep2.port<false, true>
    %13 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_3" : "process_packet_3", 2>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%8, %9, %10, %11, %12, %13) <{method = "Queue", operandSegmentSizes = array<i32: 3, 3>, parameters = [32]}> : (!ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<false, true>, !ep2.port<false, true>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_NET_RECV_3_process_packet_3(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet_3", event = "NET_RECV_3", generationIndex = 13 : i32, instances = ["cu3", "cu4", "cu5"], type = "handler"} {
    %8 = "ep2.constant"() <{value = "process_packet_4"}> : () -> !ep2.atom
    %9 = "ep2.constant"() <{value = 256 : i32}> : () -> i32
    %10 = "ep2.constant"() <{value = 1 : i32}> : () -> i32
    %11 = "ep2.context_ref"(%arg0) <{name = "context6"}> : (!ep2.context) -> !ep2.conref<i16>
    %12 = "ep2.context_ref"(%arg0) <{name = "context7"}> : (!ep2.context) -> !ep2.conref<i32>
    %13 = "ep2.context_ref"(%arg0) <{name = "context9"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>>
    %14 = "ep2.context_ref"(%arg0) <{name = "context8"}> : (!ep2.context) -> !ep2.conref<i32>
    %15 = "ep2.context_ref"(%arg0) <{name = "context2"}> : (!ep2.context) -> !ep2.conref<i32>
    %16 = "ep2.load"(%15) : (!ep2.conref<i32>) -> i32
    %17 = "ep2.load"(%14) : (!ep2.conref<i32>) -> i32
    %18 = "ep2.load"(%11) : (!ep2.conref<i16>) -> i16
    %19 = "ep2.load"(%12) : (!ep2.conref<i32>) -> i32
    %20 = "ep2.load"(%13) : (!ep2.conref<!ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>>) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %21 = "ep2.global_import"() <{name = "err_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
    %22 = "ep2.lookup"(%21, %16) : (!ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %23 = ep2.struct_access %22[0] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %24 = "ep2.add"(%23, %10) : (i32, i32) -> i32
    %25 = "ep2.sub"(%24, %19) : (i32, i32) -> i32
    %26 = "ep2.struct_update"(%22, %25) <{index = 0 : i64}> : (!ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %27 = ep2.struct_access %26[2] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %28 = "ep2.add"(%27, %9) : (i32, i32) -> i32
    %29 = "ep2.add"(%28, %17) : (i32, i32) -> i32
    %30 = "ep2.add"(%29, %18) : (i32, i16) -> i32
    %31 = "ep2.struct_update"(%26, %30) <{index = 2 : i64}> : (!ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %32 = ep2.struct_access %31[0] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %33 = ep2.struct_access %31[2] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %34 = "ep2.add"(%32, %33) : (i32, i32) -> i32
    %35 = "ep2.add"(%34, %18) : (i32, i16) -> i32
    "ep2.update"(%21, %16, %31) : (!ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>, i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>) -> ()
    %36 = ep2.struct_access %20[2] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %37 = "ep2.add"(%36, %35) : (i32, i32) -> i32
    %38 = "ep2.struct_update"(%20, %37) <{index = 2 : i64}> : (!ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %39 = ep2.struct_access %38[0] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %40 = ep2.struct_access %38[2] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %41 = "ep2.init"(%8, %arg0, %10, %39, %arg1, %40) : (!ep2.atom, !ep2.context, i32, i32, !ep2.buf, i32) -> !ep2.struct<"NET_RECV_4" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, i32, i32, !ep2.buf, i32>
    ep2.return %41 : !ep2.struct<"NET_RECV_4" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, i32, i32, !ep2.buf, i32>
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__controller_NET_RECV_4() attributes {event = "NET_RECV_4", type = "controller"} {
    %8 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_3" : "process_packet_3", 0>}> : () -> !ep2.port<true, false>
    %9 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_3" : "process_packet_3", 1>}> : () -> !ep2.port<true, false>
    %10 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_3" : "process_packet_3", 2>}> : () -> !ep2.port<true, false>
    %11 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_4" : "process_packet_4", 0>}> : () -> !ep2.port<false, true>
    %12 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_4" : "process_packet_4", 1>}> : () -> !ep2.port<false, true>
    %13 = "ep2.constant"() <{value = #ep2.port<"NET_RECV_4" : "process_packet_4", 2>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%8, %9, %10, %11, %12, %13) <{method = "Queue", operandSegmentSizes = array<i32: 3, 3>, parameters = [32]}> : (!ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<false, true>, !ep2.port<false, true>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_NET_RECV_4_process_packet_4(%arg0: !ep2.context, %arg1: i32, %arg2: i32, %arg3: !ep2.buf, %arg4: i32) attributes {atom = "process_packet_4", event = "NET_RECV_4", generationIndex = 7 : i32, instances = ["cu13", "cu14", "cu15"], type = "handler"} {
    %8 = "ep2.constant"() <{value = 60 : i32}> : () -> i32
    %9 = "ep2.constant"() <{value = 50 : i32}> : () -> i32
    %10 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %11 = "ep2.constant"() <{value = 134744071 : i32}> : () -> i32
    %12 = "ep2.constant"() <{value = 134744072 : i32}> : () -> i32
    %13 = "ep2.context_ref"(%arg0) <{name = "context4"}> : (!ep2.context) -> !ep2.conref<i16>
    %14 = "ep2.context_ref"(%arg0) <{name = "context5"}> : (!ep2.context) -> !ep2.conref<i32>
    %15 = "ep2.context_ref"(%arg0) <{name = "context7"}> : (!ep2.context) -> !ep2.conref<i32>
    %16 = "ep2.context_ref"(%arg0) <{name = "context3"}> : (!ep2.context) -> !ep2.conref<i32>
    %17 = "ep2.context_ref"(%arg0) <{name = "context2"}> : (!ep2.context) -> !ep2.conref<i32>
    %18 = "ep2.load"(%14) : (!ep2.conref<i32>) -> i32
    %19 = "ep2.load"(%17) : (!ep2.conref<i32>) -> i32
    %20 = "ep2.load"(%16) : (!ep2.conref<i32>) -> i32
    %21 = "ep2.load"(%13) : (!ep2.conref<i16>) -> i16
    %22 = "ep2.load"(%15) : (!ep2.conref<i32>) -> i32
    %23 = "ep2.global_import"() <{name = "lb_table"}> : () -> !ep2.table<i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, 64>
    %24 = "ep2.init"() : () -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %25 = "ep2.init"() : () -> !ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %26 = "ep2.init"() : () -> !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>
    %27 = "ep2.struct_update"(%24, %22) <{index = 1 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %28 = "ep2.struct_update"(%27, %20) <{index = 2 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %29 = "ep2.struct_update"(%28, %19) <{index = 0 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %30 = "ep2.struct_update"(%29, %18) <{index = 3 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    "ep2.emit"(%arg3, %30) : (!ep2.buf, !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>) -> ()
    %31 = "ep2.struct_update"(%25, %arg2) <{index = 0 : i64}> : (!ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %32 = "ep2.struct_update"(%31, %arg4) <{index = 1 : i64}> : (!ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    "ep2.emit"(%arg3, %32) : (!ep2.buf, !ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>) -> ()
    %33 = "ep2.lookup"(%23, %21) : (!ep2.table<i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, 64>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %34 = "ep2.struct_update"(%33, %arg1) <{index = 6 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %35 = ep2.struct_access %34[0] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i64
    %36 = "ep2.add"(%35, %19) : (i64, i32) -> i32
    %37 = "ep2.bitcast"(%36) : (i32) -> i64
    %38 = "ep2.struct_update"(%34, %37) <{index = 0 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %39 = ep2.struct_access %38[1] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i64
    %40 = "ep2.add"(%39, %19) : (i64, i32) -> i32
    %41 = "ep2.bitcast"(%40) : (i32) -> i64
    %42 = "ep2.struct_update"(%38, %41) <{index = 1 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %43 = "ep2.add"(%12, %19) : (i32, i32) -> i32
    %44 = "ep2.struct_update"(%42, %43) <{index = 2 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %45 = "ep2.add"(%11, %19) : (i32, i32) -> i32
    %46 = "ep2.struct_update"(%44, %45) <{index = 3 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %47 = "ep2.add"(%9, %19) : (i32, i32) -> i32
    %48 = "ep2.bitcast"(%47) : (i32) -> i16
    %49 = "ep2.struct_update"(%46, %48) <{index = 4 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %50 = "ep2.add"(%8, %19) : (i32, i32) -> i32
    %51 = "ep2.bitcast"(%50) : (i32) -> i16
    %52 = "ep2.struct_update"(%49, %51) <{index = 5 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %53 = "ep2.add"(%12, %11) : (i32, i32) -> i32
    %54 = "ep2.add"(%53, %9) : (i32, i32) -> i32
    %55 = "ep2.add"(%54, %8) : (i32, i32) -> i32
    %56 = "ep2.add"(%55, %19) : (i32, i32) -> i32
    %57 = "ep2.bitcast"(%56) : (i32) -> i64
    %58 = "ep2.struct_update"(%52, %57) <{index = 7 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    "ep2.update"(%23, %21, %58) : (!ep2.table<i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, 64>, i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>) -> ()
    %59 = ep2.struct_access %58[2] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i32
    %60 = ep2.struct_access %58[3] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i32
    %61 = "ep2.add"(%59, %60) : (i32, i32) -> i32
    %62 = ep2.struct_access %58[4] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i16
    %63 = "ep2.add"(%61, %62) : (i32, i16) -> i32
    %64 = ep2.struct_access %58[5] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i16
    %65 = "ep2.add"(%63, %64) : (i32, i16) -> i32
    %66 = "ep2.bitcast"(%65) : (i32) -> i64
    %67 = "ep2.struct_update"(%26, %66) <{index = 0 : i64}> : (!ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>, i64) -> !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>
    "ep2.emit"(%arg3, %67) : (!ep2.buf, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>) -> ()
    %68 = "ep2.init"(%10, %arg0, %arg3) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    ep2.return %68 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}

