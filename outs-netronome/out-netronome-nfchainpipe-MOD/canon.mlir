module {
  ep2.func private @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_WRITE_REQ_dma_write(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) attributes {atom = "dma_write", event = "DMA_WRITE_REQ", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  %0 = "ep2.global"() <{name = "firewall_ip_table"}> {instances = ["lmem_cu1", "lmem_cu4", "lmem_cu13", "lmem_cu17"]} : () -> !ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
  %1 = "ep2.global"() <{name = "firewall_tcpport_table"}> {instances = ["cls_island1", "cls_island1", "cls_island2", "cls_island2"]} : () -> !ep2.table<i16, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
  %2 = "ep2.global"() <{name = "priority_table"}> {instances = ["cls_island1", "cls_island1", "cls_island2", "cls_island2"]} : () -> !ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>
  %3 = "ep2.global"() <{name = "flow_tracker_table"}> {instances = ["lmem_cu2", "lmem_cu5", "lmem_cu14", "lmem_cu18"]} : () -> !ep2.table<i32, !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
  %4 = "ep2.global"() <{name = "tcp_tracker_table"}> {instances = ["lmem_cu2", "lmem_cu5", "lmem_cu15", "lmem_cu18"]} : () -> !ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
  %5 = "ep2.global"() <{name = "err_tracker_table"}> {instances = ["cls_island1", "cls_island1", "cls_island2", "cls_island2"]} : () -> !ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
  %6 = "ep2.global"() <{name = "ip_tracker_table"}> {instances = ["cls_island1", "cls_island1", "cls_island2", "cls_island2"]} : () -> !ep2.table<i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
  %7 = "ep2.global"() <{name = "lb_table"}> {instances = ["lmem_cu3", "lmem_cu6", "lmem_cu16", "lmem_cu19"]} : () -> !ep2.table<i32, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, 64>
  %8 = "ep2.global"() <{name = "lb_fwd_table"}> {instances = ["cls_island1", "cls_island1", "cls_island2", "cls_island2"]} : () -> !ep2.table<i32, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>, 64>
  ep2.func private @__controller_OoO_DETECT1() attributes {event = "OoO_DETECT1", type = "controller"} {
    %9 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 3>}> : () -> !ep2.port<false, true>
    %10 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 3>}> : () -> !ep2.port<true, false>
    %11 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 2>}> : () -> !ep2.port<false, true>
    %12 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 2>}> : () -> !ep2.port<true, false>
    %13 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 1>}> : () -> !ep2.port<false, true>
    %14 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 1>}> : () -> !ep2.port<true, false>
    %15 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 0>}> : () -> !ep2.port<true, false>
    %16 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 0>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%15, %16) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.connect"(%14, %13) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.connect"(%12, %11) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.connect"(%10, %9) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_OoO_DETECT2() attributes {event = "OoO_DETECT2", type = "controller"} {
    %9 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT2" : "OoO_detection2", 3>}> : () -> !ep2.port<false, true>
    %10 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 3>}> : () -> !ep2.port<true, false>
    %11 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT2" : "OoO_detection2", 2>}> : () -> !ep2.port<false, true>
    %12 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 2>}> : () -> !ep2.port<true, false>
    %13 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT2" : "OoO_detection2", 1>}> : () -> !ep2.port<false, true>
    %14 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 1>}> : () -> !ep2.port<true, false>
    %15 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 0>}> : () -> !ep2.port<true, false>
    %16 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT2" : "OoO_detection2", 0>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%15, %16) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.connect"(%14, %13) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.connect"(%12, %11) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.connect"(%10, %9) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["cu1", "cu4", "cu13", "cu17"], type = "handler"} {
    %9 = "ep2.constant"() <{value = "OoO_detection1"}> : () -> !ep2.atom
    %10 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %11 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %12 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %13 = ep2.struct_access %11[6] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %14 = ep2.struct_access %11[7] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %15 = "ep2.add"(%13, %14) : (i32, i32) -> i32
    %16 = ep2.struct_access %12[0] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %17 = "ep2.add"(%15, %16) : (i32, i16) -> i32
    %18 = ep2.struct_access %12[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %19 = "ep2.add"(%17, %18) : (i32, i16) -> i32
    %20 = "ep2.global_import"() <{name = "firewall_ip_table"}> : () -> !ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
    %21 = "ep2.lookup"(%20, %13) : (!ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>, i32) -> !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>
    %22 = "ep2.global_import"() <{name = "firewall_tcpport_table"}> : () -> !ep2.table<i16, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
    %23 = "ep2.lookup"(%22, %16) : (!ep2.table<i16, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>, i16) -> !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>
    %24 = ep2.struct_access %23[2] : <"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %25 = "ep2.global_import"() <{name = "priority_table"}> : () -> !ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %26 = "ep2.lookup"(%25, %24) : (!ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>
    %27 = ep2.struct_access %21[2] : <"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %28 = "ep2.lookup"(%25, %27) : (!ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>
    %29 = "ep2.init"() : () -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32>
    %30 = ep2.struct_access %23[4] : <"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %31 = ep2.struct_access %21[4] : <"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %32 = "ep2.add"(%30, %31) : (i32, i32) -> i32
    %33 = "ep2.struct_update"(%29, %32) <{index = 1 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32>
    %34 = ep2.struct_access %28[0] : <"priority_entries_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %35 = ep2.struct_access %26[0] : <"priority_entries_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %36 = "ep2.add"(%34, %35) : (i32, i32) -> i32
    %37 = "ep2.struct_update"(%33, %36) <{index = 2 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32>
    "ep2.emit"(%arg1, %37) : (!ep2.buf, !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32>) -> ()
    %38 = "ep2.init"(%9, %arg0, %arg1, %19, %13, %16) : (!ep2.atom, !ep2.context, !ep2.buf, i32, i32, i16) -> !ep2.struct<"OoO_DETECT1" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf, i32, i32, i16>
    ep2.return %38 : !ep2.struct<"OoO_DETECT1" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf, i32, i32, i16>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_OoO_DETECT1_OoO_detection1(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: i32, %arg3: i32, %arg4: i16) attributes {atom = "OoO_detection1", event = "OoO_DETECT1", instances = ["cu2", "cu5", "cu15", "cu18"], type = "handler"} {
    %9 = "ep2.constant"() <{value = 1 : i32}> : () -> i32
    %10 = "ep2.constant"() <{value = "OoO_detection2"}> : () -> !ep2.atom
    %11 = "ep2.global_import"() <{name = "flow_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %12 = "ep2.lookup"(%11, %arg2) : (!ep2.table<i32, !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %13 = ep2.struct_access %12[0] : <"flow_tracker_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %14 = "ep2.add"(%13, %9) : (i32, i32) -> i32
    %15 = "ep2.struct_update"(%12, %14) <{index = 0 : i64}> : (!ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>
    "ep2.update"(%11, %arg2, %15) : (!ep2.table<i32, !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32, !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>) -> ()
    %16 = "ep2.global_import"() <{name = "tcp_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %17 = "ep2.bitcast"(%arg4) : (i16) -> i32
    %18 = "ep2.lookup"(%16, %17) : (!ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %19 = ep2.struct_access %18[0] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %20 = "ep2.add"(%19, %9) : (i32, i32) -> i32
    %21 = "ep2.struct_update"(%18, %20) <{index = 0 : i64}> : (!ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>
    "ep2.update"(%16, %17, %21) : (!ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>) -> ()
    %22 = "ep2.global_import"() <{name = "ip_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %23 = "ep2.lookup"(%22, %arg3) : (!ep2.table<i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %24 = ep2.struct_access %23[0] : <"ip_tracker_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %25 = "ep2.add"(%24, %9) : (i32, i32) -> i32
    %26 = "ep2.struct_update"(%23, %25) <{index = 0 : i64}> : (!ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>
    "ep2.update"(%22, %arg3, %26) : (!ep2.table<i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>) -> ()
    %27 = "ep2.global_import"() <{name = "err_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %28 = "ep2.lookup"(%27, %arg2) : (!ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %29 = ep2.struct_access %28[0] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %30 = "ep2.add"(%29, %9) : (i32, i32) -> i32
    %31 = "ep2.struct_update"(%28, %30) <{index = 0 : i64}> : (!ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>
    "ep2.update"(%27, %arg2, %31) : (!ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>) -> ()
    %32 = "ep2.init"(%10, %arg0, %arg1, %arg2) : (!ep2.atom, !ep2.context, !ep2.buf, i32) -> !ep2.struct<"OoO_DETECT2" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf, i32>
    ep2.return %32 : !ep2.struct<"OoO_DETECT2" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf, i32>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_OoO_DETECT2_OoO_detection2(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: i32) attributes {atom = "OoO_detection2", event = "OoO_DETECT2", instances = ["cu3", "cu6", "cu16", "cu19"], type = "handler"} {
    %9 = "ep2.constant"() <{value = 134744072 : i32}> : () -> i32
    %10 = "ep2.constant"() <{value = 134744071 : i32}> : () -> i32
    %11 = "ep2.constant"() <{value = 50 : i16}> : () -> i16
    %12 = "ep2.constant"() <{value = 60 : i16}> : () -> i16
    %13 = "ep2.constant"() <{value = 1 : i32}> : () -> i32
    %14 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %15 = "ep2.global_import"() <{name = "lb_table"}> : () -> !ep2.table<i32, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, 64>
    %16 = "ep2.lookup"(%15, %arg2) : (!ep2.table<i32, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, 64>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %17 = "ep2.struct_update"(%16, %13) <{index = 6 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %18 = ep2.struct_access %17[0] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i64
    %19 = "ep2.add"(%18, %arg2) : (i64, i32) -> i64
    %20 = "ep2.struct_update"(%17, %19) <{index = 0 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %21 = ep2.struct_access %20[1] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i64
    %22 = "ep2.add"(%21, %arg2) : (i64, i32) -> i64
    %23 = "ep2.struct_update"(%20, %22) <{index = 1 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %24 = "ep2.add"(%9, %arg2) : (i32, i32) -> i32
    %25 = "ep2.struct_update"(%23, %24) <{index = 2 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %26 = "ep2.add"(%10, %arg2) : (i32, i32) -> i32
    %27 = "ep2.struct_update"(%25, %26) <{index = 3 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %28 = "ep2.add"(%11, %arg2) : (i16, i32) -> i16
    %29 = "ep2.struct_update"(%27, %28) <{index = 4 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %30 = "ep2.add"(%12, %arg2) : (i16, i32) -> i16
    %31 = "ep2.struct_update"(%29, %30) <{index = 5 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    "ep2.update"(%15, %arg2, %31) : (!ep2.table<i32, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, 64>, i32, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>) -> ()
    %32 = ep2.struct_access %31[2] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i32
    %33 = ep2.struct_access %31[3] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i32
    %34 = "ep2.add"(%32, %33) : (i32, i32) -> i32
    %35 = ep2.struct_access %31[4] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i16
    %36 = "ep2.add"(%34, %35) : (i32, i16) -> i32
    %37 = ep2.struct_access %31[5] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i16
    %38 = "ep2.add"(%36, %37) : (i32, i16) -> i32
    %39 = "ep2.global_import"() <{name = "lb_fwd_table"}> : () -> !ep2.table<i32, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>, 64>
    %40 = "ep2.lookup"(%39, %38) : (!ep2.table<i32, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>, 64>, i32) -> !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>
    "ep2.emit"(%arg1, %40) : (!ep2.buf, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>) -> ()
    %41 = "ep2.init"(%14, %arg0, %arg1) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    ep2.return %41 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}

