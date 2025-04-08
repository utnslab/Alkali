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
    %9 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 0>}> : () -> !ep2.port<true, false>
    %10 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 0>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%9, %10) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %11 = "ep2.nop"() : () -> none
    %12 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 1>}> : () -> !ep2.port<true, false>
    %13 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 1>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%12, %13) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %14 = "ep2.nop"() : () -> none
    %15 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 2>}> : () -> !ep2.port<true, false>
    %16 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 2>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%15, %16) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %17 = "ep2.nop"() : () -> none
    %18 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 3>}> : () -> !ep2.port<true, false>
    %19 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 3>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%18, %19) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %20 = "ep2.nop"() : () -> none
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_OoO_DETECT2() attributes {event = "OoO_DETECT2", type = "controller"} {
    %9 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 0>}> : () -> !ep2.port<true, false>
    %10 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT2" : "OoO_detection2", 0>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%9, %10) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %11 = "ep2.nop"() : () -> none
    %12 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 1>}> : () -> !ep2.port<true, false>
    %13 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT2" : "OoO_detection2", 1>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%12, %13) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %14 = "ep2.nop"() : () -> none
    %15 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 2>}> : () -> !ep2.port<true, false>
    %16 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT2" : "OoO_detection2", 2>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%15, %16) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %17 = "ep2.nop"() : () -> none
    %18 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT1" : "OoO_detection1", 3>}> : () -> !ep2.port<true, false>
    %19 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT2" : "OoO_detection2", 3>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%18, %19) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [128]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %20 = "ep2.nop"() : () -> none
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["cu1", "cu4", "cu13", "cu17"], type = "handler"} {
    %9 = "ep2.init"() : () -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %10 = "ep2.init"() : () -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %11 = "ep2.init"() : () -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %12 = "ep2.init"() : () -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16>
    %13 = "ep2.init"() : () -> !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>
    %14 = "ep2.init"() : () -> !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>
    %15 = "ep2.init"() : () -> !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>
    %16 = "ep2.init"() : () -> !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>
    %17 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %18 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %19 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %20 = "ep2.init"() : () -> i32
    %21 = ep2.struct_access %18[6] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %22 = ep2.struct_access %18[7] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %23 = "ep2.add"(%21, %22) : (i32, i32) -> i32
    %24 = ep2.struct_access %19[0] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %25 = "ep2.add"(%23, %24) : (i32, i16) -> i32
    %26 = ep2.struct_access %19[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %27 = "ep2.add"(%25, %26) : (i32, i16) -> i32
    %28 = "ep2.init"() : () -> i32
    %29 = ep2.struct_access %18[6] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %30 = "ep2.global_import"() <{name = "firewall_ip_table"}> : () -> !ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
    %31 = "ep2.lookup"(%30, %29) : (!ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>, i32) -> !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>
    %32 = "ep2.init"() : () -> i16
    %33 = ep2.struct_access %19[0] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %34 = "ep2.global_import"() <{name = "firewall_tcpport_table"}> : () -> !ep2.table<i16, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
    %35 = "ep2.lookup"(%34, %33) : (!ep2.table<i16, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>, i16) -> !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>
    %36 = ep2.struct_access %35[2] : <"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %37 = "ep2.global_import"() <{name = "priority_table"}> : () -> !ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %38 = "ep2.lookup"(%37, %36) : (!ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>
    %39 = ep2.struct_access %31[2] : <"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %40 = "ep2.global_import"() <{name = "priority_table"}> : () -> !ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %41 = "ep2.lookup"(%40, %39) : (!ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>
    %42 = "ep2.init"() : () -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32>
    %43 = ep2.struct_access %42[1] : <"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %44 = ep2.struct_access %35[4] : <"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %45 = ep2.struct_access %31[4] : <"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %46 = "ep2.add"(%44, %45) : (i32, i32) -> i32
    %47 = "ep2.struct_update"(%42, %46) <{index = 1 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32>
    %48 = ep2.struct_access %47[2] : <"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %49 = ep2.struct_access %41[0] : <"priority_entries_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %50 = ep2.struct_access %38[0] : <"priority_entries_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %51 = "ep2.add"(%49, %50) : (i32, i32) -> i32
    %52 = "ep2.struct_update"(%47, %51) <{index = 2 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32>
    "ep2.emit"(%arg1, %52) : (!ep2.buf, !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32>) -> ()
    %53 = "ep2.nop"() : () -> none
    %54 = "ep2.constant"() <{value = "OoO_detection1"}> : () -> !ep2.atom
    %55 = "ep2.init"(%54, %arg0, %arg1, %27, %29, %33) : (!ep2.atom, !ep2.context, !ep2.buf, i32, i32, i16) -> !ep2.struct<"OoO_DETECT1" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, i32, i32, i16>
    ep2.return %55 : !ep2.struct<"OoO_DETECT1" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, i32, i32, i16>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_OoO_DETECT1_OoO_detection1(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: i32, %arg3: i32, %arg4: i16) attributes {atom = "OoO_detection1", event = "OoO_DETECT1", instances = ["cu2", "cu5", "cu15", "cu18"], type = "handler"} {
    %9 = "ep2.init"() : () -> !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %10 = "ep2.global_import"() <{name = "flow_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %11 = "ep2.lookup"(%10, %arg2) : (!ep2.table<i32, !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %12 = ep2.struct_access %11[0] : <"flow_tracker_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %13 = ep2.struct_access %11[0] : <"flow_tracker_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %14 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %15 = "ep2.bitcast"(%14) : (i64) -> i32
    %16 = "ep2.add"(%13, %15) : (i32, i32) -> i32
    %17 = "ep2.struct_update"(%11, %16) <{index = 0 : i64}> : (!ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %18 = "ep2.global_import"() <{name = "flow_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
    "ep2.update"(%18, %arg2, %17) : (!ep2.table<i32, !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32, !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>) -> ()
    %19 = "ep2.nop"() : () -> none
    %20 = "ep2.init"() : () -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %21 = "ep2.global_import"() <{name = "tcp_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %22 = "ep2.lookup"(%21, %arg4) : (!ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i16) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %23 = ep2.struct_access %22[0] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %24 = ep2.struct_access %22[0] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %25 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %26 = "ep2.bitcast"(%25) : (i64) -> i32
    %27 = "ep2.add"(%24, %26) : (i32, i32) -> i32
    %28 = "ep2.struct_update"(%22, %27) <{index = 0 : i64}> : (!ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %29 = "ep2.global_import"() <{name = "tcp_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
    "ep2.update"(%29, %arg4, %28) : (!ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i16, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32>) -> ()
    %30 = "ep2.nop"() : () -> none
    %31 = "ep2.init"() : () -> !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %32 = "ep2.global_import"() <{name = "ip_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %33 = "ep2.lookup"(%32, %arg3) : (!ep2.table<i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %34 = ep2.struct_access %33[0] : <"ip_tracker_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %35 = ep2.struct_access %33[0] : <"ip_tracker_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %36 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %37 = "ep2.bitcast"(%36) : (i64) -> i32
    %38 = "ep2.add"(%35, %37) : (i32, i32) -> i32
    %39 = "ep2.struct_update"(%33, %38) <{index = 0 : i64}> : (!ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %40 = "ep2.global_import"() <{name = "ip_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
    "ep2.update"(%40, %arg3, %39) : (!ep2.table<i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32>) -> ()
    %41 = "ep2.nop"() : () -> none
    %42 = "ep2.init"() : () -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %43 = "ep2.global_import"() <{name = "err_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %44 = "ep2.lookup"(%43, %arg2) : (!ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %45 = ep2.struct_access %44[0] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %46 = ep2.struct_access %44[0] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %47 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %48 = "ep2.bitcast"(%47) : (i64) -> i32
    %49 = "ep2.add"(%46, %48) : (i32, i32) -> i32
    %50 = "ep2.struct_update"(%44, %49) <{index = 0 : i64}> : (!ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>
    %51 = "ep2.global_import"() <{name = "err_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
    "ep2.update"(%51, %arg2, %50) : (!ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32>) -> ()
    %52 = "ep2.nop"() : () -> none
    %53 = "ep2.constant"() <{value = "OoO_detection2"}> : () -> !ep2.atom
    %54 = "ep2.init"(%53, %arg0, %arg1, %arg2) : (!ep2.atom, !ep2.context, !ep2.buf, i32) -> !ep2.struct<"OoO_DETECT2" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, i32>
    ep2.return %54 : !ep2.struct<"OoO_DETECT2" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, i32>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_OoO_DETECT2_OoO_detection2(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: i32) attributes {atom = "OoO_detection2", event = "OoO_DETECT2", instances = ["cu3", "cu6", "cu16", "cu19"], type = "handler"} {
    %9 = "ep2.init"() : () -> i32
    %10 = "ep2.constant"() <{value = 134744072 : i64}> : () -> i64
    %11 = "ep2.bitcast"(%10) : (i64) -> i32
    %12 = "ep2.init"() : () -> i32
    %13 = "ep2.constant"() <{value = 134744071 : i64}> : () -> i64
    %14 = "ep2.bitcast"(%13) : (i64) -> i32
    %15 = "ep2.init"() : () -> i16
    %16 = "ep2.constant"() <{value = 50 : i64}> : () -> i64
    %17 = "ep2.bitcast"(%16) : (i64) -> i16
    %18 = "ep2.init"() : () -> i16
    %19 = "ep2.constant"() <{value = 60 : i64}> : () -> i64
    %20 = "ep2.bitcast"(%19) : (i64) -> i16
    %21 = "ep2.init"() : () -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %22 = "ep2.global_import"() <{name = "lb_table"}> : () -> !ep2.table<i32, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, 64>
    %23 = "ep2.lookup"(%22, %arg2) : (!ep2.table<i32, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, 64>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %24 = ep2.struct_access %23[6] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i32
    %25 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %26 = "ep2.bitcast"(%25) : (i64) -> i32
    %27 = "ep2.struct_update"(%23, %26) <{index = 6 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %28 = ep2.struct_access %27[0] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i64
    %29 = ep2.struct_access %27[0] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i64
    %30 = "ep2.add"(%29, %arg2) : (i64, i32) -> i64
    %31 = "ep2.struct_update"(%27, %30) <{index = 0 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %32 = ep2.struct_access %31[1] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i64
    %33 = ep2.struct_access %31[1] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i64
    %34 = "ep2.add"(%33, %arg2) : (i64, i32) -> i64
    %35 = "ep2.struct_update"(%31, %34) <{index = 1 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %36 = ep2.struct_access %35[2] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i32
    %37 = "ep2.add"(%11, %arg2) : (i32, i32) -> i32
    %38 = "ep2.struct_update"(%35, %37) <{index = 2 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %39 = ep2.struct_access %38[3] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i32
    %40 = "ep2.add"(%14, %arg2) : (i32, i32) -> i32
    %41 = "ep2.struct_update"(%38, %40) <{index = 3 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %42 = ep2.struct_access %41[4] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i16
    %43 = "ep2.add"(%17, %arg2) : (i16, i32) -> i16
    %44 = "ep2.struct_update"(%41, %43) <{index = 4 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %45 = ep2.struct_access %44[5] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i16
    %46 = "ep2.add"(%20, %arg2) : (i16, i32) -> i16
    %47 = "ep2.struct_update"(%44, %46) <{index = 5 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>
    %48 = "ep2.global_import"() <{name = "lb_table"}> : () -> !ep2.table<i32, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, 64>
    "ep2.update"(%48, %arg2, %47) : (!ep2.table<i32, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>, 64>, i32, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32>) -> ()
    %49 = "ep2.nop"() : () -> none
    %50 = "ep2.init"() : () -> i32
    %51 = ep2.struct_access %47[2] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i32
    %52 = ep2.struct_access %47[3] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i32
    %53 = "ep2.add"(%51, %52) : (i32, i32) -> i32
    %54 = ep2.struct_access %47[4] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i16
    %55 = "ep2.add"(%53, %54) : (i32, i16) -> i32
    %56 = ep2.struct_access %47[5] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32> -> i16
    %57 = "ep2.add"(%55, %56) : (i32, i16) -> i32
    %58 = "ep2.init"() : () -> !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>
    %59 = "ep2.global_import"() <{name = "lb_fwd_table"}> : () -> !ep2.table<i32, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>, 64>
    %60 = "ep2.lookup"(%59, %57) : (!ep2.table<i32, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>, 64>, i32) -> !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>
    "ep2.emit"(%arg1, %60) : (!ep2.buf, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>) -> ()
    %61 = "ep2.nop"() : () -> none
    %62 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %63 = "ep2.init"(%62, %arg0, %arg1) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    ep2.return %63 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}
