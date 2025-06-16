module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  %0 = "ep2.global"() <{name = "firewall_ip_table"}> : () -> !ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
  %1 = "ep2.global"() <{name = "firewall_tcpport_table"}> : () -> !ep2.table<i32, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
  %2 = "ep2.global"() <{name = "priority_table"}> : () -> !ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>
  %3 = "ep2.global"() <{name = "err_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
  %4 = "ep2.global"() <{name = "tcp_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
  %5 = "ep2.global"() <{name = "lb_table"}> : () -> !ep2.table<i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, 64>
  %6 = "ep2.global"() <{name = "flow_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
  %7 = "ep2.global"() <{name = "ip_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
  %8 = "ep2.global"() <{name = "lb_fwd_table"}> : () -> !ep2.table<i32, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>, 64>
  ep2.func @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_DMA_WRITE_REQ_dma_write(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.buf) attributes {atom = "dma_write", event = "DMA_WRITE_REQ", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_NET_RECV_process_packet_2Cut_source(%arg0: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", generationIndex = 2 : i32, instances = ["cu0", "cu1", "cu2"], type = "handler"} {
    %9 = "ep2.global_import"() <{name = "firewall_tcpport_table"}> : () -> !ep2.table<i32, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
    %10 = "ep2.global_import"() <{name = "priority_table"}> : () -> !ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %11, %output = "ep2.extract_value"(%arg0) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>)
    %12, %output_0 = "ep2.extract_value"(%11) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>)
    %13, %output_1 = "ep2.extract_value"(%12) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>)
    %14 = ep2.struct_access %output_0[6] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %15 = ep2.struct_access %output_0[7] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %16 = "ep2.add"(%14, %15) : (i32, i32) -> i32
    %17 = ep2.struct_access %output_1[0] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %18 = "ep2.add"(%16, %17) : (i32, i16) -> i32
    %19 = ep2.struct_access %output_1[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %20 = "ep2.add"(%18, %19) : (i32, i16) -> i32
    %21 = "ep2.lookup"(%9, %20) : (!ep2.table<i32, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>, i32) -> !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>
    %22 = "ep2.lookup"(%10, %20) : (!ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>
    %23 = ep2.struct_access %21[4] : <"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %24 = ep2.struct_access %22[0] : <"priority_entries_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %25 = "ep2.add"(%24, %24) : (i32, i32) -> i32
    %26 = ep2.struct_access %21[3] : <"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %27 = "ep2.init"(%output_1, %14, %20, %25, %23, %26, %13, %17) : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i32, i32, i32, i32, i32, !ep2.buf, i16) -> !ep2.struct<"__handler_NET_RECV_process_packet_sink" : isEvent = true, elementTypes = !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i32, i32, i32, i32, i32, !ep2.buf, i16>
    ep2.return %27 : !ep2.struct<"__handler_NET_RECV_process_packet_sink" : isEvent = true, elementTypes = !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i32, i32, i32, i32, i32, !ep2.buf, i16>
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_NET_RECV_process_packet_2Cut_sink_2Cut_source_2Cut_source_2Cut_source(%arg0: !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: !ep2.buf, %arg7: i16) attributes {generationIndex = 24 : i32, instances = ["cu11", "cu12"], type = "handler"} {
    %9 = "ep2.global_import"() <{name = "firewall_ip_table"}> : () -> !ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
    %10 = "ep2.lookup"(%9, %arg2) : (!ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>, i32) -> !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>
    %11 = ep2.struct_access %10[4] : <"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %12 = "ep2.add"(%arg4, %11) : (i32, i32) -> i32
    %13 = ep2.struct_access %10[3] : <"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %14 = "ep2.add"(%13, %arg5) : (i32, i32) -> i32
    %15 = ep2.struct_access %arg0[2] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i32
    %16 = ep2.struct_access %arg0[7] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %17 = "ep2.init"(%arg1, %arg2, %arg7, %14, %16, %12, %arg6, %arg3, %15) : (i32, i32, i16, i32, i16, i32, !ep2.buf, i32, i32) -> !ep2.struct<"__handler_NET_RECV_process_packet_2Cut_sink_2Cut_source_2Cut_source_sink" : isEvent = true, elementTypes = i32, i32, i16, i32, i16, i32, !ep2.buf, i32, i32>
    ep2.return %17 : !ep2.struct<"__handler_NET_RECV_process_packet_2Cut_sink_2Cut_source_2Cut_source_sink" : isEvent = true, elementTypes = i32, i32, i16, i32, i16, i32, !ep2.buf, i32, i32>
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_NET_RECV_process_packet_2Cut_sink_2Cut_source_2Cut_source_2Cut_sink(%arg0: i32, %arg1: i32, %arg2: i16, %arg3: i32, %arg4: i16, %arg5: i32, %arg6: !ep2.buf, %arg7: i32, %arg8: i32) attributes {generationIndex = 25 : i32, instances = ["cu6", "cu9", "cu10"], type = "handler"} {
    %9 = "ep2.global_import"() <{name = "tcp_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
    %10 = "ep2.lookup"(%9, %arg0) : (!ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>, i32) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %11 = ep2.struct_access %10[0] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %12 = "ep2.sub"(%11, %arg5) : (i32, i32) -> i32
    %13 = "ep2.struct_update"(%10, %12) <{index = 0 : i64}> : (!ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    "ep2.update"(%9, %arg0, %13) : (!ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>, i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>) -> ()
    %14 = "ep2.init"(%arg2, %arg4, %arg8, %13, %arg6, %arg3, %arg7, %arg1, %arg5) : (i16, i16, i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, !ep2.buf, i32, i32, i32, i32) -> !ep2.struct<"__handler_NET_RECV_process_packet_2Cut_sink_2Cut_source_sink" : isEvent = true, elementTypes = i16, i16, i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, !ep2.buf, i32, i32, i32, i32>
    ep2.return %14 : !ep2.struct<"__handler_NET_RECV_process_packet_2Cut_sink_2Cut_source_sink" : isEvent = true, elementTypes = i16, i16, i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, !ep2.buf, i32, i32, i32, i32>
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_NET_RECV_process_packet_2Cut_sink_2Cut_source_2Cut_sink(%arg0: i16, %arg1: i16, %arg2: i32, %arg3: !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, %arg4: !ep2.buf, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {generationIndex = 13 : i32, instances = ["cu3", "cu4", "cu5"], type = "handler"} {
    %9 = "ep2.constant"() <{value = 1 : i32}> : () -> i32
    %10 = "ep2.constant"() <{value = 256 : i32}> : () -> i32
    %11 = "ep2.global_import"() <{name = "err_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
    %12 = "ep2.lookup"(%11, %arg7) : (!ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %13 = ep2.struct_access %12[0] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %14 = "ep2.add"(%13, %9) : (i32, i32) -> i32
    %15 = "ep2.sub"(%14, %arg8) : (i32, i32) -> i32
    %16 = "ep2.struct_update"(%12, %15) <{index = 0 : i64}> : (!ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %17 = ep2.struct_access %16[2] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %18 = "ep2.add"(%17, %10) : (i32, i32) -> i32
    %19 = "ep2.add"(%18, %arg2) : (i32, i32) -> i32
    %20 = "ep2.add"(%19, %arg1) : (i32, i16) -> i32
    %21 = "ep2.struct_update"(%16, %20) <{index = 2 : i64}> : (!ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %22 = ep2.struct_access %21[0] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %23 = ep2.struct_access %21[2] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %24 = "ep2.add"(%22, %23) : (i32, i32) -> i32
    %25 = "ep2.add"(%24, %arg1) : (i32, i16) -> i32
    "ep2.update"(%11, %arg7, %21) : (!ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>, i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>) -> ()
    %26 = ep2.struct_access %arg3[2] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %27 = "ep2.add"(%26, %25) : (i32, i32) -> i32
    %28 = "ep2.struct_update"(%arg3, %27) <{index = 2 : i64}> : (!ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %29 = ep2.struct_access %28[0] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %30 = ep2.struct_access %28[2] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %31 = "ep2.init"(%arg6, %9, %29, %arg0, %arg4, %arg5, %30, %arg8, %arg7) : (i32, i32, i32, i16, !ep2.buf, i32, i32, i32, i32) -> !ep2.struct<"__handler_NET_RECV_process_packet_2Cut_sink_sink" : isEvent = true, elementTypes = i32, i32, i32, i16, !ep2.buf, i32, i32, i32, i32>
    ep2.return %31 : !ep2.struct<"__handler_NET_RECV_process_packet_2Cut_sink_sink" : isEvent = true, elementTypes = i32, i32, i32, i16, !ep2.buf, i32, i32, i32, i32>
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_NET_RECV_process_packet_2Cut_sink_2Cut_sink(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i16, %arg4: !ep2.buf, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {generationIndex = 7 : i32, instances = ["cu13", "cu14", "cu15"], type = "handler"} {
    %9 = "ep2.constant"() <{value = 134744072 : i32}> : () -> i32
    %10 = "ep2.constant"() <{value = 134744071 : i32}> : () -> i32
    %11 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %12 = "ep2.constant"() <{value = 50 : i32}> : () -> i32
    %13 = "ep2.constant"() <{value = 60 : i32}> : () -> i32
    %14 = "ep2.global_import"() <{name = "lb_table"}> : () -> !ep2.table<i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, 64>
    %15 = "ep2.init"() : () -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %16 = "ep2.init"() : () -> !ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %17 = "ep2.init"() : () -> !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>
    %18 = "ep2.struct_update"(%15, %arg7) <{index = 1 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %19 = "ep2.struct_update"(%18, %arg0) <{index = 2 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %20 = "ep2.struct_update"(%19, %arg8) <{index = 0 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %21 = "ep2.struct_update"(%20, %arg5) <{index = 3 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %22 = "ep2.emit_value"(%arg4, %21) : (!ep2.buf, !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>) -> !ep2.buf
    %23 = "ep2.struct_update"(%16, %arg2) <{index = 0 : i64}> : (!ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %24 = "ep2.struct_update"(%23, %arg6) <{index = 1 : i64}> : (!ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %25 = "ep2.emit_value"(%22, %24) : (!ep2.buf, !ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>) -> !ep2.buf
    %26 = "ep2.lookup"(%14, %arg3) : (!ep2.table<i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, 64>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %27 = "ep2.struct_update"(%26, %arg1) <{index = 6 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %28 = ep2.struct_access %27[0] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i64
    %29 = "ep2.add"(%28, %arg8) : (i64, i32) -> i32
    %30 = "ep2.bitcast"(%29) : (i32) -> i64
    %31 = "ep2.struct_update"(%27, %30) <{index = 0 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %32 = ep2.struct_access %31[1] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i64
    %33 = "ep2.add"(%32, %arg8) : (i64, i32) -> i32
    %34 = "ep2.bitcast"(%33) : (i32) -> i64
    %35 = "ep2.struct_update"(%31, %34) <{index = 1 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %36 = "ep2.add"(%9, %arg8) : (i32, i32) -> i32
    %37 = "ep2.struct_update"(%35, %36) <{index = 2 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %38 = "ep2.add"(%10, %arg8) : (i32, i32) -> i32
    %39 = "ep2.struct_update"(%37, %38) <{index = 3 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %40 = "ep2.add"(%12, %arg8) : (i32, i32) -> i32
    %41 = "ep2.bitcast"(%40) : (i32) -> i16
    %42 = "ep2.struct_update"(%39, %41) <{index = 4 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %43 = "ep2.add"(%13, %arg8) : (i32, i32) -> i32
    %44 = "ep2.bitcast"(%43) : (i32) -> i16
    %45 = "ep2.struct_update"(%42, %44) <{index = 5 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %46 = "ep2.add"(%9, %10) : (i32, i32) -> i32
    %47 = "ep2.add"(%46, %12) : (i32, i32) -> i32
    %48 = "ep2.add"(%47, %13) : (i32, i32) -> i32
    %49 = "ep2.add"(%48, %arg8) : (i32, i32) -> i32
    %50 = "ep2.bitcast"(%49) : (i32) -> i64
    %51 = "ep2.struct_update"(%45, %50) <{index = 7 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    "ep2.update"(%14, %arg3, %51) : (!ep2.table<i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, 64>, i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>) -> ()
    %52 = ep2.struct_access %51[2] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i32
    %53 = ep2.struct_access %51[3] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i32
    %54 = "ep2.add"(%52, %53) : (i32, i32) -> i32
    %55 = ep2.struct_access %51[4] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i16
    %56 = "ep2.add"(%54, %55) : (i32, i16) -> i32
    %57 = ep2.struct_access %51[5] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i16
    %58 = "ep2.add"(%56, %57) : (i32, i16) -> i32
    %59 = "ep2.bitcast"(%58) : (i32) -> i64
    %60 = "ep2.struct_update"(%17, %59) <{index = 0 : i64}> : (!ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>, i64) -> !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>
    %61 = "ep2.emit_value"(%25, %60) : (!ep2.buf, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>) -> !ep2.buf
    %62 = "ep2.init"(%11, %61) {context_names = ["", ""]} : (!ep2.atom, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.buf>
    ep2.return %62 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}

