module {
  ep2.func private @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_WRITE_REQ_dma_write(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) attributes {atom = "dma_write", event = "DMA_WRITE_REQ", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["cu2"], type = "handler"} {
    %0 = "ep2.constant"() <{value = "OoO_detection"}> : () -> !ep2.atom
    %1 = "ep2.constant"() <{value = 14 : i16}> : () -> i16
    %2 = "ep2.init"() : () -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %3 = "ep2.extract_offset"(%arg1) <{offset = 0 : i64}> : (!ep2.buf) -> !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>
    %4 = "ep2.extract_offset"(%arg1) <{offset = 128 : i64}> : (!ep2.buf) -> i16
    %5 = "ep2.extract_offset"(%arg1) <{offset = 208 : i64}> : (!ep2.buf) -> !ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>
    %6 = "ep2.extract_offset"(%arg1) <{offset = 304 : i64}> : (!ep2.buf) -> !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>
    %7 = "ep2.add"(%4, %1) : (i16, i16) -> i16
    %8 = "ep2.bitcast"(%7) : (i16) -> i32
    %9 = "ep2.struct_update"(%2, %8) <{index = 1 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %10 = ep2.struct_access %6[2] : <"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32> -> i32
    %11 = "ep2.struct_update"(%9, %10) <{index = 2 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %12 = ep2.struct_access %6[1] : <"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32> -> i16
    %13 = "ep2.bitcast"(%12) : (i16) -> i32
    %14 = "ep2.struct_update"(%11, %13) <{index = 0 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %15 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.buf>
    "ep2.store"(%15, %arg1) : (!ep2.conref<!ep2.buf>, !ep2.buf) -> ()
    %16 = "ep2.context_ref"(%arg0) <{name = "flow_id"}> : (!ep2.context) -> !ep2.conref<i16>
    "ep2.store"(%16, %12) : (!ep2.conref<i16>, i16) -> ()
    %17 = "ep2.context_ref"(%arg0) <{name = "ip_header_sub_1"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>>
    "ep2.store"(%17, %5) : (!ep2.conref<!ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>>, !ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>) -> ()
    %18 = "ep2.context_ref"(%arg0) <{name = "eth_header_sub_0"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>>
    "ep2.store"(%18, %3) : (!ep2.conref<!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>>, !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>) -> ()
    %19 = "ep2.context_ref"(%arg0) <{name = "tcp_header_sub_0"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>>
    "ep2.store"(%19, %6) : (!ep2.conref<!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>>, !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>) -> ()
    %20 = "ep2.init"(%0, %arg0, %14) : (!ep2.atom, !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>) -> !ep2.struct<"OoO_DETECT" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    ep2.return %20 : !ep2.struct<"OoO_DETECT" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_ACK_GEN() attributes {event = "ACK_GEN", type = "controller"} {
    %0 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 0>}> : () -> !ep2.port<true, false>
    %1 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 1>}> : () -> !ep2.port<true, false>
    %2 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 2>}> : () -> !ep2.port<true, false>
    %3 = "ep2.constant"() <{value = #ep2.port<"ACK_GEN" : "ack_gen", 0>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%0, %1, %2, %3) <{method = "Queue", operandSegmentSizes = array<i32: 3, 1>, parameters = []}> : (!ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_DMA_WRITE_REQ() attributes {event = "DMA_WRITE_REQ", extern = true, type = "controller"} {
    %0 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 0>}> : () -> !ep2.port<true, false>
    %1 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 1>}> : () -> !ep2.port<true, false>
    %2 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 2>}> : () -> !ep2.port<true, false>
    %3 = "ep2.constant"() <{value = #ep2.port<"DMA_WRITE_REQ" : "dma_write", 0>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%0, %1, %2, %3) <{method = "Queue", operandSegmentSizes = array<i32: 3, 1>, parameters = []}> : (!ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_OoO_DETECT() attributes {event = "OoO_DETECT", scope = #ep2<scope<"OoO_DETECT:OoO_detection"> ["flow_id"]>, type = "controller"} {
    %0 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 0>}> : () -> !ep2.port<true, false>
    %1 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 0>}> : () -> !ep2.port<false, true>
    %2 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 1>}> : () -> !ep2.port<false, true>
    %3 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 2>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%0, %1, %2, %3) <{method = "PartitionByScope", operandSegmentSizes = array<i32: 1, 3>, parameters = ["flow_id"]}> : (!ep2.port<true, false>, !ep2.port<false, true>, !ep2.port<false, true>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_OoO_DETECT_OoO_detection(%arg0: !ep2.context, %arg1: !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>) attributes {atom = "OoO_detection", event = "OoO_DETECT", instances = ["cu3", "cu4", "cu5"], instances_flow_table = ["cu3", "cu4", "cu5"], scope = #ep2<scope<"OoO_DETECT:OoO_detection"> ["flow_id"]>, type = "handler"} {
    %0 = "ep2.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "ep2.constant"() <{value = "ack_gen"}> : () -> !ep2.atom
    %2 = "ep2.constant"() <{value = "dma_write"}> : () -> !ep2.atom
    %3 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.buf>
    %4 = "ep2.load"(%3) : (!ep2.conref<!ep2.buf>) -> !ep2.buf
    %5 = ep2.struct_access %arg1[0] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %6 = "ep2.init"() : () -> !ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>
    %7 = "ep2.bitcast"(%5) : (i32) -> i16
    %8 = "ep2.lookup"(%6, %7) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>, i16) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %9 = "ep2.init"() : () -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
    %10 = "ep2.init"() : () -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %11 = ep2.struct_access %8[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %12 = ep2.struct_access %arg1[2] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %13 = "ep2.sub"(%11, %12) : (i32, i32) -> i32
    %14 = ep2.struct_access %arg1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %15 = "ep2.sub"(%14, %13) : (i32, i32) -> i32
    %16 = ep2.struct_access %8[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %17 = "ep2.cmp"(%15, %16) <{predicate = 60 : i16}> : (i32, i32) -> i1
    cf.cond_br %17, ^bb2(%0 : i32), ^bb1
  ^bb1:  // pred: ^bb0
    %18 = "ep2.sub"(%15, %16) : (i32, i32) -> i32
    cf.br ^bb2(%18 : i32)
  ^bb2(%19: i32):  // 2 preds: ^bb0, ^bb1
    %20 = "ep2.add"(%13, %19) : (i32, i32) -> i32
    %21 = "ep2.sub"(%14, %20) : (i32, i32) -> i32
    %22 = "ep2.cmp"(%21, %0) <{predicate = 62 : i16}> : (i32, i32) -> i1
    cf.cond_br %22, ^bb3, ^bb4(%8 : !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>)
  ^bb3:  // pred: ^bb2
    %23 = ep2.struct_access %8[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %24 = "ep2.sub"(%16, %21) : (i32, i32) -> i32
    %25 = "ep2.struct_update"(%8, %24) <{index = 4 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %26 = ep2.struct_access %25[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %27 = "ep2.add"(%26, %21) : (i32, i32) -> i32
    %28 = "ep2.struct_update"(%25, %27) <{index = 5 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %29 = ep2.struct_access %28[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %30 = "ep2.add"(%29, %21) : (i32, i32) -> i32
    %31 = "ep2.struct_update"(%28, %30) <{index = 6 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %32 = "ep2.struct_update"(%9, %23) <{index = 0 : i64}> : (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
    %33 = "ep2.struct_update"(%32, %21) <{index = 1 : i64}> : (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
    %34 = "ep2.init"(%2, %arg0, %4, %33) : (!ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) -> !ep2.struct<"DMA_WRITE_REQ" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>>
    ep2.return %34 : !ep2.struct<"DMA_WRITE_REQ" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>>
    cf.br ^bb4(%31 : !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>)
  ^bb4(%35: !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>):  // 2 preds: ^bb2, ^bb3
    "ep2.update"(%6, %7, %35) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>, i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>) -> ()
    %36 = ep2.struct_access %35[0] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %37 = "ep2.struct_update"(%10, %36) <{index = 0 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %38 = ep2.struct_access %35[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %39 = "ep2.struct_update"(%37, %38) <{index = 1 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %40 = ep2.struct_access %35[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %41 = "ep2.struct_update"(%39, %40) <{index = 2 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %42 = "ep2.init"(%1, %arg0, %41) : (!ep2.atom, !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>) -> !ep2.struct<"ACK_GEN" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    ep2.return %42 : !ep2.struct<"ACK_GEN" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_ACK_GEN_ack_gen(%arg0: !ep2.context, %arg1: !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>) attributes {atom = "ack_gen", event = "ACK_GEN", instances = ["cu6"], type = "handler"} {
    %0 = "ep2.constant"() <{value = 64 : i16}> : () -> i16
    %1 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %2 = "ep2.context_ref"(%arg0) <{name = "ip_header_sub_1"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>>
    %3 = "ep2.context_ref"(%arg0) <{name = "eth_header_sub_0"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>>
    %4 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.buf>
    %5 = "ep2.context_ref"(%arg0) <{name = "tcp_header_sub_0"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>>
    %6 = "ep2.load"(%5) : (!ep2.conref<!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>>) -> !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>
    %7 = "ep2.load"(%2) : (!ep2.conref<!ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>>) -> !ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>
    %8 = "ep2.load"(%3) : (!ep2.conref<!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>>) -> !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>
    %9 = "ep2.load"(%4) : (!ep2.conref<!ep2.buf>) -> !ep2.buf
    %10 = ep2.struct_access %8[0] : <"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16> -> i32
    %11 = ep2.struct_access %8[1] : <"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16> -> i16
    %12 = ep2.struct_access %8[2] : <"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16> -> i32
    %13 = "ep2.struct_update"(%8, %12) <{index = 0 : i64}> : (!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>, i32) -> !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>
    %14 = ep2.struct_access %13[3] : <"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16> -> i16
    %15 = "ep2.struct_update"(%13, %14) <{index = 1 : i64}> : (!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>, i16) -> !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>
    %16 = "ep2.struct_update"(%15, %10) <{index = 2 : i64}> : (!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>, i32) -> !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>
    %17 = "ep2.struct_update"(%16, %11) <{index = 3 : i64}> : (!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>, i16) -> !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>
    %18 = ep2.struct_access %7[0] : <"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32> -> i32
    %19 = ep2.struct_access %7[1] : <"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32> -> i32
    %20 = "ep2.struct_update"(%7, %19) <{index = 0 : i64}> : (!ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>
    %21 = "ep2.struct_update"(%20, %18) <{index = 1 : i64}> : (!ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>
    %22 = ep2.struct_access %6[0] : <"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32> -> i16
    %23 = ep2.struct_access %6[1] : <"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32> -> i16
    %24 = "ep2.struct_update"(%6, %23) <{index = 0 : i64}> : (!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>, i16) -> !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>
    %25 = "ep2.struct_update"(%24, %22) <{index = 1 : i64}> : (!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>, i16) -> !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>
    %26 = ep2.struct_access %arg1[0] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %27 = "ep2.struct_update"(%25, %26) <{index = 2 : i64}> : (!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>, i32) -> !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>
    %28 = ep2.struct_access %arg1[1] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %29 = "ep2.struct_update"(%27, %28) <{index = 3 : i64}> : (!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>, i32) -> !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>
    "ep2.emit_offset"(%9, %17) <{offset = 0 : i64}> : (!ep2.buf, !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i32, i16, i32, i16>) -> ()
    "ep2.emit_offset"(%9, %0) <{offset = 128 : i64}> : (!ep2.buf, i16) -> ()
    "ep2.emit_offset"(%9, %21) <{offset = 208 : i64}> : (!ep2.buf, !ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>) -> ()
    "ep2.emit_offset"(%9, %29) <{offset = 304 : i64}> : (!ep2.buf, !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>) -> ()
    %30 = "ep2.init"(%1, %arg0, %9) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    ep2.return %30 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}

