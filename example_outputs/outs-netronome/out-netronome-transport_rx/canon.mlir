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
    %3 = "ep2.extract_offset"(%arg1) <{offset = 0 : i64}> : (!ep2.buf) -> !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48>
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
    %18 = "ep2.context_ref"(%arg0) <{name = "eth_header_sub_0"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48>>
    "ep2.store"(%18, %3) : (!ep2.conref<!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48>>, !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48>) -> ()
    %19 = "ep2.context_ref"(%arg0) <{name = "tcp_header_sub_0"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>>
    "ep2.store"(%19, %6) : (!ep2.conref<!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>>, !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>) -> ()
    %20 = "ep2.init"(%0, %arg0, %14) : (!ep2.atom, !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>) -> !ep2.struct<"OoO_DETECT" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    ep2.return %20 : !ep2.struct<"OoO_DETECT" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_DMA_WRITE_REQ() attributes {event = "DMA_WRITE_REQ", extern = true, type = "controller"} {
    %0 = "ep2.constant"() <{value = 256 : i64}> : () -> i64
    %1 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %2 = ep2.call @Queue(%0, %1, %1) : (i64, i64, i64) -> i64
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_OoO_DETECT() attributes {event = "OoO_DETECT", type = "controller"} {
    %0 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 0>}> : () -> !ep2.port<true, false>
    %1 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 1>}> : () -> !ep2.port<true, false>
    %2 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 2>}> : () -> !ep2.port<true, false>
    %3 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 3>}> : () -> !ep2.port<true, false>
    %4 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 0>}> : () -> !ep2.port<false, true>
    %5 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 1>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%0, %1, %2, %3, %4) <{method = "PartitionByScope", operandSegmentSizes = array<i32: 4, 1>, parameters = ["flow_id", 0]}> : (!ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.connect"(%0, %1, %2, %3, %5) <{method = "PartitionByScope", operandSegmentSizes = array<i32: 4, 1>, parameters = ["flow_id", 1]}> : (!ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_OoO_DETECT_OoO_detection(%arg0: !ep2.context, %arg1: !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>) attributes {atom = "OoO_detection", event = "OoO_DETECT", instances = ["cu3", "cu4", "cu5"], scope = #ep2<scope<"OoO_DETECT:OoO_detection"> ["flow_id"]>, type = "handler"} {
    %0 = "ep2.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "ep2.constant"() <{value = "ack_gen"}> : () -> !ep2.atom
    %2 = "ep2.constant"() <{value = "dma_write"}> : () -> !ep2.atom
    %3 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.buf>
    %4 = "ep2.load"(%3) : (!ep2.conref<!ep2.buf>) -> !ep2.buf
    %5 = "ep2.init"() : () -> !ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>
    %6 = ep2.struct_access %arg1[0] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %7 = "ep2.bitcast"(%6) : (i32) -> i16
    %8 = "ep2.lookup"(%5, %7) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>, i16) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
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
    %22 = "ep2.cmp"(%13, %14) <{predicate = 41 : i16}> : (i32, i32) -> i1
    cf.cond_br %22, ^bb3, ^bb6
  ^bb3:  // pred: ^bb2
    %23 = "ep2.cmp"(%21, %0) <{predicate = 62 : i16}> : (i32, i32) -> i1
    cf.cond_br %23, ^bb4, ^bb5(%8 : !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>)
  ^bb4:  // pred: ^bb3
    %24 = ep2.struct_access %8[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %25 = "ep2.sub"(%16, %21) : (i32, i32) -> i32
    %26 = "ep2.struct_update"(%8, %25) <{index = 4 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %27 = ep2.struct_access %26[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %28 = "ep2.add"(%27, %21) : (i32, i32) -> i32
    %29 = "ep2.struct_update"(%26, %28) <{index = 5 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %30 = ep2.struct_access %29[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %31 = "ep2.add"(%30, %21) : (i32, i32) -> i32
    %32 = "ep2.struct_update"(%29, %31) <{index = 6 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %33 = "ep2.struct_update"(%9, %24) <{index = 0 : i64}> : (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
    %34 = "ep2.struct_update"(%33, %21) <{index = 1 : i64}> : (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
    %35 = "ep2.init"(%2, %arg0, %4, %34) : (!ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) -> !ep2.struct<"DMA_WRITE_REQ" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>>
    ep2.return %35 : !ep2.struct<"DMA_WRITE_REQ" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>>
    cf.br ^bb5(%32 : !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>)
  ^bb5(%36: !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>):  // 2 preds: ^bb3, ^bb4
    "ep2.update"(%5, %7, %36) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>, i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>) -> ()
    %37 = ep2.struct_access %36[0] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %38 = "ep2.struct_update"(%10, %37) <{index = 0 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %39 = ep2.struct_access %36[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %40 = "ep2.struct_update"(%38, %39) <{index = 1 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %41 = ep2.struct_access %36[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %42 = "ep2.struct_update"(%40, %41) <{index = 2 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %43 = "ep2.init"(%1, %arg0, %42) : (!ep2.atom, !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>) -> !ep2.struct<"ACK_GEN" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    ep2.return %43 : !ep2.struct<"ACK_GEN" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb2, ^bb5
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_ACK_GEN() attributes {event = "ACK_GEN", type = "controller"} {
    %0 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 0>}> : () -> !ep2.port<true, false>
    %1 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 1>}> : () -> !ep2.port<true, false>
    %2 = "ep2.constant"() <{value = #ep2.port<"ACK_GEN" : "ack_gen", 0>}> : () -> !ep2.port<false, true>
    %3 = "ep2.constant"() <{value = #ep2.port<"ACK_GEN" : "ack_gen", 1>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%0, %2) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [100]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.connect"(%1, %3) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [100]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_ACK_GEN_ack_gen(%arg0: !ep2.context, %arg1: !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>) attributes {atom = "ack_gen", event = "ACK_GEN", instances = ["cu6"], type = "handler"} {
    %0 = "ep2.constant"() <{value = 64 : i16}> : () -> i16
    %1 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %2 = "ep2.context_ref"(%arg0) <{name = "ip_header_sub_1"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>>
    %3 = "ep2.context_ref"(%arg0) <{name = "eth_header_sub_0"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48>>
    %4 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.buf>
    %5 = "ep2.context_ref"(%arg0) <{name = "tcp_header_sub_0"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>>
    %6 = "ep2.load"(%2) : (!ep2.conref<!ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>>) -> !ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>
    %7 = "ep2.load"(%3) : (!ep2.conref<!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48>>) -> !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48>
    %8 = "ep2.load"(%5) : (!ep2.conref<!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>>) -> !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>
    %9 = "ep2.load"(%4) : (!ep2.conref<!ep2.buf>) -> !ep2.buf
    %10 = ep2.struct_access %7[0] : <"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48> -> i48
    %11 = ep2.struct_access %7[1] : <"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48> -> i48
    %12 = "ep2.struct_update"(%7, %11) <{index = 0 : i64}> : (!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48>, i48) -> !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48>
    %13 = "ep2.struct_update"(%12, %10) <{index = 1 : i64}> : (!ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48>, i48) -> !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48>
    %14 = ep2.struct_access %6[0] : <"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32> -> i32
    %15 = ep2.struct_access %6[1] : <"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32> -> i32
    %16 = "ep2.struct_update"(%6, %15) <{index = 0 : i64}> : (!ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>
    %17 = "ep2.struct_update"(%16, %14) <{index = 1 : i64}> : (!ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>
    %18 = ep2.struct_access %8[0] : <"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32> -> i16
    %19 = ep2.struct_access %8[1] : <"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32> -> i16
    %20 = "ep2.struct_update"(%8, %19) <{index = 0 : i64}> : (!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>, i16) -> !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>
    %21 = "ep2.struct_update"(%20, %18) <{index = 1 : i64}> : (!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>, i16) -> !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>
    %22 = ep2.struct_access %arg1[0] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %23 = "ep2.struct_update"(%21, %22) <{index = 2 : i64}> : (!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>, i32) -> !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>
    %24 = ep2.struct_access %arg1[1] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %25 = "ep2.struct_update"(%23, %24) <{index = 3 : i64}> : (!ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>, i32) -> !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>
    "ep2.emit_offset"(%9, %13) <{offset = 0 : i64}> : (!ep2.buf, !ep2.struct<"eth_header_t_sub_0" : isEvent = false, elementTypes = i48, i48>) -> ()
    "ep2.emit_offset"(%9, %0) <{offset = 128 : i64}> : (!ep2.buf, i16) -> ()
    "ep2.emit_offset"(%9, %17) <{offset = 208 : i64}> : (!ep2.buf, !ep2.struct<"ip_header_t_sub_1" : isEvent = false, elementTypes = i32, i32>) -> ()
    "ep2.emit_offset"(%9, %25) <{offset = 304 : i64}> : (!ep2.buf, !ep2.struct<"tcp_header_t_sub_0" : isEvent = false, elementTypes = i16, i16, i32, i32>) -> ()
    %26 = "ep2.init"(%1, %arg0, %9) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    ep2.return %26 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}

