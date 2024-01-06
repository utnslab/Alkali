module {
  ep2.func private @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_WRITE_REQ_dma_write(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) attributes {atom = "dma_write", event = "DMA_WRITE_REQ", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["i1cu1", "i1cu2", "i2cu1", "i2cu2"], type = "handler"} {
    %0 = "ep2.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "ep2.constant"() <{value = "OoO_detection"}> : () -> !ep2.atom
    %2 = "ep2.constant"() <{value = 14 : i64}> : () -> i64
    %3 = "ep2.init"() : () -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %4 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %5 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %6 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %7 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>
    "ep2.store"(%7, %4) : (!ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> ()
    %8 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>>
    "ep2.store"(%8, %5) : (!ep2.conref<!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>>, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    %9 = "ep2.context_ref"(%arg0) <{name = "tcp_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>>
    "ep2.store"(%9, %6) : (!ep2.conref<!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>>, !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>) -> ()
    %10 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.buf>
    "ep2.store"(%10, %arg1) : (!ep2.conref<!ep2.buf>, !ep2.buf) -> ()
    %11 = ep2.struct_access %5[1] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i16
    %12 = "ep2.add"(%11, %2) : (i16, i64) -> i16
    %13 = "ep2.bitcast"(%12) : (i16) -> i32
    %14 = "ep2.struct_update"(%3, %13) <{index = 1 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %15 = ep2.struct_access %6[2] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i32
    %16 = "ep2.struct_update"(%14, %15) <{index = 2 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %17 = "ep2.struct_update"(%16, %0) <{index = 0 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %18 = "ep2.init"(%1, %arg0, %17) : (!ep2.atom, !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>) -> !ep2.struct<"OoO_DETECT" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    ep2.return %18 : !ep2.struct<"OoO_DETECT" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
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
    "ep2.connect"(%0, %1, %4) <{method = "Queue", operandSegmentSizes = array<i32: 2, 1>}> : (!ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.connect"(%2, %3, %5) <{method = "Queue", operandSegmentSizes = array<i32: 2, 1>}> : (!ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_OoO_DETECT_OoO_detection(%arg0: !ep2.context, %arg1: !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>) attributes {atom = "OoO_detection", event = "OoO_DETECT", instances = ["i1cu3", "i2cu3"], scope = #ep2<scope<"OoO_DETECT:OoO_detection"> ["flow_id"]>, type = "handler"} {
    %0 = "ep2.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "ep2.constant"() <{value = "ack_gen"}> : () -> !ep2.atom
    %2 = "ep2.constant"() <{value = "dma_write"}> : () -> !ep2.atom
    %3 = "ep2.constant"() <{value = 0 : i64}> : () -> i64
    %4 = "ep2.init"() : () -> !ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>
    %5 = ep2.struct_access %arg1[0] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %6 = "ep2.bitcast"(%5) : (i32) -> i16
    %7 = "ep2.lookup"(%4, %6) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>, i16) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %8 = "ep2.init"() : () -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
    %9 = "ep2.init"() : () -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %10 = ep2.struct_access %7[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %11 = ep2.struct_access %arg1[2] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %12 = "ep2.sub"(%10, %11) : (i32, i32) -> i32
    %13 = ep2.struct_access %arg1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %14 = "ep2.sub"(%13, %12) : (i32, i32) -> i32
    %15 = ep2.struct_access %7[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %16 = "ep2.cmp"(%14, %15) <{predicate = 60 : i16}> : (i32, i32) -> i1
    %17 = scf.if %16 -> (i32) {
      scf.yield %0 : i32
    } else {
      %21 = "ep2.sub"(%14, %15) : (i32, i32) -> i32
      scf.yield %21 : i32
    }
    %18 = "ep2.add"(%12, %17) : (i32, i32) -> i32
    %19 = "ep2.sub"(%13, %18) : (i32, i32) -> i32
    %20 = "ep2.cmp"(%12, %13) <{predicate = 41 : i16}> : (i32, i32) -> i1
    scf.if %20 {
      %21 = "ep2.cmp"(%19, %3) <{predicate = 62 : i16}> : (i32, i64) -> i1
      %22 = scf.if %21 -> (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>) {
        %30 = ep2.struct_access %7[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
        %31 = "ep2.sub"(%15, %19) : (i32, i32) -> i32
        %32 = "ep2.struct_update"(%7, %31) <{index = 4 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
        %33 = ep2.struct_access %32[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
        %34 = "ep2.add"(%33, %19) : (i32, i32) -> i32
        %35 = "ep2.struct_update"(%32, %34) <{index = 5 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
        %36 = ep2.struct_access %35[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
        %37 = "ep2.add"(%36, %19) : (i32, i32) -> i32
        %38 = "ep2.struct_update"(%35, %37) <{index = 6 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
        %39 = "ep2.struct_update"(%8, %30) <{index = 0 : i64}> : (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
        %40 = "ep2.struct_update"(%39, %19) <{index = 1 : i64}> : (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
        %41 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.buf>
        %42 = "ep2.load"(%41) : (!ep2.conref<!ep2.buf>) -> !ep2.buf
        %43 = "ep2.init"(%2, %arg0, %42, %40) : (!ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) -> !ep2.struct<"DMA_WRITE_REQ" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>>
        ep2.return %43 : !ep2.struct<"DMA_WRITE_REQ" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>>
        scf.yield %38 : !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
      } else {
        scf.yield %7 : !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
      }
      "ep2.update"(%4, %6, %22) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>, i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>) -> ()
      %23 = ep2.struct_access %22[0] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %24 = "ep2.struct_update"(%9, %23) <{index = 0 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
      %25 = ep2.struct_access %22[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %26 = "ep2.struct_update"(%24, %25) <{index = 1 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
      %27 = ep2.struct_access %22[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %28 = "ep2.struct_update"(%26, %27) <{index = 2 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
      %29 = "ep2.init"(%1, %arg0, %28) : (!ep2.atom, !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>) -> !ep2.struct<"ACK_GEN" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
      ep2.return %29 : !ep2.struct<"ACK_GEN" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    }
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_ACK_GEN() attributes {event = "ACK_GEN", type = "controller"} {
    %0 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 0>}> : () -> !ep2.port<true, false>
    %1 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 1>}> : () -> !ep2.port<true, false>
    %2 = "ep2.constant"() <{value = #ep2.port<"ACK_GEN" : "ack_gen", 0>}> : () -> !ep2.port<false, true>
    %3 = "ep2.constant"() <{value = #ep2.port<"ACK_GEN" : "ack_gen", 1>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%0, %2) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.connect"(%1, %3) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_ACK_GEN_ack_gen(%arg0: !ep2.context, %arg1: !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>) attributes {atom = "ack_gen", event = "ACK_GEN", instances = ["i1cu4", "i2cu4"], type = "handler"} {
    %0 = "ep2.constant"() <{value = 64 : i16}> : () -> i16
    %1 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %2 = "ep2.init"() : () -> !ep2.buf
    %3 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>
    %4 = "ep2.load"(%3) : (!ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %5 = ep2.struct_access %4[0] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %6 = ep2.struct_access %4[1] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %7 = "ep2.struct_update"(%4, %6) <{index = 0 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i48) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %8 = "ep2.struct_update"(%7, %5) <{index = 1 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i48) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %9 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>>
    %10 = "ep2.load"(%9) : (!ep2.conref<!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>>) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %11 = ep2.struct_access %10[6] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %12 = ep2.struct_access %10[7] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %13 = "ep2.struct_update"(%10, %12) <{index = 6 : i64}> : (!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>, i32) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %14 = "ep2.struct_update"(%13, %11) <{index = 7 : i64}> : (!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>, i32) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %15 = "ep2.struct_update"(%14, %0) <{index = 1 : i64}> : (!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>, i16) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %16 = "ep2.context_ref"(%arg0) <{name = "tcp_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>>
    %17 = "ep2.load"(%16) : (!ep2.conref<!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>>) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %18 = ep2.struct_access %17[0] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %19 = ep2.struct_access %17[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %20 = "ep2.struct_update"(%17, %19) <{index = 0 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i16) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %21 = "ep2.struct_update"(%20, %18) <{index = 1 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i16) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %22 = ep2.struct_access %arg1[0] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %23 = "ep2.struct_update"(%21, %22) <{index = 2 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i32) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %24 = ep2.struct_access %arg1[1] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %25 = "ep2.struct_update"(%23, %24) <{index = 3 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i32) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    "ep2.emit"(%2, %8) : (!ep2.buf, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> ()
    "ep2.emit"(%2, %15) : (!ep2.buf, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    "ep2.emit"(%2, %25) : (!ep2.buf, !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>) -> ()
    %26 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.buf>
    %27 = "ep2.load"(%26) : (!ep2.conref<!ep2.buf>) -> !ep2.buf
    "ep2.emit"(%2, %27) : (!ep2.buf, !ep2.buf) -> ()
    %28 = "ep2.init"(%1, %arg0, %2) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    ep2.return %28 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}

