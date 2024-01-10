module {
  ep2.func private @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_WRITE_REQ_dma_write(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) attributes {atom = "dma_write", event = "DMA_WRITE_REQ", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["i1cu2", "i1cu3", "i2cu2", "i2cu3"], type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %1 = "ep2.init"() : () -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %2 = "ep2.init"() : () -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %3 = "ep2.init"() : () -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %4 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %5 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %6 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %7 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%7, %4) : (!ep2.conref<!ep2.any>, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> ()
    %8 = "ep2.nop"() : () -> none
    %9 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%9, %5) : (!ep2.conref<!ep2.any>, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    %10 = "ep2.nop"() : () -> none
    %11 = "ep2.context_ref"(%arg0) <{name = "tcp_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%11, %6) : (!ep2.conref<!ep2.any>, !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>) -> ()
    %12 = "ep2.nop"() : () -> none
    %13 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%13, %arg1) : (!ep2.conref<!ep2.any>, !ep2.buf) -> ()
    %14 = "ep2.nop"() : () -> none
    %15 = ep2.struct_access %3[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %16 = ep2.struct_access %5[1] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i16
    %17 = "ep2.constant"() <{value = 14 : i64}> : () -> i64
    %18 = "ep2.add"(%16, %17) : (i16, i64) -> i16
    %19 = "ep2.struct_update"(%3, %18) <{index = 1 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i16) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %20 = ep2.struct_access %19[2] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %21 = ep2.struct_access %6[2] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i32
    %22 = "ep2.struct_update"(%19, %21) <{index = 2 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %23 = ep2.struct_access %22[0] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %24 = ep2.struct_access %6[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %25 = "ep2.struct_update"(%22, %24) <{index = 0 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i16) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %26 = "ep2.context_ref"(%arg0) <{name = "flow_id"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %27 = ep2.struct_access %6[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    "ep2.store"(%26, %27) : (!ep2.conref<!ep2.any>, i16) -> ()
    %28 = "ep2.nop"() : () -> none
    %29 = "ep2.constant"() <{value = "OoO_detection"}> : () -> !ep2.atom
    %30 = "ep2.init"(%29, %arg0, %25) : (!ep2.atom, !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>) -> !ep2.struct<"OoO_DETECT" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    ep2.return %30 : !ep2.struct<"OoO_DETECT" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_DMA_WRITE_REQ() attributes {event = "DMA_WRITE_REQ", extern = true, type = "controller"} {
    %0 = "ep2.constant"() <{value = 256 : i64}> : () -> i64
    %1 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %2 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %3 = ep2.call @Queue(%0, %1, %2) : (i64, i64, i64) -> i64
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
    %6 = "ep2.nop"() : () -> none
    "ep2.connect"(%0, %1, %2, %3, %5) <{method = "PartitionByScope", operandSegmentSizes = array<i32: 4, 1>, parameters = ["flow_id", 1]}> : (!ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %7 = "ep2.nop"() : () -> none
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_OoO_DETECT_OoO_detection(%arg0: !ep2.context, %arg1: !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>) attributes {atom = "OoO_detection", event = "OoO_DETECT", instances = ["i1cu4", "i2cu4"], scope = #ep2<scope<"OoO_DETECT:OoO_detection"> ["flow_id"]>, type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>
    %1 = "ep2.init"() : () -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %2 = ep2.struct_access %arg1[0] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %3 = "ep2.lookup"(%0, %2) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %4 = "ep2.init"() : () -> i32
    %5 = "ep2.init"() : () -> i32
    %6 = "ep2.init"() : () -> i32
    %7 = "ep2.init"() : () -> i32
    %8 = "ep2.init"() : () -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
    %9 = "ep2.init"() : () -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %10 = ep2.struct_access %3[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %11 = ep2.struct_access %arg1[2] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %12 = "ep2.sub"(%10, %11) : (i32, i32) -> i32
    %13 = ep2.struct_access %arg1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %14 = "ep2.sub"(%13, %12) : (i32, i32) -> i32
    %15 = ep2.struct_access %3[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %16 = "ep2.cmp"(%14, %15) <{predicate = 60 : i16}> : (i32, i32) -> i1
    %17 = scf.if %16 -> (i32) {
      %24 = "ep2.constant"() <{value = 0 : i64}> : () -> i64
      %25 = "ep2.bitcast"(%24) : (i64) -> i32
      scf.yield %25 : i32
    } else {
      %24 = ep2.struct_access %arg1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
      %25 = "ep2.sub"(%24, %12) : (i32, i32) -> i32
      %26 = ep2.struct_access %3[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %27 = "ep2.sub"(%25, %26) : (i32, i32) -> i32
      scf.yield %27 : i32
    }
    %18 = ep2.struct_access %arg1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %19 = "ep2.add"(%12, %17) : (i32, i32) -> i32
    %20 = "ep2.sub"(%18, %19) : (i32, i32) -> i32
    %21 = ep2.struct_access %arg1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %22 = "ep2.cmp"(%12, %21) <{predicate = 41 : i16}> : (i32, i32) -> i1
    %23:4 = scf.if %22 -> (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>) {
      %24 = "ep2.constant"() <{value = 0 : i64}> : () -> i64
      %25 = "ep2.cmp"(%20, %24) <{predicate = 62 : i16}> : (i32, i64) -> i1
      %26:3 = scf.if %25 -> (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>) {
        %40 = ep2.struct_access %3[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
        %41 = ep2.struct_access %3[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
        %42 = ep2.struct_access %3[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
        %43 = "ep2.sub"(%42, %20) : (i32, i32) -> i32
        %44 = "ep2.struct_update"(%3, %43) <{index = 4 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
        %45 = ep2.struct_access %44[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
        %46 = ep2.struct_access %44[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
        %47 = "ep2.add"(%46, %20) : (i32, i32) -> i32
        %48 = "ep2.struct_update"(%44, %47) <{index = 5 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
        %49 = ep2.struct_access %48[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
        %50 = ep2.struct_access %48[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
        %51 = "ep2.add"(%50, %20) : (i32, i32) -> i32
        %52 = "ep2.struct_update"(%48, %51) <{index = 6 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
        %53 = ep2.struct_access %8[0] : <"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32> -> i32
        %54 = "ep2.struct_update"(%8, %40) <{index = 0 : i64}> : (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
        %55 = ep2.struct_access %54[1] : <"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32> -> i32
        %56 = "ep2.struct_update"(%54, %20) <{index = 1 : i64}> : (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
        %57 = "ep2.constant"() <{value = "dma_write"}> : () -> !ep2.atom
        %58 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
        %59 = "ep2.load"(%58) : (!ep2.conref<!ep2.any>) -> !ep2.any
        %60 = "ep2.init"(%57, %arg0, %59, %56) : (!ep2.atom, !ep2.context, !ep2.any, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) -> !ep2.struct<"DMA_WRITE_REQ" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>>
        ep2.return %60 : !ep2.struct<"DMA_WRITE_REQ" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>>
        scf.yield %56, %40, %52 : !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
      } else {
        scf.yield %8, %7, %3 : !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
      }
      %27 = ep2.struct_access %arg1[0] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
      "ep2.update"(%0, %27, %26#2) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>) -> ()
      %28 = "ep2.nop"() : () -> none
      %29 = ep2.struct_access %9[0] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
      %30 = ep2.struct_access %26#2[0] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %31 = "ep2.struct_update"(%9, %30) <{index = 0 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
      %32 = ep2.struct_access %31[1] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
      %33 = ep2.struct_access %26#2[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %34 = "ep2.struct_update"(%31, %33) <{index = 1 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
      %35 = ep2.struct_access %34[2] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
      %36 = ep2.struct_access %26#2[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %37 = "ep2.struct_update"(%34, %36) <{index = 2 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
      %38 = "ep2.constant"() <{value = "ack_gen"}> : () -> !ep2.atom
      %39 = "ep2.init"(%38, %arg0, %37) : (!ep2.atom, !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>) -> !ep2.struct<"ACK_GEN" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
      ep2.return %39 : !ep2.struct<"ACK_GEN" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
      scf.yield %37, %26#0, %26#1, %26#2 : !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    } else {
      scf.yield %9, %8, %7, %3 : !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    }
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_ACK_GEN() attributes {event = "ACK_GEN", type = "controller"} {
    %0 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 0>}> : () -> !ep2.port<true, false>
    %1 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", 1>}> : () -> !ep2.port<true, false>
    %2 = "ep2.constant"() <{value = #ep2.port<"ACK_GEN" : "ack_gen", 0>}> : () -> !ep2.port<false, true>
    %3 = "ep2.constant"() <{value = #ep2.port<"ACK_GEN" : "ack_gen", 1>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%0, %2) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [100]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %4 = "ep2.nop"() : () -> none
    "ep2.connect"(%1, %3) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [100]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %5 = "ep2.nop"() : () -> none
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_ACK_GEN_ack_gen(%arg0: !ep2.context, %arg1: !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>) attributes {atom = "ack_gen", event = "ACK_GEN", instances = ["i1cu5", "i2cu5"], type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %1 = "ep2.init"() : () -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %2 = "ep2.init"() : () -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %3 = "ep2.init"() : () -> i32
    %4 = "ep2.init"() : () -> i48
    %5 = "ep2.init"() : () -> i16
    %6 = "ep2.init"() : () -> !ep2.buf
    %7 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %8 = "ep2.load"(%7) : (!ep2.conref<!ep2.any>) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %9 = ep2.struct_access %8[0] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %10 = ep2.struct_access %8[0] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %11 = ep2.struct_access %8[1] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %12 = "ep2.struct_update"(%8, %11) <{index = 0 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i48) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %13 = ep2.struct_access %12[1] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %14 = "ep2.struct_update"(%12, %9) <{index = 1 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i48) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %15 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %16 = "ep2.load"(%15) : (!ep2.conref<!ep2.any>) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %17 = ep2.struct_access %16[6] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %18 = ep2.struct_access %16[6] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %19 = ep2.struct_access %16[7] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %20 = "ep2.struct_update"(%16, %19) <{index = 6 : i64}> : (!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>, i32) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %21 = ep2.struct_access %20[7] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %22 = "ep2.struct_update"(%20, %17) <{index = 7 : i64}> : (!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>, i32) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %23 = ep2.struct_access %22[1] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i16
    %24 = "ep2.constant"() <{value = 64 : i64}> : () -> i64
    %25 = "ep2.struct_update"(%22, %24) <{index = 1 : i64}> : (!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>, i64) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %26 = "ep2.context_ref"(%arg0) <{name = "tcp_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %27 = "ep2.load"(%26) : (!ep2.conref<!ep2.any>) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %28 = ep2.struct_access %27[0] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %29 = ep2.struct_access %27[0] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %30 = ep2.struct_access %27[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %31 = "ep2.struct_update"(%27, %30) <{index = 0 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i16) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %32 = ep2.struct_access %31[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %33 = "ep2.struct_update"(%31, %28) <{index = 1 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i16) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %34 = ep2.struct_access %33[2] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i32
    %35 = ep2.struct_access %arg1[0] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %36 = "ep2.struct_update"(%33, %35) <{index = 2 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i32) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %37 = ep2.struct_access %36[3] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i32
    %38 = ep2.struct_access %arg1[1] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %39 = "ep2.struct_update"(%36, %38) <{index = 3 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i32) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    "ep2.emit"(%6, %14) : (!ep2.buf, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> ()
    %40 = "ep2.nop"() : () -> none
    "ep2.emit"(%6, %25) : (!ep2.buf, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    %41 = "ep2.nop"() : () -> none
    "ep2.emit"(%6, %39) : (!ep2.buf, !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>) -> ()
    %42 = "ep2.nop"() : () -> none
    %43 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %44 = "ep2.load"(%43) : (!ep2.conref<!ep2.any>) -> !ep2.buf
    "ep2.emit"(%6, %44) : (!ep2.buf, !ep2.buf) -> ()
    %45 = "ep2.nop"() : () -> none
    %46 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %47 = "ep2.init"(%46, %arg0, %6) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    ep2.return %47 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}
