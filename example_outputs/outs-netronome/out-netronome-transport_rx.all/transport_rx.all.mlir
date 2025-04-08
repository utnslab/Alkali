module {
  ep2.func private @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_WRITE_REQ_dma_write(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) attributes {atom = "dma_write", event = "DMA_WRITE_REQ", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", type = "handler"} {
    %1 = "ep2.init"() : () -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %2 = "ep2.init"() : () -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %3 = "ep2.init"() : () -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %4 = "ep2.init"() : () -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %5 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %6 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %7 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %8 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%8, %5) : (!ep2.conref<!ep2.any>, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>) -> ()
    %9 = "ep2.nop"() : () -> none
    %10 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%10, %6) : (!ep2.conref<!ep2.any>, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    %11 = "ep2.nop"() : () -> none
    %12 = "ep2.context_ref"(%arg0) <{name = "tcp_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%12, %7) : (!ep2.conref<!ep2.any>, !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>) -> ()
    %13 = "ep2.nop"() : () -> none
    %14 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%14, %arg1) : (!ep2.conref<!ep2.any>, !ep2.buf) -> ()
    %15 = "ep2.nop"() : () -> none
    %16 = ep2.struct_access %4[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %17 = ep2.struct_access %6[1] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i16
    %18 = "ep2.constant"() <{value = 14 : i64}> : () -> i64
    %19 = "ep2.bitcast"(%18) : (i64) -> i16
    %20 = "ep2.add"(%17, %19) : (i16, i16) -> i16
    %21 = "ep2.struct_update"(%4, %20) <{index = 1 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i16) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %22 = ep2.struct_access %21[2] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %23 = ep2.struct_access %7[2] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i32
    %24 = "ep2.struct_update"(%21, %23) <{index = 2 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %25 = ep2.struct_access %24[0] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %26 = ep2.struct_access %7[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %27 = "ep2.struct_update"(%24, %26) <{index = 0 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i16) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %28 = "ep2.context_ref"(%arg0) <{name = "flow_id"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %29 = ep2.struct_access %7[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    "ep2.store"(%28, %29) : (!ep2.conref<!ep2.any>, i16) -> ()
    %30 = "ep2.nop"() : () -> none
    %31 = "ep2.constant"() <{value = "OoO_detection"}> : () -> !ep2.atom
    %32 = "ep2.init"(%31, %arg0, %27) : (!ep2.atom, !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>) -> !ep2.struct<"OoO_DETECT" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    ep2.return %32 : !ep2.struct<"OoO_DETECT" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_ACK_GEN() attributes {event = "ACK_GEN", type = "controller"} {
    %1 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", -1>}> : () -> !ep2.port<true, false>
    %2 = "ep2.constant"() <{value = #ep2.port<"ACK_GEN" : "ack_gen", -1>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%1, %2) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = []}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %3 = "ep2.nop"() : () -> none
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_DMA_WRITE_REQ() attributes {event = "DMA_WRITE_REQ", extern = true, type = "controller"} {
    %1 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", -1>}> : () -> !ep2.port<true, false>
    %2 = "ep2.constant"() <{value = #ep2.port<"DMA_WRITE_REQ" : "dma_write", -1>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%1, %2) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = []}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %3 = "ep2.nop"() : () -> none
    "ep2.terminate"() : () -> ()
  }
  %0 = "ep2.global"() <{name = "flow_table"}> {scope = #ep2<scope<"OoO_DETECT:OoO_detection"> ["flow_id"]>} : () -> !ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>
  ep2.func private @__controller_OoO_DETECT() attributes {event = "OoO_DETECT", scope = #ep2<scope<"OoO_DETECT:OoO_detection"> ["flow_id"]>, type = "controller"} {
    %1 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", -1>}> : () -> !ep2.port<true, false>
    %2 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", -1>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%1, %2) <{method = "PartitionByScope", operandSegmentSizes = array<i32: 1, 1>, parameters = ["flow_id"]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %3 = "ep2.nop"() : () -> none
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_OoO_DETECT_OoO_detection(%arg0: !ep2.context, %arg1: !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>) attributes {atom = "OoO_detection", event = "OoO_DETECT", scope = #ep2<scope<"OoO_DETECT:OoO_detection"> ["flow_id"]>, type = "handler"} {
    %1 = "ep2.init"() : () -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %2 = ep2.struct_access %arg1[0] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %3 = "ep2.global_import"() <{name = "flow_table"}> : () -> !ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>
    %4 = "ep2.lookup"(%3, %2) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %5 = "ep2.init"() : () -> i32
    %6 = "ep2.init"() : () -> i32
    %7 = "ep2.init"() : () -> i32
    %8 = "ep2.init"() : () -> i32
    %9 = "ep2.init"() : () -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
    %10 = "ep2.init"() : () -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %11 = ep2.struct_access %4[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %12 = ep2.struct_access %arg1[2] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %13 = "ep2.sub"(%11, %12) : (i32, i32) -> i32
    %14 = ep2.struct_access %arg1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %15 = "ep2.sub"(%14, %13) : (i32, i32) -> i32
    %16 = ep2.struct_access %4[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %17 = "ep2.cmp"(%15, %16) <{predicate = 60 : i16}> : (i32, i32) -> i1
    %18 = scf.if %17 -> (i32) {
      %40 = "ep2.constant"() <{value = 0 : i64}> : () -> i64
      %41 = "ep2.bitcast"(%40) : (i64) -> i32
      scf.yield %41 : i32
    } else {
      %40 = ep2.struct_access %arg1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
      %41 = "ep2.sub"(%40, %13) : (i32, i32) -> i32
      %42 = ep2.struct_access %4[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %43 = "ep2.sub"(%41, %42) : (i32, i32) -> i32
      scf.yield %43 : i32
    }
    %19 = ep2.struct_access %arg1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %20 = "ep2.add"(%13, %18) : (i32, i32) -> i32
    %21 = "ep2.sub"(%19, %20) : (i32, i32) -> i32
    %22 = "ep2.constant"() <{value = 0 : i64}> : () -> i64
    %23 = "ep2.bitcast"(%22) : (i64) -> i32
    %24 = "ep2.cmp"(%21, %23) <{predicate = 62 : i16}> : (i32, i32) -> i1
    %25:3 = scf.if %24 -> (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>) {
      %40 = ep2.struct_access %4[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %41 = ep2.struct_access %4[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %42 = ep2.struct_access %4[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %43 = "ep2.sub"(%42, %21) : (i32, i32) -> i32
      %44 = "ep2.struct_update"(%4, %43) <{index = 4 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
      %45 = ep2.struct_access %44[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %46 = ep2.struct_access %44[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %47 = "ep2.add"(%46, %21) : (i32, i32) -> i32
      %48 = "ep2.struct_update"(%44, %47) <{index = 5 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
      %49 = ep2.struct_access %48[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %50 = ep2.struct_access %48[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %51 = "ep2.add"(%50, %21) : (i32, i32) -> i32
      %52 = "ep2.struct_update"(%48, %51) <{index = 6 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
      %53 = ep2.struct_access %9[0] : <"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32> -> i32
      %54 = "ep2.struct_update"(%9, %40) <{index = 0 : i64}> : (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
      %55 = ep2.struct_access %54[1] : <"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32> -> i32
      %56 = "ep2.struct_update"(%54, %21) <{index = 1 : i64}> : (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
      %57 = "ep2.constant"() <{value = "dma_write"}> : () -> !ep2.atom
      %58 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
      %59 = "ep2.load"(%58) : (!ep2.conref<!ep2.any>) -> !ep2.any
      %60 = "ep2.init"(%57, %arg0, %59, %56) : (!ep2.atom, !ep2.context, !ep2.any, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) -> !ep2.struct<"DMA_WRITE_REQ" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>>
      ep2.return %60 : !ep2.struct<"DMA_WRITE_REQ" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>>
      scf.yield %56, %40, %52 : !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    } else {
      scf.yield %9, %8, %4 : !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    }
    %26 = ep2.struct_access %arg1[0] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %27 = "ep2.global_import"() <{name = "flow_table"}> : () -> !ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>
    "ep2.update"(%27, %26, %25#2) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>) -> ()
    %28 = "ep2.nop"() : () -> none
    %29 = ep2.struct_access %10[0] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %30 = ep2.struct_access %25#2[0] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %31 = "ep2.struct_update"(%10, %30) <{index = 0 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %32 = ep2.struct_access %31[1] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %33 = ep2.struct_access %25#2[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %34 = "ep2.struct_update"(%31, %33) <{index = 1 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %35 = ep2.struct_access %34[2] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %36 = ep2.struct_access %25#2[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %37 = "ep2.struct_update"(%34, %36) <{index = 2 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %38 = "ep2.constant"() <{value = "ack_gen"}> : () -> !ep2.atom
    %39 = "ep2.init"(%38, %arg0, %37) : (!ep2.atom, !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>) -> !ep2.struct<"ACK_GEN" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    ep2.return %39 : !ep2.struct<"ACK_GEN" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_ACK_GEN_ack_gen(%arg0: !ep2.context, %arg1: !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>) attributes {atom = "ack_gen", event = "ACK_GEN", type = "handler"} {
    %1 = "ep2.init"() : () -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %2 = "ep2.init"() : () -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %3 = "ep2.init"() : () -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %4 = "ep2.init"() : () -> i32
    %5 = "ep2.init"() : () -> i32
    %6 = "ep2.init"() : () -> i16
    %7 = "ep2.init"() : () -> i16
    %8 = "ep2.init"() : () -> !ep2.buf
    %9 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %10 = "ep2.load"(%9) : (!ep2.conref<!ep2.any>) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %11 = ep2.struct_access %10[0] : <"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16> -> i32
    %12 = ep2.struct_access %10[1] : <"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16> -> i16
    %13 = ep2.struct_access %10[0] : <"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16> -> i32
    %14 = ep2.struct_access %10[2] : <"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16> -> i32
    %15 = "ep2.struct_update"(%10, %14) <{index = 0 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>, i32) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %16 = ep2.struct_access %15[1] : <"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16> -> i16
    %17 = ep2.struct_access %15[3] : <"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16> -> i16
    %18 = "ep2.struct_update"(%15, %17) <{index = 1 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>, i16) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %19 = ep2.struct_access %18[2] : <"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16> -> i32
    %20 = "ep2.struct_update"(%18, %11) <{index = 2 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>, i32) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %21 = ep2.struct_access %20[3] : <"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16> -> i16
    %22 = "ep2.struct_update"(%20, %12) <{index = 3 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>, i16) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %23 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %24 = "ep2.load"(%23) : (!ep2.conref<!ep2.any>) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %25 = ep2.struct_access %24[6] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %26 = ep2.struct_access %24[6] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %27 = ep2.struct_access %24[7] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %28 = "ep2.struct_update"(%24, %27) <{index = 6 : i64}> : (!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>, i32) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %29 = ep2.struct_access %28[7] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %30 = "ep2.struct_update"(%28, %25) <{index = 7 : i64}> : (!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>, i32) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %31 = ep2.struct_access %30[1] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i16
    %32 = "ep2.constant"() <{value = 64 : i64}> : () -> i64
    %33 = "ep2.bitcast"(%32) : (i64) -> i16
    %34 = "ep2.struct_update"(%30, %33) <{index = 1 : i64}> : (!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>, i16) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %35 = "ep2.context_ref"(%arg0) <{name = "tcp_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %36 = "ep2.load"(%35) : (!ep2.conref<!ep2.any>) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %37 = ep2.struct_access %36[0] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %38 = ep2.struct_access %36[0] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %39 = ep2.struct_access %36[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %40 = "ep2.struct_update"(%36, %39) <{index = 0 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i16) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %41 = ep2.struct_access %40[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %42 = "ep2.struct_update"(%40, %37) <{index = 1 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i16) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %43 = ep2.struct_access %42[2] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i32
    %44 = ep2.struct_access %arg1[0] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %45 = "ep2.struct_update"(%42, %44) <{index = 2 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i32) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %46 = ep2.struct_access %45[3] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i32
    %47 = ep2.struct_access %arg1[1] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %48 = "ep2.struct_update"(%45, %47) <{index = 3 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i32) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    "ep2.emit"(%8, %22) : (!ep2.buf, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>) -> ()
    %49 = "ep2.nop"() : () -> none
    "ep2.emit"(%8, %34) : (!ep2.buf, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    %50 = "ep2.nop"() : () -> none
    "ep2.emit"(%8, %48) : (!ep2.buf, !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>) -> ()
    %51 = "ep2.nop"() : () -> none
    %52 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %53 = "ep2.load"(%52) : (!ep2.conref<!ep2.any>) -> !ep2.any
    "ep2.emit"(%8, %53) : (!ep2.buf, !ep2.any) -> ()
    %54 = "ep2.nop"() : () -> none
    %55 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %56 = "ep2.init"(%55, %arg0, %8) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    ep2.return %56 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}
