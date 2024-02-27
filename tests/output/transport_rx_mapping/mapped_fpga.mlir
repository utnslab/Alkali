
// ./build/bin/ep2c tests/experiments/transport_rx.all.ep2 -o tmp.mlir
// ./build/bin/ep2c-opt --ep2-mapping="arch-spec-file=tests/specs/fpga.json cost-model=fpga" --ep2-global-to-partition tmp.mlir -o mapped.mlir

module {
  ep2.func private @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_WRITE_REQ_dma_write(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) attributes {atom = "dma_write", event = "DMA_WRITE_REQ", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["cu2"], type = "handler"} {
    %0 = "ep2.constant"() <{value = 14 : i16}> : () -> i16
    %1 = "ep2.constant"() <{value = "OoO_detection"}> : () -> !ep2.atom
    %2 = "ep2.init"() : () -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %3 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %4 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %5 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %6 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%6, %3) : (!ep2.conref<!ep2.any>, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>) -> ()
    %7 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%7, %4) : (!ep2.conref<!ep2.any>, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    %8 = "ep2.context_ref"(%arg0) <{name = "tcp_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%8, %5) : (!ep2.conref<!ep2.any>, !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>) -> ()
    %9 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%9, %arg1) : (!ep2.conref<!ep2.any>, !ep2.buf) -> ()
    %10 = ep2.struct_access %4[1] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i16
    %11 = "ep2.add"(%10, %0) : (i16, i16) -> i16
    %12 = "ep2.struct_update"(%2, %11) <{index = 1 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i16) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %13 = ep2.struct_access %5[2] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i32
    %14 = "ep2.struct_update"(%12, %13) <{index = 2 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %15 = ep2.struct_access %5[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %16 = "ep2.struct_update"(%14, %15) <{index = 0 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i16) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %17 = "ep2.context_ref"(%arg0) <{name = "flow_id"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %18 = ep2.struct_access %5[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    "ep2.store"(%17, %18) : (!ep2.conref<!ep2.any>, i16) -> ()
    %19 = "ep2.init"(%1, %arg0, %16) : (!ep2.atom, !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>) -> !ep2.struct<"OoO_DETECT" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    ep2.return %19 : !ep2.struct<"OoO_DETECT" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_ACK_GEN() attributes {event = "ACK_GEN", type = "controller"} {
    %0 = "ep2.constant"() <{value = 256 : i64}> : () -> i64
    %1 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %2 = ep2.call @Queue(%0, %1, %1) : (i64, i64, i64) -> i64
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_DMA_WRITE_REQ() attributes {event = "DMA_WRITE_REQ", extern = true, type = "controller"} {
    %0 = "ep2.constant"() <{value = 256 : i64}> : () -> i64
    %1 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %2 = ep2.call @Queue(%0, %1, %1) : (i64, i64, i64) -> i64
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_OoO_DETECT() attributes {event = "OoO_DETECT", scope = #ep2<scope<"OoO_DETECT:OoO_detection"> ["flow_id"]>, type = "controller"} {
    %0 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", -1>}> : () -> !ep2.port<true, false>
    %1 = "ep2.constant"() <{value = #ep2.port<"OoO_DETECT" : "OoO_detection", -1>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%0, %1) <{method = "PartitionBy", operandSegmentSizes = array<i32: 1, 1>, parameters = []}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_OoO_DETECT_OoO_detection(%arg0: !ep2.context, %arg1: !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32>) attributes {atom = "OoO_detection", event = "OoO_DETECT", instances = ["cu3", "cu4", "cu5"], instances_flow_table = ["cu3", "cu4", "cu5"], scope = #ep2<scope<"OoO_DETECT:OoO_detection"> ["flow_id"]>, type = "handler"} {
    %0 = "ep2.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "ep2.constant"() <{value = "ack_gen"}> : () -> !ep2.atom
    %2 = "ep2.constant"() <{value = "dma_write"}> : () -> !ep2.atom
    %3 = ep2.struct_access %arg1[0] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %4 = "ep2.init"() : () -> !ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>
    %5 = "ep2.lookup"(%4, %3) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    %6 = "ep2.init"() : () -> i32
    %7 = "ep2.init"() : () -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
    %8 = "ep2.init"() : () -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %9 = ep2.struct_access %5[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %10 = ep2.struct_access %arg1[2] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %11 = "ep2.sub"(%9, %10) : (i32, i32) -> i32
    %12 = ep2.struct_access %arg1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %13 = "ep2.sub"(%12, %11) : (i32, i32) -> i32
    %14 = ep2.struct_access %5[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %15 = "ep2.cmp"(%13, %14) <{predicate = 60 : i16}> : (i32, i32) -> i1
    %16 = scf.if %15 -> (i32) {
      scf.yield %0 : i32
    } else {
      %31 = ep2.struct_access %arg1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
      %32 = "ep2.sub"(%31, %11) : (i32, i32) -> i32
      %33 = ep2.struct_access %5[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %34 = "ep2.sub"(%32, %33) : (i32, i32) -> i32
      scf.yield %34 : i32
    }
    %17 = ep2.struct_access %arg1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %18 = "ep2.add"(%11, %16) : (i32, i32) -> i32
    %19 = "ep2.sub"(%17, %18) : (i32, i32) -> i32
    %20 = "ep2.cmp"(%19, %0) <{predicate = 62 : i16}> : (i32, i32) -> i1
    %21:3 = scf.if %20 -> (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>) {
      %31 = ep2.struct_access %5[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %32 = ep2.struct_access %5[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %33 = "ep2.sub"(%32, %19) : (i32, i32) -> i32
      %34 = "ep2.struct_update"(%5, %33) <{index = 4 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
      %35 = ep2.struct_access %34[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %36 = "ep2.add"(%35, %19) : (i32, i32) -> i32
      %37 = "ep2.struct_update"(%34, %36) <{index = 5 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
      %38 = ep2.struct_access %37[6] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
      %39 = "ep2.add"(%38, %19) : (i32, i32) -> i32
      %40 = "ep2.struct_update"(%37, %39) <{index = 6 : i64}> : (!ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
      %41 = "ep2.struct_update"(%7, %31) <{index = 0 : i64}> : (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
      %42 = "ep2.struct_update"(%41, %19) <{index = 1 : i64}> : (!ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>
      %43 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
      %44 = "ep2.load"(%43) : (!ep2.conref<!ep2.any>) -> !ep2.any
      %45 = "ep2.init"(%2, %arg0, %44, %42) : (!ep2.atom, !ep2.context, !ep2.any, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) -> !ep2.struct<"DMA_WRITE_REQ" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>>
      ep2.return %45 : !ep2.struct<"DMA_WRITE_REQ" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>>
      scf.yield %42, %31, %40 : !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    } else {
      scf.yield %7, %6, %5 : !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>
    }
    %22 = ep2.struct_access %arg1[0] : <"pkt_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %23 = "ep2.init"() : () -> !ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>
    "ep2.update"(%23, %22, %21#2) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>, 16>, i32, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32>) -> ()
    %24 = ep2.struct_access %21#2[0] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %25 = "ep2.struct_update"(%8, %24) <{index = 0 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %26 = ep2.struct_access %21#2[5] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %27 = "ep2.struct_update"(%25, %26) <{index = 1 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %28 = ep2.struct_access %21#2[4] : <"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32> -> i32
    %29 = "ep2.struct_update"(%27, %28) <{index = 2 : i64}> : (!ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>
    %30 = "ep2.init"(%1, %arg0, %29) : (!ep2.atom, !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>) -> !ep2.struct<"ACK_GEN" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    ep2.return %30 : !ep2.struct<"ACK_GEN" : isEvent = true, elementTypes = !ep2.context, !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_ACK_GEN_ack_gen(%arg0: !ep2.context, %arg1: !ep2.struct<"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32>) attributes {atom = "ack_gen", event = "ACK_GEN", instances = ["cu6"], type = "handler"} {
    %0 = "ep2.constant"() <{value = 64 : i16}> : () -> i16
    %1 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %2 = "ep2.init"() : () -> !ep2.buf
    %3 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %4 = "ep2.load"(%3) : (!ep2.conref<!ep2.any>) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %5 = ep2.struct_access %4[0] : <"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16> -> i32
    %6 = ep2.struct_access %4[1] : <"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16> -> i16
    %7 = ep2.struct_access %4[2] : <"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16> -> i32
    %8 = "ep2.struct_update"(%4, %7) <{index = 0 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>, i32) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %9 = ep2.struct_access %8[3] : <"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16> -> i16
    %10 = "ep2.struct_update"(%8, %9) <{index = 1 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>, i16) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %11 = "ep2.struct_update"(%10, %5) <{index = 2 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>, i32) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %12 = "ep2.struct_update"(%11, %6) <{index = 3 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>, i16) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>
    %13 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %14 = "ep2.load"(%13) : (!ep2.conref<!ep2.any>) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %15 = ep2.struct_access %14[6] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %16 = ep2.struct_access %14[7] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %17 = "ep2.struct_update"(%14, %16) <{index = 6 : i64}> : (!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>, i32) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %18 = "ep2.struct_update"(%17, %15) <{index = 7 : i64}> : (!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>, i32) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %19 = "ep2.struct_update"(%18, %0) <{index = 1 : i64}> : (!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>, i16) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %20 = "ep2.context_ref"(%arg0) <{name = "tcp_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %21 = "ep2.load"(%20) : (!ep2.conref<!ep2.any>) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %22 = ep2.struct_access %21[0] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %23 = ep2.struct_access %21[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %24 = "ep2.struct_update"(%21, %23) <{index = 0 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i16) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %25 = "ep2.struct_update"(%24, %22) <{index = 1 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i16) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %26 = ep2.struct_access %arg1[0] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %27 = "ep2.struct_update"(%25, %26) <{index = 2 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i32) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    %28 = ep2.struct_access %arg1[1] : <"ack_info_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %29 = "ep2.struct_update"(%27, %28) <{index = 3 : i64}> : (!ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>, i32) -> !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>
    "ep2.emit"(%2, %12) : (!ep2.buf, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i32, i16, i32, i16, i16>) -> ()
    "ep2.emit"(%2, %19) : (!ep2.buf, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    "ep2.emit"(%2, %29) : (!ep2.buf, !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>) -> ()
    %30 = "ep2.context_ref"(%arg0) <{name = "packet"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %31 = "ep2.load"(%30) : (!ep2.conref<!ep2.any>) -> !ep2.any
    "ep2.emit"(%2, %31) : (!ep2.buf, !ep2.any) -> ()
    %32 = "ep2.init"(%1, %arg0, %2) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    ep2.return %32 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}

