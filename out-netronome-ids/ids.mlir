module {
  ep2.func private @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_MSG_REASSEMBLE() attributes {event = "MSG_REASSEMBLE", type = "controller"} {
    %0 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 0>}> : () -> !ep2.port<true, false>
    %1 = "ep2.constant"() <{value = #ep2.port<"MSG_REASSEMBLE" : "msg_reassemble", 0>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%0, %1) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [32]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %2 = "ep2.nop"() : () -> none
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["i1cu2"], type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %1 = "ep2.init"() : () -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %2 = "ep2.init"() : () -> !ep2.struct<"udp_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    %3 = "ep2.init"() : () -> !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>
    %4 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %5 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %6 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"udp_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    %7 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>
    %8 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%8, %4) : (!ep2.conref<!ep2.any>, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> ()
    %9 = "ep2.nop"() : () -> none
    %10 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%10, %5) : (!ep2.conref<!ep2.any>, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    %11 = "ep2.nop"() : () -> none
    %12 = "ep2.constant"() <{value = "msg_reassemble"}> : () -> !ep2.atom
    %13 = "ep2.init"(%12, %arg0, %arg1, %7) : (!ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>) -> !ep2.struct<"MSG_REASSEMBLE" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>>
    ep2.return %13 : !ep2.struct<"MSG_REASSEMBLE" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_MSG_REASSEMBLE_msg_reassemble(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>) attributes {atom = "msg_reassemble", event = "MSG_REASSEMBLE", instances = ["i1cu3"], type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.table<i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, 16>
    %1 = "ep2.init"() : () -> !ep2.table<i16, !ep2.buf, 16>
    %2 = "ep2.init"() : () -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    %3 = "ep2.init"() : () -> !ep2.buf
    %4 = "ep2.init"() : () -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    %5 = "ep2.init"() : () -> !ep2.buf
    %6 = ep2.struct_access %arg2[0] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    %7 = "ep2.lookup"(%0, %6) : (!ep2.table<i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, 16>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    %8 = ep2.struct_access %arg2[0] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    %9 = "ep2.lookup"(%1, %8) : (!ep2.table<i16, !ep2.buf, 16>, i32) -> !ep2.buf
    %10 = ep2.struct_access %7[0] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %11 = "ep2.constant"() <{value = 0 : i64}> : () -> i64
    %12 = "ep2.bitcast"(%11) : (i64) -> i32
    %13 = "ep2.cmp"(%10, %12) <{predicate = 40 : i16}> : (i32, i32) -> i1
    %14 = scf.if %13 -> (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>) {
      %27 = ep2.struct_access %7[0] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
      %28 = ep2.struct_access %arg2[3] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
      %29 = "ep2.struct_update"(%7, %28) <{index = 0 : i64}> : (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
      scf.yield %29 : !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    } else {
      scf.yield %7 : !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    }
    %15 = ep2.struct_access %14[1] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %16 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %17 = "ep2.bitcast"(%16) : (i64) -> i32
    %18 = "ep2.add"(%15, %17) : (i32, i32) -> i32
    %19 = ep2.struct_access %arg2[1] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    %20 = "ep2.cmp"(%18, %19) <{predicate = 40 : i16}> : (i32, i32) -> i1
    %21 = scf.if %20 -> (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>) {
      %27 = ep2.struct_access %14[1] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
      %28 = ep2.struct_access %14[1] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
      %29 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
      %30 = "ep2.bitcast"(%29) : (i64) -> i32
      %31 = "ep2.add"(%28, %30) : (i32, i32) -> i32
      %32 = "ep2.struct_update"(%14, %31) <{index = 1 : i64}> : (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
      "ep2.emit"(%9, %arg1) : (!ep2.buf, !ep2.buf) -> ()
      %33 = "ep2.nop"() : () -> none
      %34 = ep2.struct_access %32[0] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
      %35 = ep2.struct_access %32[0] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
      %36 = ep2.struct_access %arg2[2] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
      %37 = "ep2.sub"(%35, %36) : (i32, i32) -> i32
      %38 = "ep2.struct_update"(%32, %37) <{index = 0 : i64}> : (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
      scf.yield %38 : !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    } else {
      scf.yield %14 : !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    }
    %22 = ep2.struct_access %21[0] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %23 = "ep2.constant"() <{value = 0 : i64}> : () -> i64
    %24 = "ep2.bitcast"(%23) : (i64) -> i32
    %25 = "ep2.cmp"(%22, %24) <{predicate = 62 : i16}> : (i32, i32) -> i1
    %26 = scf.if %25 -> (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>) {
      %27 = ep2.struct_access %arg2[0] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
      "ep2.update"(%0, %27, %21) : (!ep2.table<i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, 16>, i32, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>) -> ()
      %28 = "ep2.nop"() : () -> none
      %29 = ep2.struct_access %arg2[0] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
      "ep2.update"(%1, %29, %9) : (!ep2.table<i16, !ep2.buf, 16>, i32, !ep2.buf) -> ()
      %30 = "ep2.nop"() : () -> none
      scf.yield %4 : !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    } else {
      %27 = ep2.struct_access %4[0] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
      %28 = "ep2.constant"() <{value = 0 : i64}> : () -> i64
      %29 = "ep2.bitcast"(%28) : (i64) -> i32
      %30 = "ep2.struct_update"(%4, %29) <{index = 0 : i64}> : (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
      %31 = ep2.struct_access %arg2[0] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
      "ep2.update"(%0, %31, %30) : (!ep2.table<i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, 16>, i32, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>) -> ()
      %32 = "ep2.nop"() : () -> none
      %33 = ep2.struct_access %arg2[0] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
      "ep2.update"(%1, %33, %5) : (!ep2.table<i16, !ep2.buf, 16>, i32, !ep2.buf) -> ()
      %34 = "ep2.nop"() : () -> none
      %35 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
      %36 = "ep2.init"(%35, %arg0, %9) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
      ep2.return %36 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
      scf.yield %30 : !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    }
    "ep2.terminate"() : () -> ()
  }
}
