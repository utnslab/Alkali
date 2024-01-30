module {
  ep2.func private @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_MSG_REASSEMBLE() attributes {event = "MSG_REASSEMBLE", type = "controller"} {
    %0 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 0>}> : () -> !ep2.port<true, false>
    %1 = "ep2.constant"() <{value = #ep2.port<"MSG_REASSEMBLE" : "msg_reassemble", 0>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%0, %1) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [32]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["i1cu2"], type = "handler"} {
    %0 = "ep2.constant"() <{value = "msg_reassemble"}> : () -> !ep2.atom
    %1 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %2 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32>
    %3 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"udp_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    %4 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>
    %5 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32>>
    "ep2.store"(%5, %2) : (!ep2.conref<!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32>>, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32>) -> ()
    %6 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>
    "ep2.store"(%6, %1) : (!ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> ()
    %7 = "ep2.init"(%0, %arg0, %arg1, %4) : (!ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>) -> !ep2.struct<"MSG_REASSEMBLE" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>>
    ep2.return %7 : !ep2.struct<"MSG_REASSEMBLE" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_MSG_REASSEMBLE_msg_reassemble(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>) attributes {atom = "msg_reassemble", event = "MSG_REASSEMBLE", instances = ["i1cu3"], type = "handler"} {
    %0 = "ep2.constant"() <{value = 0 : i32}> : () -> i32
    %1 = "ep2.constant"() <{value = 1 : i32}> : () -> i32
    %2 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %3 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32>>
    %4 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>
    %5 = "ep2.load"(%4) : (!ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %6 = "ep2.load"(%3) : (!ep2.conref<!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32>>) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32>
    %7 = "ep2.init"() : () -> !ep2.table<i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, 16>
    %8 = "ep2.init"() : () -> !ep2.table<i16, !ep2.buf, 16>
    %9 = "ep2.init"() : () -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    %10 = ep2.struct_access %arg2[0] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    %11 = "ep2.bitcast"(%10) : (i32) -> i16
    %12 = "ep2.lookup"(%7, %11) : (!ep2.table<i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, 16>, i16) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    %13 = "ep2.lookup"(%8, %11) : (!ep2.table<i16, !ep2.buf, 16>, i16) -> !ep2.buf
    %14 = ep2.struct_access %12[0] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %15 = "ep2.cmp"(%14, %0) <{predicate = 40 : i16}> : (i32, i32) -> i1
    cf.cond_br %15, ^bb1, ^bb2(%12 : !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>)
  ^bb1:  // pred: ^bb0
    %16 = ep2.struct_access %arg2[3] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    %17 = "ep2.struct_update"(%12, %16) <{index = 0 : i64}> : (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    "ep2.emit"(%13, %5) : (!ep2.buf, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> ()
    "ep2.emit"(%13, %6) : (!ep2.buf, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32>) -> ()
    cf.br ^bb2(%17 : !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>)
  ^bb2(%18: !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>):  // 2 preds: ^bb0, ^bb1
    %19 = ep2.struct_access %18[1] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %20 = "ep2.add"(%19, %1) : (i32, i32) -> i32
    %21 = "ep2.struct_update"(%18, %20) <{index = 1 : i64}> : (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    "ep2.emit"(%13, %arg1) : (!ep2.buf, !ep2.buf) -> ()
    %22 = ep2.struct_access %21[0] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %23 = ep2.struct_access %arg2[2] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    %24 = "ep2.sub"(%22, %23) : (i32, i32) -> i32
    %25 = "ep2.struct_update"(%21, %24) <{index = 0 : i64}> : (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    %26 = ep2.struct_access %25[0] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %27 = "ep2.cmp"(%26, %0) <{predicate = 62 : i16}> : (i32, i32) -> i1
    cf.cond_br %27, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    "ep2.update"(%7, %11, %25) : (!ep2.table<i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, 16>, i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>) -> ()
    "ep2.update"(%8, %11, %13) : (!ep2.table<i16, !ep2.buf, 16>, i16, !ep2.buf) -> ()
    cf.br ^bb5
  ^bb4:  // pred: ^bb2
    %28 = "ep2.init"() : () -> !ep2.buf
    %29 = "ep2.struct_update"(%9, %0) <{index = 0 : i64}> : (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    "ep2.update"(%7, %11, %29) : (!ep2.table<i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, 16>, i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>) -> ()
    "ep2.update"(%8, %11, %28) : (!ep2.table<i16, !ep2.buf, 16>, i16, !ep2.buf) -> ()
    %30 = "ep2.init"(%2, %arg0, %13) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    ep2.return %30 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    cf.br ^bb5
  ^bb5:  // 2 preds: ^bb3, ^bb4
    "ep2.terminate"() : () -> ()
  }
}

