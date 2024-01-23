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
    %2 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>
    %3 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"udp_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    %4 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>
    %5 = "ep2.init"(%0, %arg0, %arg1, %4) : (!ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>) -> !ep2.struct<"MSG_REASSEMBLE" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>>
    ep2.return %5 : !ep2.struct<"MSG_REASSEMBLE" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf, !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_MSG_REASSEMBLE_msg_reassemble(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32>) attributes {atom = "msg_reassemble", event = "MSG_REASSEMBLE", instances = ["i1cu3"], type = "handler"} {
    %0 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %1 = "ep2.constant"() <{value = 1 : i32}> : () -> i32
    %2 = "ep2.constant"() <{value = 0 : i32}> : () -> i32
    %3 = "ep2.init"() : () -> !ep2.table<i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, 16>
    %4 = "ep2.init"() : () -> !ep2.table<i16, !ep2.buf, 16>
    %5 = "ep2.init"() : () -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    %6 = "ep2.init"() : () -> !ep2.buf
    %7 = ep2.struct_access %arg2[0] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    %8 = "ep2.bitcast"(%7) : (i32) -> i16
    %9 = "ep2.lookup"(%3, %8) : (!ep2.table<i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, 16>, i16) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    %10 = "ep2.lookup"(%4, %8) : (!ep2.table<i16, !ep2.buf, 16>, i16) -> !ep2.buf
    %11 = ep2.struct_access %9[0] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %12 = "ep2.cmp"(%11, %2) <{predicate = 40 : i16}> : (i32, i32) -> i1
    cf.cond_br %12, ^bb1, ^bb2(%9 : !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>)
  ^bb1:  // pred: ^bb0
    %13 = ep2.struct_access %arg2[3] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    %14 = "ep2.struct_update"(%9, %13) <{index = 0 : i64}> : (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    cf.br ^bb2(%14 : !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>)
  ^bb2(%15: !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>):  // 2 preds: ^bb0, ^bb1
    %16 = ep2.struct_access %15[1] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %17 = "ep2.add"(%16, %1) : (i32, i32) -> i32
    %18 = ep2.struct_access %arg2[1] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    %19 = "ep2.cmp"(%17, %18) <{predicate = 40 : i16}> : (i32, i32) -> i1
    cf.cond_br %19, ^bb3, ^bb4(%15 : !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>)
  ^bb3:  // pred: ^bb2
    %20 = "ep2.struct_update"(%15, %17) <{index = 1 : i64}> : (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    "ep2.emit"(%10, %arg1) : (!ep2.buf, !ep2.buf) -> ()
    %21 = ep2.struct_access %20[0] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %22 = ep2.struct_access %arg2[2] : <"rpc_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    %23 = "ep2.sub"(%21, %22) : (i32, i32) -> i32
    %24 = "ep2.struct_update"(%20, %23) <{index = 0 : i64}> : (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    cf.br ^bb4(%24 : !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>)
  ^bb4(%25: !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>):  // 2 preds: ^bb2, ^bb3
    %26 = ep2.struct_access %25[0] : <"agg_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %27 = "ep2.cmp"(%26, %2) <{predicate = 62 : i16}> : (i32, i32) -> i1
    cf.cond_br %27, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    "ep2.update"(%3, %8, %25) : (!ep2.table<i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, 16>, i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>) -> ()
    "ep2.update"(%4, %8, %10) : (!ep2.table<i16, !ep2.buf, 16>, i16, !ep2.buf) -> ()
    cf.br ^bb7
  ^bb6:  // pred: ^bb4
    %28 = "ep2.struct_update"(%5, %2) <{index = 0 : i64}> : (!ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, i32) -> !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>
    "ep2.update"(%3, %8, %28) : (!ep2.table<i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>, 16>, i16, !ep2.struct<"agg_t" : isEvent = false, elementTypes = i32, i32>) -> ()
    "ep2.update"(%4, %8, %6) : (!ep2.table<i16, !ep2.buf, 16>, i16, !ep2.buf) -> ()
    %29 = "ep2.init"(%0, %arg0, %10) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    ep2.return %29 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    cf.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    "ep2.terminate"() : () -> ()
  }
}

