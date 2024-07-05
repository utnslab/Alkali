module {
  ep2.func private @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_SEND_dma_send(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16>) attributes {atom = "dma_send", event = "DMA_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  %0 = "ep2.global"() <{name = "service_load"}> {scope = #ep2<scope<"LOAD_TABLE_ADD:load_table_add"> [""]>} : () -> !ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["i1cu2"], type = "handler"} {
    %1 = "ep2.constant"() <{value = 0 : i32}> : () -> i32
    %2 = "ep2.constant"() <{value = 0 : i16}> : () -> i16
    %3 = "ep2.constant"() <{value = 1 : i16}> : () -> i16
    %4 = "ep2.constant"() <{value = 2 : i16}> : () -> i16
    %5 = "ep2.constant"() <{value = 3 : i16}> : () -> i16
    %6 = "ep2.constant"() <{value = 4 : i16}> : () -> i16
    %7 = "ep2.constant"() <{value = "load_table_add"}> : () -> !ep2.atom
    %8 = "ep2.global_import"() <{name = "service_load"}> : () -> !ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>
    %9 = "ep2.lookup"(%8, %1) : (!ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>, i32) -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    %10 = ep2.struct_access %9[0] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
    %11 = ep2.struct_access %9[1] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
    %12 = "ep2.cmp"(%10, %11) <{predicate = 60 : i16}> : (i16, i16) -> i1
    %13 = arith.select %12, %3, %4 : i16
    %14 = arith.select %12, %10, %11 : i16
    %15 = "ep2.bitcast"(%14) : (i16) -> i32
    %16 = ep2.struct_access %9[2] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
    %17 = ep2.struct_access %9[3] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
    %18 = "ep2.cmp"(%16, %17) <{predicate = 60 : i16}> : (i16, i16) -> i1
    %19 = arith.select %18, %5, %6 : i16
    %20 = "ep2.bitcast"(%17) : (i16) -> i32
    %21 = "ep2.cmp"(%15, %20) <{predicate = 60 : i16}> : (i32, i32) -> i1
    %22 = arith.select %21, %13, %19 : i16
    %23 = "ep2.init"(%7, %arg0, %2, %22) : (!ep2.atom, !ep2.context, i16, i16) -> !ep2.struct<"LOAD_TABLE_ADD" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, i16, i16>
    ep2.return %23 : !ep2.struct<"LOAD_TABLE_ADD" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, i16, i16>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_LOAD_TABLE_ADD() attributes {event = "LOAD_TABLE_ADD", type = "controller"} {
    %1 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 0>}> : () -> !ep2.port<true, false>
    %2 = "ep2.constant"() <{value = #ep2.port<"LOAD_TABLE_ADD" : "load_table_add", 0>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%1, %2) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [32]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_LOAD_TABLE_ADD_load_table_add(%arg0: !ep2.context, %arg1: i16, %arg2: i16) attributes {atom = "load_table_add", event = "LOAD_TABLE_ADD", instances = ["i1cu3"], scope = #ep2<scope<"LOAD_TABLE_ADD:load_table_add"> [""]>, type = "handler"} {
    %1 = "ep2.constant"() <{value = 1 : i16}> : () -> i16
    %2 = "ep2.constant"() <{value = 2 : i16}> : () -> i16
    %3 = "ep2.constant"() <{value = 3 : i16}> : () -> i16
    %4 = "ep2.constant"() <{value = 4 : i16}> : () -> i16
    %5 = "ep2.global_import"() <{name = "service_load"}> : () -> !ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>
    %6 = "ep2.bitcast"(%arg1) : (i16) -> i32
    %7 = "ep2.lookup"(%5, %6) : (!ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>, i32) -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    %8 = "ep2.cmp"(%arg2, %1) <{predicate = 40 : i16}> : (i16, i16) -> i1
    cf.cond_br %8, ^bb1, ^bb2(%7 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>)
  ^bb1:  // pred: ^bb0
    %9 = ep2.struct_access %7[0] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
    %10 = "ep2.add"(%9, %1) : (i16, i16) -> i16
    %11 = "ep2.struct_update"(%7, %10) <{index = 0 : i64}> : (!ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, i16) -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    cf.br ^bb2(%11 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>)
  ^bb2(%12: !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>):  // 2 preds: ^bb0, ^bb1
    %13 = "ep2.cmp"(%arg2, %2) <{predicate = 40 : i16}> : (i16, i16) -> i1
    cf.cond_br %13, ^bb3, ^bb4(%12 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>)
  ^bb3:  // pred: ^bb2
    %14 = ep2.struct_access %12[1] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
    %15 = "ep2.add"(%14, %1) : (i16, i16) -> i16
    %16 = "ep2.struct_update"(%12, %15) <{index = 1 : i64}> : (!ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, i16) -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    cf.br ^bb4(%16 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>)
  ^bb4(%17: !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>):  // 2 preds: ^bb2, ^bb3
    %18 = "ep2.cmp"(%arg2, %3) <{predicate = 40 : i16}> : (i16, i16) -> i1
    cf.cond_br %18, ^bb5, ^bb6(%17 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>)
  ^bb5:  // pred: ^bb4
    %19 = ep2.struct_access %17[2] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
    %20 = "ep2.add"(%19, %1) : (i16, i16) -> i16
    %21 = "ep2.struct_update"(%17, %20) <{index = 2 : i64}> : (!ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, i16) -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    cf.br ^bb6(%21 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>)
  ^bb6(%22: !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>):  // 2 preds: ^bb4, ^bb5
    %23 = "ep2.cmp"(%arg2, %4) <{predicate = 40 : i16}> : (i16, i16) -> i1
    cf.cond_br %23, ^bb7, ^bb8(%22 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>)
  ^bb7:  // pred: ^bb6
    %24 = ep2.struct_access %22[3] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
    %25 = "ep2.add"(%24, %1) : (i16, i16) -> i16
    %26 = "ep2.struct_update"(%22, %25) <{index = 3 : i64}> : (!ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, i16) -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    cf.br ^bb8(%26 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>)
  ^bb8(%27: !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>):  // 2 preds: ^bb6, ^bb7
    "ep2.update"(%5, %6, %27) : (!ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>, i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>) -> ()
    "ep2.terminate"() : () -> ()
  }
}

