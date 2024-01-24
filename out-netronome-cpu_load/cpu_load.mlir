module {
  ep2.func private @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_SEND_dma_send(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16>) attributes {atom = "dma_send", event = "DMA_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  %0 = "ep2.global"() <{name = "service_load"}> {scope = #ep2<scope<"LOAD_TABLE_ADD:load_table_add"> [""]>} : () -> !ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["i1cu2"], type = "handler"} {
    %1 = "ep2.init"() : () -> i16
    %2 = "ep2.constant"() <{value = 0 : i64}> : () -> i64
    %3 = "ep2.bitcast"(%2) : (i64) -> i16
    %4 = "ep2.init"() : () -> !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16>
    %5 = "ep2.init"() : () -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    %6 = "ep2.init"() : () -> i32
    %7 = "ep2.init"() : () -> i32
    %8 = "ep2.init"() : () -> i32
    %9 = "ep2.init"() : () -> i16
    %10 = "ep2.init"() : () -> i16
    %11 = "ep2.init"() : () -> i16
    %12 = "ep2.init"() : () -> i32
    %13 = "ep2.init"() : () -> i32
    %14 = "ep2.init"() : () -> i32
    %15 = "ep2.global_import"() <{name = "service_load"}> : () -> !ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>
    %16 = "ep2.lookup"(%15, %3) : (!ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>, i16) -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    %17 = ep2.struct_access %16[0] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
    %18 = ep2.struct_access %16[1] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
    %19 = "ep2.cmp"(%17, %18) <{predicate = 60 : i16}> : (i16, i16) -> i1
    %20:2 = scf.if %19 -> (i32, i16) {
      %37 = ep2.struct_access %16[0] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
      %38 = "ep2.bitcast"(%37) : (i16) -> i32
      %39 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
      %40 = "ep2.bitcast"(%39) : (i64) -> i16
      scf.yield %38, %40 : i32, i16
    } else {
      %37 = ep2.struct_access %16[1] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
      %38 = "ep2.bitcast"(%37) : (i16) -> i32
      %39 = "ep2.constant"() <{value = 2 : i64}> : () -> i64
      %40 = "ep2.bitcast"(%39) : (i64) -> i16
      scf.yield %38, %40 : i32, i16
    }
    %21 = ep2.struct_access %16[2] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
    %22 = ep2.struct_access %16[3] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
    %23 = "ep2.cmp"(%21, %22) <{predicate = 60 : i16}> : (i16, i16) -> i1
    %24:2 = scf.if %23 -> (i32, i16) {
      %37 = ep2.struct_access %16[3] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
      %38 = "ep2.bitcast"(%37) : (i16) -> i32
      %39 = "ep2.constant"() <{value = 3 : i64}> : () -> i64
      %40 = "ep2.bitcast"(%39) : (i64) -> i16
      scf.yield %38, %40 : i32, i16
    } else {
      %37 = ep2.struct_access %16[3] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
      %38 = "ep2.bitcast"(%37) : (i16) -> i32
      %39 = "ep2.constant"() <{value = 4 : i64}> : () -> i64
      %40 = "ep2.bitcast"(%39) : (i64) -> i16
      scf.yield %38, %40 : i32, i16
    }
    %25 = "ep2.cmp"(%20#0, %24#0) <{predicate = 60 : i16}> : (i32, i32) -> i1
    %26:2 = scf.if %25 -> (i32, i16) {
      scf.yield %20#0, %20#1 : i32, i16
    } else {
      scf.yield %24#0, %24#1 : i32, i16
    }
    %27 = ep2.struct_access %4[0] : <"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16> -> i16
    %28 = "ep2.struct_update"(%4, %26#1) <{index = 0 : i64}> : (!ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16>, i16) -> !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16>
    %29 = ep2.struct_access %28[1] : <"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16> -> i16
    %30 = "ep2.struct_update"(%28, %3) <{index = 1 : i64}> : (!ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16>, i16) -> !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16>
    %31 = "ep2.constant"() <{value = 16 : i64}> : () -> i64
    %32 = "ep2.bitcast"(%31) : (i64) -> i32
    %33 = "ep2.cmp"(%26#0, %32) <{predicate = 62 : i16}> : (i32, i32) -> i1
    %34 = scf.if %33 -> (!ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16>) {
      %37 = ep2.struct_access %30[2] : <"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16> -> i16
      %38 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
      %39 = "ep2.bitcast"(%38) : (i64) -> i16
      %40 = "ep2.struct_update"(%30, %39) <{index = 2 : i64}> : (!ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16>, i16) -> !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16>
      scf.yield %40 : !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16>
    } else {
      scf.yield %30 : !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16, i16>
    }
    %35 = "ep2.constant"() <{value = "load_table_add"}> : () -> !ep2.atom
    %36 = "ep2.init"(%35, %arg0, %3, %26#1) : (!ep2.atom, !ep2.context, i16, i16) -> !ep2.struct<"LOAD_TABLE_ADD" : isEvent = true, elementTypes = !ep2.context, i16, i16>
    ep2.return %36 : !ep2.struct<"LOAD_TABLE_ADD" : isEvent = true, elementTypes = !ep2.context, i16, i16>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_LOAD_TABLE_ADD() attributes {event = "LOAD_TABLE_ADD", type = "controller"} {
    %1 = "ep2.constant"() <{value = #ep2.port<"NET_RECV" : "process_packet", 0>}> : () -> !ep2.port<true, false>
    %2 = "ep2.constant"() <{value = #ep2.port<"LOAD_TABLE_ADD" : "load_table_add", 0>}> : () -> !ep2.port<false, true>
    "ep2.connect"(%1, %2) <{method = "Queue", operandSegmentSizes = array<i32: 1, 1>, parameters = [32]}> : (!ep2.port<true, false>, !ep2.port<false, true>) -> ()
    %3 = "ep2.nop"() : () -> none
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_LOAD_TABLE_ADD_load_table_add(%arg0: !ep2.context, %arg1: i16, %arg2: i16) attributes {atom = "load_table_add", event = "LOAD_TABLE_ADD", instances = ["i1cu3"], scope = #ep2<scope<"LOAD_TABLE_ADD:load_table_add"> [""]>, type = "handler"} {
    %1 = "ep2.init"() : () -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    %2 = "ep2.global_import"() <{name = "service_load"}> : () -> !ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>
    %3 = "ep2.lookup"(%2, %arg1) : (!ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>, i16) -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    %4 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %5 = "ep2.bitcast"(%4) : (i64) -> i16
    %6 = "ep2.cmp"(%arg2, %5) <{predicate = 40 : i16}> : (i16, i16) -> i1
    %7 = scf.if %6 -> (!ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>) {
      %22 = ep2.struct_access %3[0] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
      %23 = ep2.struct_access %3[0] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
      %24 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
      %25 = "ep2.bitcast"(%24) : (i64) -> i16
      %26 = "ep2.add"(%23, %25) : (i16, i16) -> i16
      %27 = "ep2.struct_update"(%3, %26) <{index = 0 : i64}> : (!ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, i16) -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
      scf.yield %27 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    } else {
      scf.yield %3 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    }
    %8 = "ep2.constant"() <{value = 2 : i64}> : () -> i64
    %9 = "ep2.bitcast"(%8) : (i64) -> i16
    %10 = "ep2.cmp"(%arg2, %9) <{predicate = 40 : i16}> : (i16, i16) -> i1
    %11 = scf.if %10 -> (!ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>) {
      %22 = ep2.struct_access %7[1] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
      %23 = ep2.struct_access %7[1] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
      %24 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
      %25 = "ep2.bitcast"(%24) : (i64) -> i16
      %26 = "ep2.add"(%23, %25) : (i16, i16) -> i16
      %27 = "ep2.struct_update"(%7, %26) <{index = 1 : i64}> : (!ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, i16) -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
      scf.yield %27 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    } else {
      scf.yield %7 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    }
    %12 = "ep2.constant"() <{value = 3 : i64}> : () -> i64
    %13 = "ep2.bitcast"(%12) : (i64) -> i16
    %14 = "ep2.cmp"(%arg2, %13) <{predicate = 40 : i16}> : (i16, i16) -> i1
    %15 = scf.if %14 -> (!ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>) {
      %22 = ep2.struct_access %11[2] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
      %23 = ep2.struct_access %11[2] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
      %24 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
      %25 = "ep2.bitcast"(%24) : (i64) -> i16
      %26 = "ep2.add"(%23, %25) : (i16, i16) -> i16
      %27 = "ep2.struct_update"(%11, %26) <{index = 2 : i64}> : (!ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, i16) -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
      scf.yield %27 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    } else {
      scf.yield %11 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    }
    %16 = "ep2.constant"() <{value = 4 : i64}> : () -> i64
    %17 = "ep2.bitcast"(%16) : (i64) -> i16
    %18 = "ep2.cmp"(%arg2, %17) <{predicate = 40 : i16}> : (i16, i16) -> i1
    %19 = scf.if %18 -> (!ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>) {
      %22 = ep2.struct_access %15[3] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
      %23 = ep2.struct_access %15[3] : <"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16> -> i16
      %24 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
      %25 = "ep2.bitcast"(%24) : (i64) -> i16
      %26 = "ep2.add"(%23, %25) : (i16, i16) -> i16
      %27 = "ep2.struct_update"(%15, %26) <{index = 3 : i64}> : (!ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, i16) -> !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
      scf.yield %27 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    } else {
      scf.yield %15 : !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>
    }
    %20 = "ep2.global_import"() <{name = "service_load"}> : () -> !ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>
    "ep2.update"(%20, %arg1, %19) : (!ep2.table<i32, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>, 16>, i16, !ep2.struct<"coremap_t" : isEvent = false, elementTypes = i16, i16, i16, i16>) -> ()
    %21 = "ep2.nop"() : () -> none
    "ep2.terminate"() : () -> ()
  }
}
