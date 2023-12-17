// ep2c-opt --canonicalize --ep2-context-infer tests/output/example.mlir -o tests/output/example.opt.mlir

module {
  ep2.func private @__controller_DMA_READ_REQ() attributes {event = "DMA_READ_REQ", type = "controller"} {
    %0 = "ep2.constant"() <{value = 100 : i64}> : () -> i64
    %1 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %2 = "ep2.constant"() <{value = 10 : i64}> : () -> i64
    %3 = ep2.call @Queue(%0, %1, %2) : (i64, i64, i64) -> i64
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_DMA_READ_CMPL() attributes {event = "DMA_READ_CMPL", type = "controller"} {
    %0 = "ep2.constant"() <{value = 100 : i64}> : () -> i64
    %1 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %2 = "ep2.constant"() <{value = 10 : i64}> : () -> i64
    %3 = ep2.call @Queue(%0, %1, %2) : (i64, i64, i64) -> i64
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_READ_REQ(%arg0: !ep2.atom, %arg1: !ep2.context, %arg2: i64, %arg3: i32) attributes {event = "DMA_READ_REQ", extern = true, type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.buf
    %1 = "ep2.init"(%arg0, %arg1, %0) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"DMA_READ_CMPL" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    ep2.return %1 : !ep2.struct<"DMA_READ_CMPL" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_READ_CMPL_receive_desc(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "receive_desc", event = "DMA_READ_CMPL", type = "handler"} {
    %0 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"Desc_Hdr" : isEvent = false, elementTypes = i64, i32>
    %1 = "ep2.context_ref"(%arg0) <{name = "desc_addr"}> : (!ep2.context) -> !ep2.conref<i64>
    %2 = ep2.struct_access %0[0] : <"Desc_Hdr" : isEvent = false, elementTypes = i64, i32> -> i64
    "ep2.store"(%1, %2) : (!ep2.conref<i64>, i64) -> ()
    %3 = "ep2.context_ref"(%arg0) <{name = "desc_size"}> : (!ep2.context) -> !ep2.conref<i32>
    %4 = ep2.struct_access %0[1] : <"Desc_Hdr" : isEvent = false, elementTypes = i64, i32> -> i32
    "ep2.store"(%3, %4) : (!ep2.conref<i32>, i32) -> ()
    %5 = "ep2.constant"() <{value = "receive_payload_1"}> : () -> !ep2.atom
    %6 = ep2.struct_access %0[0] : <"Desc_Hdr" : isEvent = false, elementTypes = i64, i32> -> i64
    %7 = "ep2.constant"() <{value = 100 : i64}> : () -> i64
    %8 = "ep2.init"(%5, %arg0, %6, %7) : (!ep2.atom, !ep2.context, i64, i64) -> !ep2.struct<"DMA_READ_REQ" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, i64, i32>
    ep2.return %8 : !ep2.struct<"DMA_READ_REQ" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, i64, i32>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_READ_REQ_receive_payload_1(%arg0: !ep2.context, %arg1: i64, %arg2: i32) attributes {atom = "receive_payload_1", event = "DMA_READ_REQ", type = "handler"} {
    %0 = "ep2.context_ref"(%arg0) <{name = "desc_addr"}> : (!ep2.context) -> !ep2.conref<i64>
    %1 = "ep2.load"(%0) : (!ep2.conref<i64>) -> i64
    %2 = "ep2.add"(%1, %arg1) : (i64, i64) -> !ep2.any
    "ep2.store"(%0, %2) : (!ep2.conref<i64>, !ep2.any) -> ()
    "ep2.terminate"() : () -> ()
  }
}

