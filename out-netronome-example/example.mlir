module {
  ep2.func private @__controller_DMA_RECV_CMPL() attributes {event = "DMA_RECV_CMPL", extern = true, type = "controller"} {
    %0 = "ep2.constant"() <{value = 256 : i64}> : () -> i64
    %1 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %2 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %3 = ep2.call @Queue(%0, %1, %2) : (i64, i64, i64) -> i64
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_USER_EVENT1() attributes {event = "USER_EVENT1", type = "controller"} {
    %0 = "ep2.constant"() <{value = 256 : i64}> : () -> i64
    %1 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %2 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %3 = ep2.call @Queue(%0, %1, %2) : (i64, i64, i64) -> i64
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_RECV_CMPL_receive_desc(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "receive_desc", event = "DMA_RECV_CMPL", type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.struct<"recv_desc_t" : isEvent = false, elementTypes = i32, i32, i32, i32>
    %1 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"recv_desc_t" : isEvent = false, elementTypes = i32, i32, i32, i32>
    %2 = "ep2.context_ref"(%arg0) <{name = "flow_id"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %3 = ep2.struct_access %1[0] : <"recv_desc_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    "ep2.store"(%2, %3) : (!ep2.conref<!ep2.any>, i32) -> ()
    %4 = "ep2.nop"() : () -> none
    %5 = "ep2.context_ref"(%arg0) <{name = "bump_seq"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %6 = ep2.struct_access %1[1] : <"recv_desc_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    "ep2.store"(%5, %6) : (!ep2.conref<!ep2.any>, i32) -> ()
    %7 = "ep2.nop"() : () -> none
    %8 = "ep2.context_ref"(%arg0) <{name = "flags"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %9 = ep2.struct_access %1[2] : <"recv_desc_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    "ep2.store"(%8, %9) : (!ep2.conref<!ep2.any>, i32) -> ()
    %10 = "ep2.nop"() : () -> none
    %11 = "ep2.context_ref"(%arg0) <{name = "flow_grp"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %12 = ep2.struct_access %1[3] : <"recv_desc_t" : isEvent = false, elementTypes = i32, i32, i32, i32> -> i32
    "ep2.store"(%11, %12) : (!ep2.conref<!ep2.any>, i32) -> ()
    %13 = "ep2.nop"() : () -> none
    %14 = "ep2.constant"() <{value = "process_desc"}> : () -> !ep2.atom
    %15 = "ep2.constant"() <{value = 100 : i64}> : () -> i64
    %16 = "ep2.init"(%14, %arg0, %15) : (!ep2.atom, !ep2.context, i64) -> !ep2.struct<"USER_EVENT1" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, i32>
    ep2.return %16 : !ep2.struct<"USER_EVENT1" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, i32>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_USER_EVENT1_process_desc(%arg0: !ep2.context, %arg1: i32) attributes {atom = "process_desc", event = "USER_EVENT1", type = "handler"} {
    %0 = "ep2.context_ref"(%arg0) <{name = "flow_grp"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    "ep2.store"(%0, %arg1) : (!ep2.conref<!ep2.any>, i32) -> ()
    %1 = "ep2.nop"() : () -> none
    "ep2.terminate"() : () -> ()
  }
}
