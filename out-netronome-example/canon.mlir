module {
  ep2.func private @__controller_DMA_RECV_CMPL() attributes {event = "DMA_RECV_CMPL", extern = true, type = "controller"} {
    %0 = "ep2.constant"() <{value = 256 : i64}> : () -> i64
    %1 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %2 = ep2.call @Queue(%0, %1, %1) : (i64, i64, i64) -> i64
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__controller_USER_EVENT1() attributes {event = "USER_EVENT1", type = "controller"} {
    %0 = "ep2.constant"() <{value = 256 : i64}> : () -> i64
    %1 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %2 = ep2.call @Queue(%0, %1, %1) : (i64, i64, i64) -> i64
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_RECV_CMPL_receive_desc(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "receive_desc", event = "DMA_RECV_CMPL", type = "handler"} {
    %0 = "ep2.constant"() <{value = 100 : i64}> : () -> i64
    %1 = "ep2.constant"() <{value = "process_desc"}> : () -> !ep2.atom
    %2 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"recv_desc_t" : isEvent = false, elementTypes = i32, i32, i32, i32>
    %3 = "ep2.context_ref"(%arg0) <{name = "desc"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"recv_desc_t" : isEvent = false, elementTypes = i32, i32, i32, i32>>
    "ep2.store"(%3, %2) : (!ep2.conref<!ep2.struct<"recv_desc_t" : isEvent = false, elementTypes = i32, i32, i32, i32>>, !ep2.struct<"recv_desc_t" : isEvent = false, elementTypes = i32, i32, i32, i32>) -> ()
    %4 = "ep2.init"(%1, %arg0, %0) : (!ep2.atom, !ep2.context, i64) -> !ep2.struct<"USER_EVENT1" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, i32>
    ep2.return %4 : !ep2.struct<"USER_EVENT1" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, i32>
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_USER_EVENT1_process_desc(%arg0: !ep2.context, %arg1: i32) attributes {atom = "process_desc", event = "USER_EVENT1", type = "handler"} {
    %0 = "ep2.context_ref"(%arg0) <{name = "flow_grp"}> : (!ep2.context) -> !ep2.conref<i32>
    "ep2.store"(%0, %arg1) : (!ep2.conref<i32>, i32) -> ()
    "ep2.terminate"() : () -> ()
  }
}

