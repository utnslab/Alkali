module {
  ep2.func private @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, instances = ["cu7"], location = "cu7", type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_WRITE_REQ_dma_write(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) attributes {atom = "dma_write", event = "DMA_WRITE_REQ", extern = true, instances = ["cu8"], location = "cu8", type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  %0 = "ep2.global"() <{name = "flow_table"}> : () -> !ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32, i16, i16, i32, i32, i8, i8, i16, i16, i16, i16, i16, i32, i32, i32>, 16>
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["cu2", "cu3"], location = "cu2", type = "handler"} {
    %1 = "ep2.constant"() <{value = 1 : i16}> : () -> i16
    %2 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %3 = "ep2.global_import"() <{name = "flow_table"}> : () -> !ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32, i16, i16, i32, i32, i8, i8, i16, i16, i16, i16, i16, i32, i32, i32>, 16>
    %4 = "ep2.lookup"(%3, %1) : (!ep2.table<i16, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32, i16, i16, i32, i32, i8, i8, i16, i16, i16, i16, i16, i32, i32, i32>, 16>, i16) -> !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32, i16, i16, i32, i32, i8, i8, i16, i16, i16, i16, i16, i32, i32, i32>
    "ep2.emit"(%arg1, %4) : (!ep2.buf, !ep2.struct<"flow_state_t" : isEvent = false, elementTypes = i32, i16, i16, i32, i32, i32, i32, i32, i32, i16, i16, i32, i32, i8, i8, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    %5 = "ep2.init"(%2, %arg0, %arg1) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    ep2.return %5 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}

