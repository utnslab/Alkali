module {
  ep2.func private @__handler_NET_SEND_main_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "main_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_NET_RECV_main_recv(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "main_recv", event = "NET_RECV", type = "handler"} {
    %0 = "ep2.constant"() <{value = "main_send"}> : () -> !ep2.atom
    %1 = "ep2.init"() : () -> !ep2.buf
    %2 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %3 = ep2.struct_access %2[1] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %4 = ep2.struct_access %2[0] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %5 = "ep2.struct_update"(%2, %4) <{index = 1 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i48) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %6 = "ep2.struct_update"(%5, %3) <{index = 0 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i48) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    "ep2.emit"(%1, %6) : (!ep2.buf, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> ()
    "ep2.emit"(%1, %arg1) : (!ep2.buf, !ep2.buf) -> ()
    %7 = "ep2.init"(%0, %arg0, %1) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    ep2.return %7 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}

