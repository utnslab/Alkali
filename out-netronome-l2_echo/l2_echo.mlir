module {
  ep2.func private @__handler_NET_SEND_main_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "main_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_NET_RECV_main_recv(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "main_recv", event = "NET_RECV", type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %1 = "ep2.init"() : () -> i48
    %2 = "ep2.init"() : () -> !ep2.buf
    %3 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %4 = ep2.struct_access %3[1] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %5 = ep2.struct_access %3[1] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %6 = ep2.struct_access %3[0] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %7 = "ep2.struct_update"(%3, %6) <{index = 1 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i48) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %8 = ep2.struct_access %7[0] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %9 = "ep2.struct_update"(%7, %4) <{index = 0 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i48) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    "ep2.emit"(%2, %9) : (!ep2.buf, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> ()
    %10 = "ep2.nop"() : () -> none
    "ep2.emit"(%2, %arg1) : (!ep2.buf, !ep2.buf) -> ()
    %11 = "ep2.nop"() : () -> none
    %12 = "ep2.constant"() <{value = "main_send"}> : () -> !ep2.atom
    %13 = "ep2.init"(%12, %arg0, %2) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    ep2.return %13 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}
