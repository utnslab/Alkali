// ep2c --emit=mlir tests/l2_echo.ep2.txt

module {
  ep2.func private @__handler_NET_SEND(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {event = "NET_SEND", type = "handler"} {
    ep2.return
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf> attributes {atom = "process_packet", event = "NET_RECV", type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %1 = "ep2.init"() : () -> i48
    %2 = "ep2.init"() : () -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>
    %3 = "ep2.init"() : () -> !ep2.buf
    %4 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %5 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>
    %6 = ep2.struct_access %4[1] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %7 = ep2.struct_access %4[0] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %8 = ep2.struct_access %4[1] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %9 = "ep2.struct_update"(%4, %7) <{index = 1 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i48) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %10 = ep2.struct_access %9[0] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %11 = "ep2.struct_update"(%9, %6) <{index = 0 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i48) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    "ep2.emit"(%3, %11) : (!ep2.buf, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> ()
    %12 = "ep2.nop"() : () -> none
    "ep2.emit"(%3, %5) : (!ep2.buf, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    %13 = "ep2.nop"() : () -> none
    "ep2.emit"(%3, %arg1) : (!ep2.buf, !ep2.buf) -> ()
    %14 = "ep2.nop"() : () -> none
    %15 = "ep2.init"(%arg0, %3) : (!ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    ep2.return %15 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
  }
}
