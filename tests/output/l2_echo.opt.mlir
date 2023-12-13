// ep2c-opt l2_echo.mlir --canonicalize -o l2_echo.opt.mlir

module {
  ep2.func private @__handler_NET_SEND(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {event = "NET_SEND", type = "handler"} {
    ep2.return
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf> attributes {atom = "process_packet", event = "NET_RECV", type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.buf
    %1 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %2 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>
    %3 = ep2.struct_access %1[1] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %4 = ep2.struct_access %1[0] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %5 = "ep2.struct_update"(%1, %4) <{index = 1 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i48) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %6 = "ep2.struct_update"(%5, %3) <{index = 0 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i48) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    "ep2.emit"(%0, %6) : (!ep2.buf, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> ()
    "ep2.emit"(%0, %2) : (!ep2.buf, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    "ep2.emit"(%0, %arg1) : (!ep2.buf, !ep2.buf) -> ()
    %7 = "ep2.init"(%arg0, %0) : (!ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    ep2.return %7 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
  }
}
