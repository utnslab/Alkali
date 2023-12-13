module {
  ep2.func private @__handler_LOAD_TABLE_load_table(%arg0: !ep2.context) attributes {atom = "load_table", event = "LOAD_TABLE", type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.table<i16, i32, 16>
    %1 = "ep2.init"() : () -> !ep2.table<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, !ep2.buf, 16>
    %2 = "ep2.init"() : () -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %3 = "ep2.constant"() <{value = 10 : i64}> : () -> i64
    %4 = "ep2.constant"() <{value = 11 : i64}> : () -> i64
    "ep2.update"(%0, %3, %4) : (!ep2.table<i16, i32, 16>, i64, i64) -> ()
    %5 = "ep2.nop"() : () -> none
    %6 = "ep2.constant"() <{value = 12 : i64}> : () -> i64
    "ep2.update"(%0, %2, %6) : (!ep2.table<i16, i32, 16>, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i64) -> ()
    %7 = "ep2.nop"() : () -> none
    %8 = "ep2.constant"() <{value = 10 : i64}> : () -> i64
    %9 = "ep2.lookup"(%0, %8) : (!ep2.table<i16, i32, 16>, i64) -> i32
    %10 = "ep2.lookup"(%1, %2) : (!ep2.table<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, !ep2.buf, 16>, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> !ep2.buf
    ep2.return
  }
}
