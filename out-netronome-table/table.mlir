module {
  ep2.func private @__handler_LOAD_TABLE_load_table(%arg0: !ep2.context) attributes {atom = "load_table", event = "LOAD_TABLE", type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.table<i16, i32, 16>
    %1 = "ep2.init"() : () -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %2 = ep2.struct_access %1[0] : <"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16> -> i48
    %3 = "ep2.constant"() <{value = 2 : i64}> : () -> i64
    %4 = "ep2.struct_update"(%1, %3) <{index = 0 : i64}> : (!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>, i64) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %5 = "ep2.constant"() <{value = 10 : i64}> : () -> i64
    %6 = "ep2.constant"() <{value = 11 : i64}> : () -> i64
    "ep2.update"(%0, %5, %6) : (!ep2.table<i16, i32, 16>, i64, i64) -> ()
    %7 = "ep2.nop"() : () -> none
    %8 = "ep2.context_ref"(%arg0) <{name = "a"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %9 = "ep2.constant"() <{value = 10 : i64}> : () -> i64
    %10 = "ep2.lookup"(%0, %9) : (!ep2.table<i16, i32, 16>, i64) -> i32
    "ep2.store"(%8, %10) : (!ep2.conref<!ep2.any>, i32) -> ()
    %11 = "ep2.nop"() : () -> none
    "ep2.terminate"() : () -> ()
  }
}
