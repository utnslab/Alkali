module {
  ep2.func private @__handler_LOAD_TABLE_load_table(%arg0: !ep2.context) attributes {atom = "load_table", event = "LOAD_TABLE", type = "handler"} {
    %0 = "ep2.constant"() <{value = 10 : i16}> : () -> i16
    %1 = "ep2.constant"() <{value = 11 : i32}> : () -> i32
    %2 = "ep2.init"() : () -> !ep2.table<i16, i32, 16>
    "ep2.update"(%2, %0, %1) : (!ep2.table<i16, i32, 16>, i16, i32) -> ()
    %3 = "ep2.context_ref"(%arg0) <{name = "a"}> : (!ep2.context) -> !ep2.conref<i32>
    %4 = "ep2.lookup"(%2, %0) : (!ep2.table<i16, i32, 16>, i16) -> i32
    "ep2.store"(%3, %4) : (!ep2.conref<i32>, i32) -> ()
    "ep2.terminate"() : () -> ()
  }
}

