module {
  ep2.func private @__handler_MY_EVENT_my_handler_event(%arg0: !ep2.context, %arg1: i32, %arg2: i32) attributes {atom = "my_handler_event", event = "MY_EVENT", type = "handler"} {
    %0 = "ep2.constant"() <{value = 233 : i32}> : () -> i32
    %1 = "ep2.constant"() <{value = 1 : i32}> : () -> i32
    %2 = "ep2.constant"() <{value = 2 : i64}> : () -> i64
    %3 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %4 = "ep2.cmp"(%arg1, %3) <{predicate = 40 : i16}> : (i32, i64) -> i1
    %5 = "ep2.cmp"(%arg2, %3) <{predicate = 40 : i16}> : (i32, i64) -> i1
    %6 = arith.select %4, %1, %0 : i32
    %7 = "ep2.add"(%6, %2) : (i32, i64) -> i32
    scf.if %5 {
      %8 = "ep2.context_ref"(%arg0) <{name = "const"}> : (!ep2.context) -> !ep2.conref<i64>
      "ep2.store"(%8, %2) : (!ep2.conref<i64>, i64) -> ()
      %9 = "ep2.context_ref"(%arg0) <{name = "name"}> : (!ep2.context) -> !ep2.conref<i32>
      "ep2.store"(%9, %7) : (!ep2.conref<i32>, i32) -> ()
    }
    "ep2.terminate"() : () -> ()
  }
}

