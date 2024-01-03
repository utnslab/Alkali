module {
  ep2.func private @__handler_MY_EVENT_my_handler_event(%arg0: !ep2.context, %arg1: i32, %arg2: i32) attributes {atom = "my_handler_event", event = "MY_EVENT", type = "handler"} {
    %0 = "ep2.init"() : () -> i32
    %1 = "ep2.init"() : () -> i1
    %2 = "ep2.init"() : () -> i1
    %3 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %4 = "ep2.bitcast"(%3) : (i64) -> i32
    %5 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %6 = "ep2.cmp"(%arg1, %5) <{predicate = 40 : i16}> : (i32, i64) -> i1
    %7 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %8 = "ep2.cmp"(%arg2, %7) <{predicate = 40 : i16}> : (i32, i64) -> i1
    %9:2 = scf.if %6 -> (i1, i32) {
      %12 = "ep2.constant"() <{value = 0 : i64}> : () -> i64
      %13 = "ep2.bitcast"(%12) : (i64) -> i1
      scf.yield %13, %4 : i1, i32
    } else {
      %12 = "ep2.constant"() <{value = 233 : i64}> : () -> i64
      %13 = "ep2.bitcast"(%12) : (i64) -> i32
      scf.yield %6, %13 : i1, i32
    }
    %10 = "ep2.constant"() <{value = 2 : i64}> : () -> i64
    %11 = "ep2.add"(%9#1, %10) : (i32, i64) -> i32
    scf.if %8 {
      %12 = "ep2.context_ref"(%arg0) <{name = "const"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
      %13 = "ep2.constant"() <{value = 2 : i64}> : () -> i64
      "ep2.store"(%12, %13) : (!ep2.conref<!ep2.any>, i64) -> ()
      %14 = "ep2.nop"() : () -> none
      %15 = "ep2.context_ref"(%arg0) <{name = "name"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
      "ep2.store"(%15, %11) : (!ep2.conref<!ep2.any>, i32) -> ()
      %16 = "ep2.nop"() : () -> none
    }
    "ep2.terminate"() : () -> ()
  }
}
