module {
  ep2.func private @__handler_MY_EVENT_my_handler_event(%arg0: !ep2.context) attributes {atom = "my_handler_event", event = "MY_EVENT", type = "handler"} {
    %0 = "ep2.init"() : () -> i1
    %1 = "ep2.init"() : () -> i32
    %2 = "ep2.constant"() <{value = 1 : i64}> : () -> i32
    %3 = "ep2.constant"() <{value = 0 : i64}> : () -> i1
    %4:2 = scf.if %3 -> (i1, i32) {
      %9 = "ep2.constant"() <{value = 0 : i64}> : () -> i1
      scf.yield %9, %2 : i1, i32
    } else {
      %9 = "ep2.constant"() <{value = 233 : i64}> : () -> i32
      scf.yield %3, %9 : i1, i32
    }
    scf.if %4#0 {
      %9 = "ep2.context_ref"(%arg0) <{name = "const"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
      %10 = "ep2.constant"() <{value = 2 : i64}> : () -> i64
      "ep2.store"(%9, %10) : (!ep2.conref<!ep2.any>, i64) -> ()
      %11 = "ep2.nop"() : () -> none
      %12 = "ep2.context_ref"(%arg0) <{name = "name"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
      "ep2.store"(%12, %4#1) : (!ep2.conref<!ep2.any>, i32) -> ()
      %13 = "ep2.nop"() : () -> none
    }
    %5 = "ep2.constant"() <{value = 2 : i64}> : () -> i64
    %6 = "ep2.add"(%4#1, %5) : (i32, i64) -> i32
    %7 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %8 = "ep2.add"(%4#0, %7) : (i1, i64) -> i1
    "ep2.terminate"() : () -> ()
  }
}
