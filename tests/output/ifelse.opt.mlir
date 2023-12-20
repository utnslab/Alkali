module {
  ep2.func private @__handler_MY_EVENT_my_handler_event(%arg0: !ep2.context) attributes {atom = "my_handler_event", event = "MY_EVENT", type = "handler"} {
    %false = arith.constant false
    %0 = "ep2.constant"() <{value = 1 : i64}> : () -> i32
    %1 = "ep2.constant"() <{value = 0 : i64}> : () -> i1
    %2:2 = scf.if %1 -> (i1, i32) {
      %3 = "ep2.constant"() <{value = 0 : i64}> : () -> i1
      scf.yield %3, %0 : i1, i32
    } else {
      %3 = "ep2.constant"() <{value = 233 : i64}> : () -> i32
      scf.yield %false, %3 : i1, i32
    }
    scf.if %2#0 {
      %3 = "ep2.context_ref"(%arg0) <{name = "const"}> : (!ep2.context) -> !ep2.conref<i64>
      %4 = "ep2.constant"() <{value = 2 : i64}> : () -> i64
      "ep2.store"(%3, %4) : (!ep2.conref<i64>, i64) -> ()
      %5 = "ep2.context_ref"(%arg0) <{name = "name"}> : (!ep2.context) -> !ep2.conref<i32>
      "ep2.store"(%5, %2#1) : (!ep2.conref<i32>, i32) -> ()
    }
    "ep2.terminate"() : () -> ()
  }
}

