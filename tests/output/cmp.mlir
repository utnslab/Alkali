module {
  ep2.func private @__handler_MY_EVENT_my_handler_event(%arg0: !ep2.context) attributes {atom = "my_handler_event", event = "MY_EVENT", type = "handler"} {
    %0 = "ep2.init"() : () -> i32
    %1 = "ep2.init"() : () -> i32
    %2 = "ep2.constant"() <{value = 1 : i64}> : () -> i64
    %3 = "ep2.bitcast"(%2) : (i64) -> i32
    %4 = "ep2.constant"() <{value = 2 : i64}> : () -> i64
    %5 = "ep2.bitcast"(%4) : (i64) -> i32
    %6 = "ep2.cmp"(%3, %5) <{predicate = 62 : i16}> : (i32, i32) -> i1
    scf.if %6 {
      %9 = "ep2.context_ref"(%arg0) <{name = "a"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
      "ep2.store"(%9, %3) : (!ep2.conref<!ep2.any>, i32) -> ()
      %10 = "ep2.nop"() : () -> none
    }
    %7 = "ep2.cmp"(%3, %5) <{predicate = 60 : i16}> : (i32, i32) -> i1
    scf.if %7 {
      %9 = "ep2.context_ref"(%arg0) <{name = "a"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
      "ep2.store"(%9, %3) : (!ep2.conref<!ep2.any>, i32) -> ()
      %10 = "ep2.nop"() : () -> none
    }
    %8 = "ep2.cmp"(%3, %5) <{predicate = 42 : i16}> : (i32, i32) -> i1
    scf.if %8 {
      %9 = "ep2.context_ref"(%arg0) <{name = "a"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
      "ep2.store"(%9, %3) : (!ep2.conref<!ep2.any>, i32) -> ()
      %10 = "ep2.nop"() : () -> none
    }
    "ep2.terminate"() : () -> ()
  }
}
