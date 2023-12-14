// ep2c --emit=mlir tests/ctx_assign_type.ep2 

module {
  ep2.func private @__handler_A_a(%arg0: !ep2.context) -> !ep2.struct<"B" : isEvent = true, elementTypes = !ep2.atom, !ep2.context> attributes {atom = "a", event = "A", type = "handler"} {
    %0 = "ep2.init"() : () -> i48
    %1 = "ep2.init"() : () -> !ep2.buf
    %2 = "ep2.context_ref"(%arg0) <{name = "t1"}> : (!ep2.context) -> !ep2.conref<i48>
    "ep2.store"(%2, %0) : (!ep2.conref<i48>, i48) -> ()
    %3 = "ep2.nop"() : () -> none
    %4 = "ep2.context_ref"(%arg0) <{name = "t2"}> : (!ep2.context) -> !ep2.conref<!ep2.buf>
    "ep2.store"(%4, %1) : (!ep2.conref<!ep2.buf>, !ep2.buf) -> ()
    %5 = "ep2.nop"() : () -> none
    %6 = "ep2.init"(%arg0) : (!ep2.context) -> !ep2.struct<"B" : isEvent = true, elementTypes = !ep2.atom, !ep2.context>
    ep2.return %6 : !ep2.struct<"B" : isEvent = true, elementTypes = !ep2.atom, !ep2.context>
  }
  ep2.func private @__handler_B_b(%arg0: !ep2.context) attributes {atom = "b", event = "B", type = "handler"} {
    %0 = "ep2.init"() : () -> i48
    %1 = "ep2.init"() : () -> !ep2.buf
    %2 = "ep2.context_ref"(%arg0) <{name = "t1"}> : (!ep2.context) -> !ep2.conref<i48>
    %3 = "ep2.load"(%2) : (!ep2.conref<i48>) -> i48
    %4 = "ep2.context_ref"(%arg0) <{name = "t2"}> : (!ep2.context) -> !ep2.conref<!ep2.buf>
    "ep2.emit"(%1, %4) : (!ep2.buf, !ep2.conref<!ep2.buf>) -> ()
    %5 = "ep2.nop"() : () -> none
    ep2.return
  }
}