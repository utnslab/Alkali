// ./build/bin/ep2c-opt --canonicalize -cse --ep2-context-infer --ep2-context-to-argument  tmp.mlir > tests/output/buf_to_value.buffer_to_value.mlir
module {
  ep2.func private @__handler_MY_EVENT_my_handler_event(%arg0: !ep2.buf, %arg1: i32, %arg2: i32 {ep2.context_name = "test"}, %arg3: i32 {ep2.context_name = "bench"}) attributes {atom = "my_handler_event", event = "MY_EVENT", type = "handler"} {
    %0 = "ep2.constant"() <{value = "my_handler_event"}> : () -> !ep2.atom
    %1 = "ep2.constant"() <{value = 1 : i32}> : () -> i32
    %2 = "ep2.constant"() <{value = 2 : i32}> : () -> i32
    %3 = "ep2.constant"() <{value = 4 : i32}> : () -> i32
    %4 = "ep2.extract"(%arg0) : (!ep2.buf) -> !ep2.struct<"header" : isEvent = false, elementTypes = i32, i32>
    "ep2.emit"(%arg0, %4) : (!ep2.buf, !ep2.struct<"header" : isEvent = false, elementTypes = i32, i32>) -> ()
    %5 = ep2.struct_access %4[0] : <"header" : isEvent = false, elementTypes = i32, i32> -> i32
    %6 = "ep2.cmp"(%1, %arg1) <{predicate = 60 : i16}> : (i32, i32) -> i1
    cf.cond_br %6, ^bb1(%3, %3 : i32, i32), ^bb1(%arg3, %2 : i32, i32)
  ^bb1(%7: i32, %8: i32):  // 2 preds: ^bb0, ^bb0
    cf.br ^bb2(%7, %8 : i32, i32)
  ^bb2(%9: i32, %10: i32):  // pred: ^bb1
    %11 = "ep2.init"(%0, %arg0, %5, %10, %9) : (!ep2.atom, !ep2.buf, i32, i32, i32) -> !ep2.struct<"MY_EVENT_OUT" : isEvent = true, elementTypes = !ep2.atom, !ep2.buf, i32, i32, i32>
    ep2.return %11 : !ep2.struct<"MY_EVENT_OUT" : isEvent = true, elementTypes = !ep2.atom, !ep2.buf, i32, i32, i32>
    "ep2.terminate"() : () -> ()
  }
}

