// ./build/bin/ep2c-opt --canonicalize -cse --ep2-context-infer --ep2-context-to-argument -ep2-buffer-to-value tmp.mlir > tests/output/buf_to_value.buffer_to_value.mlir 
module {
  ep2.func private @__handler_MY_EVENT_my_handler_event(%arg0: !ep2.buf, %arg1: i32, %arg2: i32 {ep2.context_name = "test"}, %arg3: i32 {ep2.context_name = "bench"}) attributes {atom = "my_handler_event", event = "MY_EVENT", type = "handler"} {
    %0 = "ep2.constant"() <{value = 4 : i32}> : () -> i32
    %1 = "ep2.constant"() <{value = 2 : i32}> : () -> i32
    %2 = "ep2.constant"() <{value = 1 : i32}> : () -> i32
    %3 = "ep2.constant"() <{value = "my_handler_event"}> : () -> !ep2.atom
    %4, %output = "ep2.extract_value"(%arg0) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"header" : isEvent = false, elementTypes = i32, i32>)
    %5 = "ep2.emit_value"(%4, %output) : (!ep2.buf, !ep2.struct<"header" : isEvent = false, elementTypes = i32, i32>) -> !ep2.buf
    %6 = ep2.struct_access %output[0] : <"header" : isEvent = false, elementTypes = i32, i32> -> i32
    %7 = "ep2.cmp"(%2, %arg1) <{predicate = 60 : i16}> : (i32, i32) -> i1
    %8 = arith.select %7, %0, %arg3 : i32
    %9 = arith.select %7, %0, %1 : i32
    %10 = "ep2.init"(%3, %5, %6, %9, %8) : (!ep2.atom, !ep2.buf, i32, i32, i32) -> !ep2.struct<"MY_EVENT_OUT" : isEvent = true, elementTypes = !ep2.atom, !ep2.buf, i32, i32, i32>
    ep2.return %10 : !ep2.struct<"MY_EVENT_OUT" : isEvent = true, elementTypes = !ep2.atom, !ep2.buf, i32, i32, i32>
    "ep2.terminate"() : () -> ()
  }
}

