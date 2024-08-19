// ./build/bin/ep2c-opt -canonicalize -cse -canonicalize -ep2-buffer-to-value nopipeline.mlir -o opt.mlir

module {
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["i1cu2", "i1cu3", "i1cu4", "i1cu5"], type = "handler"} {
    %0 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %1, %output = "ep2.extract_value"(%arg1) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>)
    %23 = ep2.struct_access %output[0] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %24 = ep2.struct_access %output[1] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %25 = "ep2.add"(%23, %24) : (i64, i64) -> i64
    %73 = "ep2.struct_update"(%output, %25) <{index = 3 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>, i64) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>
    %74 = "ep2.emit_value"(%1, %73) : (!ep2.buf, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>) -> !ep2.buf
    %75 = "ep2.init"(%0, %arg0, %74) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    ep2.return %75 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}

