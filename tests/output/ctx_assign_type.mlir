// ep2c --emit=mlir tests/ctx_assign_type.ep2 

module {
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", type = "handler"} {
    %0 = "ep2.init"() : () -> i48
    %1 = "ep2.context_ref"(%arg0) <{name = "tmp_mac"}> : (!ep2.context) -> !ep2.conref<!ep2.any>
    %2 = "ep2.load"(%1) : (!ep2.conref<!ep2.any>) -> i48
    ep2.return
  }
}