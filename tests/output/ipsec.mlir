module {
  ep2.func private @__handler_NET_SEND(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {event = "NET_SEND", type = "handler"} {
    ep2.return
  }
  ep2.func private @__handler_DECRYPT_REQ(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"decrypt_cmd_t" : isEvent = false, elementTypes = i16, i256>) -> !ep2.struct<"DECRYPT_CMPL" : isEvent = true, elementTypes = !ep2.context, !ep2.buf> attributes {event = "DECRYPT_REQ", type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.buf
    %1 = "ep2.init"(%arg0, %0) : (!ep2.context, !ep2.buf) -> !ep2.struct<"DECRYPT_CMPL" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    ep2.return %1 : !ep2.struct<"DECRYPT_CMPL" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
  }
  ep2.func private @__handler_DMA_SEND(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16>) attributes {event = "DMA_SEND", type = "handler"} {
    ep2.return
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) -> !ep2.struct<"DECRYPT_REQ" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"decrypt_cmd_t" : isEvent = false, elementTypes = i16, i256>> attributes {atom = "process_packet", event = "NET_RECV", type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %1 = "ep2.init"() : () -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>
    %2 = "ep2.init"() : () -> !ep2.struct<"esp_h" : isEvent = false, elementTypes = i32, i32>
    %3 = "ep2.init"() : () -> !ep2.struct<"decrypt_cmd_t" : isEvent = false, elementTypes = i16, i256>
    %4 = "ep2.init"() : () -> i32
    %5 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %6 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>
    %7 = "ep2.extract"(%arg1) : (!ep2.buf) -> !ep2.struct<"esp_h" : isEvent = false, elementTypes = i32, i32>
    %8 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>
    "ep2.store"(%8, %5) : (!ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>) -> ()
    %9 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>>
    "ep2.store"(%9, %6) : (!ep2.conref<!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>>, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    %10 = "ep2.context_ref"(%arg0) <{name = "esp_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"esp_h" : isEvent = false, elementTypes = i32, i32>>
    "ep2.store"(%10, %7) : (!ep2.conref<!ep2.struct<"esp_h" : isEvent = false, elementTypes = i32, i32>>, !ep2.struct<"esp_h" : isEvent = false, elementTypes = i32, i32>) -> ()
    %11 = "ep2.init"(%arg0, %arg1, %3) : (!ep2.context, !ep2.buf, !ep2.struct<"decrypt_cmd_t" : isEvent = false, elementTypes = i16, i256>) -> !ep2.struct<"DECRYPT_REQ" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"decrypt_cmd_t" : isEvent = false, elementTypes = i16, i256>>
    ep2.return %11 : !ep2.struct<"DECRYPT_REQ" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"decrypt_cmd_t" : isEvent = false, elementTypes = i16, i256>>
  }
  ep2.func private @__controller_DECRYPT_REQ() attributes {event = "DECRYPT_REQ", type = "controller"} {
    %0 = "ep2.constant"() <{value = 100 : i64}> : () -> i64
    %1 = "ep2.constant"() <{value = 4 : i64}> : () -> i64
    %2 = ep2.call @Queue(%0, %1) : (i64, i64) -> i64
    ep2.return
  }
  ep2.func private @__handler_DECRYPT_CMPL_dma_to_host(%arg0: !ep2.context, %arg1: !ep2.buf) -> !ep2.struct<"DMA_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16>> attributes {atom = "dma_to_host", event = "DECRYPT_CMPL", type = "handler"} {
    %0 = "ep2.init"() : () -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %1 = "ep2.init"() : () -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>
    %2 = "ep2.init"() : () -> !ep2.struct<"esp_h" : isEvent = false, elementTypes = i32, i32>
    %3 = "ep2.init"() : () -> !ep2.buf
    %4 = "ep2.init"() : () -> !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16>
    %5 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>
    %6 = "ep2.load"(%5) : (!ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>) -> !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>
    %7 = "ep2.context_ref"(%arg0) <{name = "ip_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>>
    %8 = "ep2.load"(%7) : (!ep2.conref<!ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>>) -> !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>
    %9 = "ep2.context_ref"(%arg0) <{name = "esp_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"esp_h" : isEvent = false, elementTypes = i32, i32>>
    %10 = "ep2.load"(%9) : (!ep2.conref<!ep2.struct<"esp_h" : isEvent = false, elementTypes = i32, i32>>) -> !ep2.struct<"esp_h" : isEvent = false, elementTypes = i32, i32>
    %11 = "ep2.context_ref"(%arg0) <{name = "eth_header"}> : (!ep2.context) -> !ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>
    "ep2.emit"(%3, %11) : (!ep2.buf, !ep2.conref<!ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>>) -> ()
    "ep2.emit"(%3, %8) : (!ep2.buf, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i8, i16, i16, i16, i16, i16, i32, i32, i32>) -> ()
    "ep2.emit"(%3, %10) : (!ep2.buf, !ep2.struct<"esp_h" : isEvent = false, elementTypes = i32, i32>) -> ()
    "ep2.emit"(%3, %arg1) : (!ep2.buf, !ep2.buf) -> ()
    %12 = ep2.struct_access %4[0] : <"dma_desc_t" : isEvent = false, elementTypes = i16, i16> -> i16
    %13 = "ep2.constant"() <{value = 0 : i64}> : () -> i64
    %14 = "ep2.struct_update"(%4, %13) <{index = 0 : i64}> : (!ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16>, i64) -> !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16>
    %15 = "ep2.init"(%arg0, %3, %14) : (!ep2.context, !ep2.buf, !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16>) -> !ep2.struct<"DMA_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16>>
    ep2.return %15 : !ep2.struct<"DMA_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf, !ep2.struct<"dma_desc_t" : isEvent = false, elementTypes = i16, i16>>
  }
}

