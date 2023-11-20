module {
  func.func @__controller_DMA_READ_REQ() {
    %0 = "emitc.constant"() <{value = 100 : i64}> : () -> i64
    %1 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %2 = "emitc.constant"() <{value = 10 : i64}> : () -> i64
    %3 = emitc.call "Queue"(%0, %1, %2) : (i64, i64, i64) -> i64
    return
  }
  func.func @__controller_DMA_READ_CMPL() {
    %0 = "emitc.constant"() <{value = 100 : i64}> : () -> i64
    %1 = "emitc.constant"() <{value = 1 : i64}> : () -> i64
    %2 = "emitc.constant"() <{value = 10 : i64}> : () -> i64
    %3 = emitc.call "Queue"(%0, %1, %2) : (i64, i64, i64) -> i64
    return
  }
  func.func @__event___handler_DMA_RECV_CMPL_receive_desc(%arg0: !emitc.ptr<!emitc.opaque<"struct __wrapper_arg">>, %arg1: !emitc.ptr<!emitc.opaque<"struct __wrapper_arg">>) {
    %0 = emitc.call "__ep2_intrin_struct_access"(%arg0) {args = [0 : i32]} : (!emitc.ptr<!emitc.opaque<"struct __wrapper_arg">>) -> i32
    %1 = emitc.call "__ep2_intrin_struct_access"(%arg0) {args = [1 : i32]} : (!emitc.ptr<!emitc.opaque<"struct __wrapper_arg">>) -> !emitc.ptr<!emitc.opaque<"struct in_t">>
    %2 = emitc.call "__ep2_intrin_struct_access"(%1) {args = [0 : i32]} : (!emitc.ptr<!emitc.opaque<"struct in_t">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %3 = emitc.call "__ep2_rt_alloc_struct"() {args = [0 : i32]} : () -> !emitc.ptr<!emitc.opaque<"struct Desc_Hdr">>
    %4 = emitc.call "__ep2_rt_alloc_buf"() : () -> !emitc.ptr<!emitc.opaque<"struct Desc_Hdr">>
    emitc.call "__ep2_rt_extract"(%4, %2) {args = [0 : i32]} : (!emitc.ptr<!emitc.opaque<"struct Desc_Hdr">>, !emitc.ptr<!emitc.opaque<"void">>) -> ()
    emitc.call "__ep2_rt_wr"(%0, %4) {args = [8 : i32]} : (i32, !emitc.ptr<!emitc.opaque<"struct Desc_Hdr">>) -> ()
    %5 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %6 = emitc.call "__ep2_intrin_struct_access"(%3) {args = [0 : i32]} : (!emitc.ptr<!emitc.opaque<"struct Desc_Hdr">>) -> i64
    %7 = "emitc.constant"() <{value = 100 : i64}> : () -> i64
    %8 = emitc.call "__ep2_rt_alloc_struct"() {args = [0 : i32]} : () -> !emitc.ptr<!emitc.opaque<"struct DMA_READ_REQ">>
    emitc.call "__ep2_intrin_struct_write"(%5, %8) {args = [0 : i32]} : (i32, !emitc.ptr<!emitc.opaque<"struct DMA_READ_REQ">>) -> ()
    emitc.call "__ep2_intrin_struct_write"(%0, %8) {args = [1 : i32]} : (i32, !emitc.ptr<!emitc.opaque<"struct DMA_READ_REQ">>) -> ()
    emitc.call "__ep2_intrin_struct_write"(%6, %8) {args = [2 : i32]} : (i64, !emitc.ptr<!emitc.opaque<"struct DMA_READ_REQ">>) -> ()
    emitc.call "__ep2_intrin_struct_write"(%7, %8) {args = [3 : i32]} : (i64, !emitc.ptr<!emitc.opaque<"struct DMA_READ_REQ">>) -> ()
    return
  }
  func.func @__event___handler_DMA_READ_REQ_receive_payload_1(%arg0: !emitc.ptr<!emitc.opaque<"struct __wrapper_arg">>) {
    %0 = emitc.call "__ep2_intrin_struct_access"(%arg0) {args = [0 : i32]} : (!emitc.ptr<!emitc.opaque<"struct __wrapper_arg">>) -> i32
    %1 = emitc.call "__ep2_intrin_struct_access"(%arg0) {args = [1 : i32]} : (!emitc.ptr<!emitc.opaque<"struct __wrapper_arg">>) -> !emitc.ptr<!emitc.opaque<"struct in_t">>
    %2 = emitc.call "__ep2_intrin_struct_access"(%1) {args = [0 : i32]} : (!emitc.ptr<!emitc.opaque<"struct in_t">>) -> i64
    %3 = emitc.call "__ep2_intrin_struct_access"(%1) {args = [1 : i32]} : (!emitc.ptr<!emitc.opaque<"struct in_t">>) -> i64
    emitc.call "__ep2_rt_wr"(%0, %2) {args = [8 : i32]} : (i32, i64) -> ()
    return
  }
}

