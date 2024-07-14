// ./build/bin/ep2c-opt -canonicalize -cse -canonicalize -ep2-buffer-to-value nopipeline.mlir -o opt.mlir

module {
  ep2.func private @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_DMA_WRITE_REQ_dma_write(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.struct<"dma_write_cmd_t" : isEvent = false, elementTypes = i32, i32>) attributes {atom = "dma_write", event = "DMA_WRITE_REQ", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func private @__handler_NET_RECV_process_packet(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", instances = ["i1cu2", "i1cu3", "i1cu4", "i1cu5"], type = "handler"} {
    %0 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %1, %output = "ep2.extract_value"(%arg1) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>)
    %2, %output_0 = "ep2.extract_value"(%1) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>)
    %3, %output_1 = "ep2.extract_value"(%2) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>)
    %4, %output_2 = "ep2.extract_value"(%3) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>)
    %5, %output_3 = "ep2.extract_value"(%4) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>)
    %6, %output_4 = "ep2.extract_value"(%5) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>)
    %7, %output_5 = "ep2.extract_value"(%6) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>)
    %8, %output_6 = "ep2.extract_value"(%7) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>)
    %9, %output_7 = "ep2.extract_value"(%8) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>)
    %10 = ep2.struct_access %output_6[0] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %11 = ep2.struct_access %output_5[0] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %12 = "ep2.add"(%10, %11) : (i64, i64) -> i64
    %13 = ep2.struct_access %output_4[0] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %14 = "ep2.add"(%12, %13) : (i64, i64) -> i64
    %15 = ep2.struct_access %output_3[0] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %16 = "ep2.add"(%14, %15) : (i64, i64) -> i64
    %17 = ep2.struct_access %output_2[0] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %18 = "ep2.add"(%16, %17) : (i64, i64) -> i64
    %19 = ep2.struct_access %output_1[0] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %20 = "ep2.add"(%18, %19) : (i64, i64) -> i64
    %21 = ep2.struct_access %output_0[0] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %22 = "ep2.add"(%20, %21) : (i64, i64) -> i64
    %23 = ep2.struct_access %output[0] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %24 = "ep2.add"(%22, %23) : (i64, i64) -> i64
    %25 = "ep2.struct_update"(%output_7, %24) <{index = 1 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>, i64) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>
    %26 = ep2.struct_access %output_6[1] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %27 = ep2.struct_access %output_5[1] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %28 = "ep2.add"(%26, %27) : (i64, i64) -> i64
    %29 = ep2.struct_access %output_4[1] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %30 = "ep2.add"(%28, %29) : (i64, i64) -> i64
    %31 = ep2.struct_access %output_3[1] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %32 = "ep2.add"(%30, %31) : (i64, i64) -> i64
    %33 = ep2.struct_access %output_2[1] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %34 = "ep2.add"(%32, %33) : (i64, i64) -> i64
    %35 = ep2.struct_access %output_1[1] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %36 = "ep2.add"(%34, %35) : (i64, i64) -> i64
    %37 = ep2.struct_access %output_0[1] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %38 = "ep2.add"(%36, %37) : (i64, i64) -> i64
    %39 = ep2.struct_access %output[1] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %40 = "ep2.add"(%38, %39) : (i64, i64) -> i64
    %41 = "ep2.struct_update"(%25, %40) <{index = 2 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>, i64) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>
    %42 = ep2.struct_access %output_6[2] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %43 = ep2.struct_access %output_5[2] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %44 = "ep2.add"(%42, %43) : (i64, i64) -> i64
    %45 = ep2.struct_access %output_4[2] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %46 = "ep2.add"(%44, %45) : (i64, i64) -> i64
    %47 = ep2.struct_access %output_3[2] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %48 = "ep2.add"(%46, %47) : (i64, i64) -> i64
    %49 = ep2.struct_access %output_2[2] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %50 = "ep2.add"(%48, %49) : (i64, i64) -> i64
    %51 = ep2.struct_access %output_1[2] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %52 = "ep2.add"(%50, %51) : (i64, i64) -> i64
    %53 = ep2.struct_access %output_0[2] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %54 = "ep2.add"(%52, %53) : (i64, i64) -> i64
    %55 = ep2.struct_access %output[2] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %56 = "ep2.add"(%54, %55) : (i64, i64) -> i64
    %57 = "ep2.struct_update"(%41, %56) <{index = 0 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>, i64) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>
    %58 = ep2.struct_access %output_6[3] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %59 = ep2.struct_access %output_5[3] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %60 = "ep2.add"(%58, %59) : (i64, i64) -> i64
    %61 = ep2.struct_access %output_4[3] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %62 = "ep2.add"(%60, %61) : (i64, i64) -> i64
    %63 = ep2.struct_access %output_3[3] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %64 = "ep2.add"(%62, %63) : (i64, i64) -> i64
    %65 = ep2.struct_access %output_2[3] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %66 = "ep2.add"(%64, %65) : (i64, i64) -> i64
    %67 = ep2.struct_access %output_1[3] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %68 = "ep2.add"(%66, %67) : (i64, i64) -> i64
    %69 = ep2.struct_access %output_0[3] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %70 = "ep2.add"(%68, %69) : (i64, i64) -> i64
    %71 = ep2.struct_access %output[3] : <"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64> -> i64
    %72 = "ep2.add"(%70, %71) : (i64, i64) -> i64
    %73 = "ep2.struct_update"(%57, %72) <{index = 3 : i64}> : (!ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>, i64) -> !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>
    %74 = "ep2.emit_value"(%9, %73) : (!ep2.buf, !ep2.struct<"pkt_info_t" : isEvent = false, elementTypes = i64, i64, i64, i64>) -> !ep2.buf
    %75 = "ep2.init"(%0, %arg0, %74) : (!ep2.atom, !ep2.context, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    ep2.return %75 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.context, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}

