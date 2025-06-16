module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>} {
  %0 = "ep2.global"() <{name = "firewall_ip_table"}> : () -> !ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
  %1 = "ep2.global"() <{name = "firewall_tcpport_table"}> : () -> !ep2.table<i32, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
  %2 = "ep2.global"() <{name = "priority_table"}> : () -> !ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>
  %3 = "ep2.global"() <{name = "err_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
  %4 = "ep2.global"() <{name = "tcp_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
  %5 = "ep2.global"() <{name = "lb_table"}> : () -> !ep2.table<i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, 64>
  %6 = "ep2.global"() <{name = "flow_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"flow_tracker_t" : isEvent = false, elementTypes = i32, i32>, 64>
  %7 = "ep2.global"() <{name = "ip_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"ip_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
  %8 = "ep2.global"() <{name = "lb_fwd_table"}> : () -> !ep2.table<i32, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>, 64>
  ep2.func @__handler_NET_SEND_net_send(%arg0: !ep2.context, %arg1: !ep2.buf) attributes {atom = "net_send", event = "NET_SEND", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_DMA_WRITE_REQ_dma_write(%arg0: !ep2.context, %arg1: !ep2.buf, %arg2: !ep2.buf) attributes {atom = "dma_write", event = "DMA_WRITE_REQ", extern = true, type = "handler"} {
    "ep2.terminate"() : () -> ()
  }
  ep2.func @__handler_NET_RECV_process_packet(%arg0: !ep2.buf) attributes {atom = "process_packet", event = "NET_RECV", type = "handler"} {
    %9 = "ep2.constant"() <{value = 1 : i32}> : () -> i32
    %10 = "ep2.constant"() <{value = 256 : i32}> : () -> i32
    %11 = "ep2.constant"() <{value = 134744072 : i32}> : () -> i32
    %12 = "ep2.constant"() <{value = 134744071 : i32}> : () -> i32
    %13 = "ep2.constant"() <{value = "net_send"}> : () -> !ep2.atom
    %14 = "ep2.constant"() <{value = 50 : i32}> : () -> i32
    %15 = "ep2.constant"() <{value = 60 : i32}> : () -> i32
    %16 = "ep2.global_import"() <{name = "firewall_ip_table"}> : () -> !ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
    %17 = "ep2.global_import"() <{name = "firewall_tcpport_table"}> : () -> !ep2.table<i32, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>
    %18 = "ep2.global_import"() <{name = "priority_table"}> : () -> !ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>
    %19 = "ep2.global_import"() <{name = "err_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
    %20 = "ep2.global_import"() <{name = "tcp_tracker_table"}> : () -> !ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>
    %21 = "ep2.global_import"() <{name = "lb_table"}> : () -> !ep2.table<i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, 64>
    %22 = "ep2.init"() : () -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %23 = "ep2.init"() : () -> !ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %24 = "ep2.init"() : () -> !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>
    %25, %output = "ep2.extract_value"(%arg0) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"eth_header_t" : isEvent = false, elementTypes = i48, i48, i16>)
    %26, %output_0 = "ep2.extract_value"(%25) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32>)
    %27, %output_1 = "ep2.extract_value"(%26) : (!ep2.buf) -> (!ep2.buf, !ep2.struct<"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16>)
    %28 = ep2.struct_access %output_0[6] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %29 = ep2.struct_access %output_0[7] : <"ip_header_t" : isEvent = false, elementTypes = i16, i16, i16, i16, i16, i16, i32, i32, i32> -> i32
    %30 = "ep2.add"(%28, %29) : (i32, i32) -> i32
    %31 = ep2.struct_access %output_1[0] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %32 = "ep2.add"(%30, %31) : (i32, i16) -> i32
    %33 = ep2.struct_access %output_1[1] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %34 = "ep2.add"(%32, %33) : (i32, i16) -> i32
    %35 = "ep2.lookup"(%16, %34) : (!ep2.table<i32, !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>, i32) -> !ep2.struct<"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>
    %36 = "ep2.lookup"(%17, %34) : (!ep2.table<i32, !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>, 64>, i32) -> !ep2.struct<"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32>
    %37 = "ep2.lookup"(%18, %34) : (!ep2.table<i32, !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>, 64>, i32) -> !ep2.struct<"priority_entries_t" : isEvent = false, elementTypes = i32, i32>
    %38 = ep2.struct_access %36[4] : <"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %39 = ep2.struct_access %35[4] : <"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %40 = "ep2.add"(%38, %39) : (i32, i32) -> i32
    %41 = "ep2.struct_update"(%22, %40) <{index = 1 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %42 = ep2.struct_access %37[0] : <"priority_entries_t" : isEvent = false, elementTypes = i32, i32> -> i32
    %43 = "ep2.add"(%42, %42) : (i32, i32) -> i32
    %44 = "ep2.struct_update"(%41, %43) <{index = 2 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %45 = "ep2.struct_update"(%44, %34) <{index = 0 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %46 = ep2.struct_access %35[3] : <"firewall_ip_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %47 = ep2.struct_access %36[3] : <"firewall_tcpport_entries_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32> -> i32
    %48 = "ep2.add"(%46, %47) : (i32, i32) -> i32
    %49 = "ep2.struct_update"(%45, %48) <{index = 3 : i64}> : (!ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %50 = "ep2.emit_value"(%27, %49) : (!ep2.buf, !ep2.struct<"firewall_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>) -> !ep2.buf
    %51 = "ep2.lookup"(%19, %34) : (!ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %52 = ep2.struct_access %51[0] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %53 = "ep2.add"(%52, %9) : (i32, i32) -> i32
    %54 = "ep2.sub"(%53, %40) : (i32, i32) -> i32
    %55 = "ep2.struct_update"(%51, %54) <{index = 0 : i64}> : (!ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %56 = ep2.struct_access %55[2] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %57 = "ep2.add"(%56, %10) : (i32, i32) -> i32
    %58 = ep2.struct_access %output_1[2] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i32
    %59 = "ep2.add"(%57, %58) : (i32, i32) -> i32
    %60 = ep2.struct_access %output_1[7] : <"tcp_header_t" : isEvent = false, elementTypes = i16, i16, i32, i32, i8, i8, i16, i16, i16> -> i16
    %61 = "ep2.add"(%59, %60) : (i32, i16) -> i32
    %62 = "ep2.struct_update"(%55, %61) <{index = 2 : i64}> : (!ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %63 = ep2.struct_access %62[0] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %64 = ep2.struct_access %62[2] : <"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %65 = "ep2.add"(%63, %64) : (i32, i32) -> i32
    %66 = "ep2.add"(%65, %60) : (i32, i16) -> i32
    "ep2.update"(%19, %34, %62) : (!ep2.table<i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>, i32, !ep2.struct<"err_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>) -> ()
    %67 = "ep2.lookup"(%20, %28) : (!ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>, i32) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %68 = ep2.struct_access %67[0] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %69 = "ep2.sub"(%68, %40) : (i32, i32) -> i32
    %70 = "ep2.struct_update"(%67, %69) <{index = 0 : i64}> : (!ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    "ep2.update"(%20, %28, %70) : (!ep2.table<i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, 64>, i32, !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>) -> ()
    %71 = ep2.struct_access %70[2] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %72 = "ep2.add"(%71, %66) : (i32, i32) -> i32
    %73 = "ep2.struct_update"(%70, %72) <{index = 2 : i64}> : (!ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>, i32) -> !ep2.struct<"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32>
    %74 = ep2.struct_access %73[0] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %75 = "ep2.struct_update"(%23, %74) <{index = 0 : i64}> : (!ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %76 = ep2.struct_access %73[2] : <"tcp_tracker_t" : isEvent = false, elementTypes = i32, i32, i32> -> i32
    %77 = "ep2.struct_update"(%75, %76) <{index = 1 : i64}> : (!ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>, i32) -> !ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>
    %78 = "ep2.emit_value"(%50, %77) : (!ep2.buf, !ep2.struct<"connect_tracker_meta_header_t" : isEvent = false, elementTypes = i32, i32, i32, i32, i32, i32, i32>) -> !ep2.buf
    %79 = "ep2.lookup"(%21, %31) : (!ep2.table<i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, 64>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %80 = "ep2.struct_update"(%79, %9) <{index = 6 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %81 = ep2.struct_access %80[0] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i64
    %82 = "ep2.add"(%81, %34) : (i64, i32) -> i32
    %83 = "ep2.bitcast"(%82) : (i32) -> i64
    %84 = "ep2.struct_update"(%80, %83) <{index = 0 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %85 = ep2.struct_access %84[1] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i64
    %86 = "ep2.add"(%85, %34) : (i64, i32) -> i32
    %87 = "ep2.bitcast"(%86) : (i32) -> i64
    %88 = "ep2.struct_update"(%84, %87) <{index = 1 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %89 = "ep2.add"(%11, %34) : (i32, i32) -> i32
    %90 = "ep2.struct_update"(%88, %89) <{index = 2 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %91 = "ep2.add"(%12, %34) : (i32, i32) -> i32
    %92 = "ep2.struct_update"(%90, %91) <{index = 3 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i32) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %93 = "ep2.add"(%14, %34) : (i32, i32) -> i32
    %94 = "ep2.bitcast"(%93) : (i32) -> i16
    %95 = "ep2.struct_update"(%92, %94) <{index = 4 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %96 = "ep2.add"(%15, %34) : (i32, i32) -> i32
    %97 = "ep2.bitcast"(%96) : (i32) -> i16
    %98 = "ep2.struct_update"(%95, %97) <{index = 5 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i16) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    %99 = "ep2.add"(%11, %12) : (i32, i32) -> i32
    %100 = "ep2.add"(%99, %14) : (i32, i32) -> i32
    %101 = "ep2.add"(%100, %15) : (i32, i32) -> i32
    %102 = "ep2.add"(%101, %34) : (i32, i32) -> i32
    %103 = "ep2.bitcast"(%102) : (i32) -> i64
    %104 = "ep2.struct_update"(%98, %103) <{index = 7 : i64}> : (!ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, i64) -> !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>
    "ep2.update"(%21, %31, %104) : (!ep2.table<i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>, 64>, i16, !ep2.struct<"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64>) -> ()
    %105 = ep2.struct_access %104[2] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i32
    %106 = ep2.struct_access %104[3] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i32
    %107 = "ep2.add"(%105, %106) : (i32, i32) -> i32
    %108 = ep2.struct_access %104[4] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i16
    %109 = "ep2.add"(%107, %108) : (i32, i16) -> i32
    %110 = ep2.struct_access %104[5] : <"lb_DIP_entries_t" : isEvent = false, elementTypes = i64, i64, i32, i32, i16, i16, i32, i64> -> i16
    %111 = "ep2.add"(%109, %110) : (i32, i16) -> i32
    %112 = "ep2.bitcast"(%111) : (i32) -> i64
    %113 = "ep2.struct_update"(%24, %112) <{index = 0 : i64}> : (!ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>, i64) -> !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>
    %114 = "ep2.emit_value"(%78, %113) : (!ep2.buf, !ep2.struct<"lb_fwd_tcp_hdr_t" : isEvent = false, elementTypes = i64, i64, i32>) -> !ep2.buf
    %115 = "ep2.init"(%13, %114) {context_names = ["", ""]} : (!ep2.atom, !ep2.buf) -> !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.buf>
    ep2.return %115 : !ep2.struct<"NET_SEND" : isEvent = true, elementTypes = !ep2.atom, !ep2.buf>
    "ep2.terminate"() : () -> ()
  }
}

