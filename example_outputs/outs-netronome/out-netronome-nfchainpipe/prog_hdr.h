#ifndef _PROG_HDR_H_
#define _PROG_HDR_H_

#include "nfplib.h"
#include <nfp/mem_ring.h>
#include "extern/extern_net_meta.h"

typedef __packed struct __int48 {
	uint8_t storage[6];
} uint48_t;

__packed struct __buf_t {
	char* buf;
	unsigned offs;
	unsigned sz;
};

__packed struct eth_header_t {
	uint48_t f0;
	uint48_t f1;
	uint16_t f2;
	uint8_t pad0[2];
};

__packed struct ip_header_t {
	uint16_t f0;
	uint16_t f1;
	uint16_t f2;
	uint16_t f3;
	uint16_t f4;
	uint16_t f5;
	uint32_t f6;
	uint32_t f7;
	uint32_t f8;
};

__packed struct tcp_header_t {
	uint16_t f0;
	uint16_t f1;
	uint32_t f2;
	uint32_t f3;
	uint8_t f4;
	uint8_t f5;
	uint16_t f6;
	uint16_t f7;
	uint16_t f8;
};

__packed struct firewall_ip_entries_t {
	uint32_t f0;
	uint32_t f1;
	uint32_t f2;
	uint32_t f3;
	uint32_t f4;
};

__packed struct firewall_tcpport_entries_t {
	uint32_t f0;
	uint32_t f1;
	uint32_t f2;
	uint32_t f3;
	uint32_t f4;
};

__packed struct priority_entries_t {
	uint32_t f0;
	uint32_t f1;
};

__packed struct firewall_meta_header_t {
	uint32_t f0;
	uint32_t f1;
	uint32_t f2;
};

__packed struct flow_tracker_t {
	uint32_t f0;
	uint32_t f1;
};

__packed struct tcp_tracker_t {
	uint32_t f0;
	uint32_t f1;
};

__packed struct ip_tracker_t {
	uint32_t f0;
	uint32_t f1;
};

__packed struct err_tracker_t {
	uint32_t f0;
	uint32_t f1;
};

__packed struct lb_DIP_entries_t {
	uint64_t f0;
	uint64_t f1;
	uint32_t f2;
	uint32_t f3;
	uint16_t f4;
	uint16_t f5;
	uint32_t f6;
};

__packed struct lb_fwd_tcp_hdr_t {
	uint64_t f0;
	uint64_t f1;
	uint32_t f2;
};

__packed struct context_chain_1_t {
};

__packed struct event_param_NET_RECV {
	struct __buf_t f0;
	uint8_t pad0[4];
	struct recv_meta_t meta;
	__shared __cls struct context_chain_1_t* ctx;
};

__packed struct event_param_OoO_DETECT1 {
	struct __buf_t f0;
	uint32_t f1;
	uint32_t f2;
	uint16_t f3;
	uint8_t pad0[2];
	__shared __cls struct context_chain_1_t* ctx;
};

__packed struct event_param_OoO_DETECT2 {
	struct __buf_t f0;
	uint32_t f1;
	__shared __cls struct context_chain_1_t* ctx;
};

__packed struct event_param_NET_SEND {
	struct __buf_t f0;
	uint8_t pad0[4];
	struct send_meta_t meta;
	__shared __cls struct context_chain_1_t* ctx;
};

#define WORKQ_SIZE_OoO_DETECT2 2048
#define WORKQ_TYPE_OoO_DETECT2 MEM_TYEP_CLS
#define WORKQ_ID_OoO_DETECT2_1 5
CLS_WORKQ_DECLARE(workq_OoO_DETECT2_1, WORKQ_SIZE_OoO_DETECT2);

#define WORKQ_ID_OoO_DETECT2_2 6
CLS_WORKQ_DECLARE(workq_OoO_DETECT2_2, WORKQ_SIZE_OoO_DETECT2);

#define WORKQ_ID_OoO_DETECT2_3 7
CLS_WORKQ_DECLARE(workq_OoO_DETECT2_3, WORKQ_SIZE_OoO_DETECT2);

#define WORKQ_ID_OoO_DETECT2_4 8
CLS_WORKQ_DECLARE(workq_OoO_DETECT2_4, WORKQ_SIZE_OoO_DETECT2);

#define WORKQ_SIZE_OoO_DETECT1 4096
#define WORKQ_TYPE_OoO_DETECT1 MEM_TYEP_CLS
#define WORKQ_ID_OoO_DETECT1_1 9
CLS_WORKQ_DECLARE(workq_OoO_DETECT1_1, WORKQ_SIZE_OoO_DETECT1);

#define WORKQ_ID_OoO_DETECT1_2 10
CLS_WORKQ_DECLARE(workq_OoO_DETECT1_2, WORKQ_SIZE_OoO_DETECT1);

#define WORKQ_ID_OoO_DETECT1_3 11
CLS_WORKQ_DECLARE(workq_OoO_DETECT1_3, WORKQ_SIZE_OoO_DETECT1);

#define WORKQ_ID_OoO_DETECT1_4 12
CLS_WORKQ_DECLARE(workq_OoO_DETECT1_4, WORKQ_SIZE_OoO_DETECT1);

__packed struct table_i32_lb_DIP_entries_t_64_t {
	struct lb_DIP_entries_t table[64];
};
__shared __lmem struct table_i32_lb_DIP_entries_t_64_t lb_table;
__shared __lmem struct flowht_entry_t lb_table_index[64];

__packed struct table_i32_err_tracker_t_64_t {
	struct err_tracker_t table[64];
};
__export __shared __cls struct table_i32_err_tracker_t_64_t err_tracker_table;
__shared __lmem struct flowht_entry_t err_tracker_table_index[64];

__packed struct table_i32_lb_fwd_tcp_hdr_t_64_t {
	struct lb_fwd_tcp_hdr_t table[64];
};
__export __shared __cls struct table_i32_lb_fwd_tcp_hdr_t_64_t lb_fwd_table;
__shared __lmem struct flowht_entry_t lb_fwd_table_index[64];

__packed struct table_i32_tcp_tracker_t_64_t {
	struct tcp_tracker_t table[64];
};
__shared __lmem struct table_i32_tcp_tracker_t_64_t tcp_tracker_table;
__shared __lmem struct flowht_entry_t tcp_tracker_table_index[64];

__packed struct table_i32_priority_entries_t_64_t {
	struct priority_entries_t table[64];
};
__export __shared __cls struct table_i32_priority_entries_t_64_t priority_table;
__shared __lmem struct flowht_entry_t priority_table_index[64];

__packed struct table_i32_ip_tracker_t_64_t {
	struct ip_tracker_t table[64];
};
__export __shared __cls struct table_i32_ip_tracker_t_64_t ip_tracker_table;
__shared __lmem struct flowht_entry_t ip_tracker_table_index[64];

__packed struct table_i16_firewall_tcpport_entries_t_64_t {
	struct firewall_tcpport_entries_t table[64];
};
__export __shared __cls struct table_i16_firewall_tcpport_entries_t_64_t firewall_tcpport_table;
__shared __lmem struct flowht_entry_t firewall_tcpport_table_index[64];

__packed struct table_i32_flow_tracker_t_64_t {
	struct flow_tracker_t table[64];
};
__shared __lmem struct table_i32_flow_tracker_t_64_t flow_tracker_table;
__shared __lmem struct flowht_entry_t flow_tracker_table_index[64];

__packed struct table_i32_firewall_ip_entries_t_64_t {
	struct firewall_ip_entries_t table[64];
};
__shared __lmem struct table_i32_firewall_ip_entries_t_64_t firewall_ip_table;
__shared __lmem struct flowht_entry_t firewall_ip_table_index[64];

CLS_CONTEXTQ_DECLARE(context_chain_1_t, context_chain_pool, 128);
#ifdef DO_CTXQ_INIT
__export __shared __cls int context_chain_ring_qHead = 0;
#else
__import __shared __cls int context_chain_ring_qHead;
#endif

__forceinline static __shared __cls struct context_chain_1_t* alloc_context_chain_ring_entry() {
	__xrw int context_idx = 1;
	cls_test_add(&context_idx, &context_chain_ring_qHead, sizeof(context_idx));
	return &context_chain_pool[context_idx & 127];
}

__forceinline static struct __buf_t alloc_packet_buf() {
	struct __buf_t buf;
	buf.buf = alloc_packet_buffer();
	buf.offs = 0;
	buf.sz = 0;
	return buf;
}

__forceinline static int hash(int x) {
	return x;
}

#endif
