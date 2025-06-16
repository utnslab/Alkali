#ifndef _PROG_HDR_H_
#define _PROG_HDR_H_

#include "nfplib.h"
#include <nfp/mem_ring.h>
#include "extern/extern_net_meta.h"

__packed struct __buf_t {
	char* buf;
	unsigned offs;
	unsigned sz;
};

__packed struct flow_state_t {
	uint32_t f0;
	uint16_t f1;
	uint16_t f2;
	uint32_t f3;
	uint32_t f4;
	uint32_t f5;
	uint32_t f6;
	uint32_t f7;
	uint32_t f8;
	uint16_t f9;
	uint16_t f10;
	uint32_t f11;
	uint32_t f12;
	uint8_t f13;
	uint8_t f14;
	uint16_t f15;
	uint16_t f16;
	uint16_t f17;
	uint16_t f18;
	uint16_t f19;
	uint32_t f20;
	uint32_t f21;
	uint32_t f22;
};

__packed struct context_chain_1_t {
};

__packed struct event_param_NET_RECV {
	struct __buf_t f0;
	struct recv_meta_t meta;
	__shared __cls struct context_chain_1_t* ctx;
};

__packed struct event_param_NET_SEND {
	struct __buf_t f0;
	struct send_meta_t meta;
	__shared __cls struct context_chain_1_t* ctx;
};

#define SIZE_table_i16_flow_state_t_16_1t 64   // should have this defined for each table
__packed struct table_i16_flow_state_t_16_1t {
	struct flow_state_t table[SIZE_table_i16_flow_state_t_16_1t]; // now the table size can more than 16
};

#define SIZE_table_i16_flow_state_t_16_2t 32   // should have this defined for each table
__packed struct table_i16_flow_state_t_16_2t {
	struct flow_state_t table[SIZE_table_i16_flow_state_t_16_2t]; // now the table size can more than 16
};

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
