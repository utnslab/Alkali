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
};

__packed struct udp_header_t {
	uint16_t f0;
	uint16_t f1;
	uint16_t f2;
	uint16_t f3;
};

__packed struct rpc_header_t {
	uint32_t f0;
	uint32_t f1;
	uint32_t f2;
	uint32_t f3;
};

__packed struct agg_t {
	uint32_t f0;
	uint32_t f1;
};

__packed struct context_chain_1_t {
	struct ip_header_t f0;
	struct eth_header_t f1;
};

__packed struct event_param_NET_RECV {
	struct __buf_t f0;
	struct recv_meta_t meta;
	__shared __cls struct context_chain_1_t* ctx;
};

__packed struct event_param_MSG_REASSEMBLE {
	struct __buf_t f0;
	struct rpc_header_t f1;
	__shared __cls struct context_chain_1_t* ctx;
};

__packed struct event_param_NET_SEND {
	struct __buf_t f0;
	struct send_meta_t meta;
	__shared __cls struct context_chain_1_t* ctx;
};

#define WORKQ_SIZE_MSG_REASSEMBLE 1024
#define WORKQ_TYPE_MSG_REASSEMBLE MEM_TYEP_CLS
#define WORKQ_ID_MSG_REASSEMBLE_1 10
CLS_WORKQ_DECLARE(workq_MSG_REASSEMBLE_1, WORKQ_SIZE_MSG_REASSEMBLE);

__packed struct table_i16___buf_t_16_t {
	struct __buf_t table[16];
};
__shared __lmem struct table_i16___buf_t_16_t table_9;

__packed struct table_i16_agg_t_16_t {
	struct agg_t table[16];
};
__shared __lmem struct table_i16_agg_t_16_t table_8;

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
