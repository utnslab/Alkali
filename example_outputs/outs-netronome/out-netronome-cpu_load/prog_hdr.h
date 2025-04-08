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

__packed struct coremap_t {
	uint16_t f0;
	uint16_t f1;
	uint16_t f2;
	uint16_t f3;
};

__packed struct context_chain_1_t {
};

__packed struct event_param_NET_RECV {
	struct __buf_t f0;
	struct recv_meta_t meta;
	__shared __cls struct context_chain_1_t* ctx;
};

__packed struct event_param_LOAD_TABLE_ADD {
	uint16_t f0;
	uint16_t f1;
	__shared __cls struct context_chain_1_t* ctx;
};

#define WORKQ_SIZE_LOAD_TABLE_ADD 128
#define WORKQ_TYPE_LOAD_TABLE_ADD MEM_TYEP_CLS
#define WORKQ_ID_LOAD_TABLE_ADD_1 10
CLS_WORKQ_DECLARE(workq_LOAD_TABLE_ADD_1, WORKQ_SIZE_LOAD_TABLE_ADD);

__packed struct table_i32_coremap_t_16_t {
	struct coremap_t table[16];
};
__export __shared __cls struct table_i32_coremap_t_16_t service_load;

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
