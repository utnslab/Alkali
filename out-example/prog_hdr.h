#ifndef _PROG_HDR_H_
#define _PROG_HDR_H_

#include "nfplib.h"
#include <nfp/mem_ring.h>

__packed struct __wrapper_arg_t {
	int32_t f0;
	char* f1;
};

__packed struct context_chain_1_t {
	int32_t f0;
	int32_t f1;
	int32_t f2;
	int32_t f3;
	int32_t ctx_id;
};

__packed struct recv_desc_t {
	int32_t f0;
	int32_t f1;
	int32_t f2;
	int32_t f3;
};

__packed struct event_param_DMA_RECV_CMPL {
	char* f0;
	struct context_chain_1_t* ctx;
};

__packed struct event_param_USER_EVENT1 {
	int32_t f0;
	struct context_chain_1_t* ctx;
};

#define WORKQ_SIZE_USER_EVENT1 256
#define WORKQ_ID_USER_EVENT1 10
#define WORKQ_TYPE_USER_EVENT1 MEM_TYEP_CLS
CLS_WORKQ_DECLARE(workq_USER_EVENT1, WORKQ_SIZE_USER_EVENT1);

#define WORKQ_SIZE_DMA_RECV_CMPL 256
#define WORKQ_ID_DMA_RECV_CMPL 11
#define WORKQ_TYPE_DMA_RECV_CMPL MEM_TYEP_CLS
CLS_WORKQ_DECLARE(workq_DMA_RECV_CMPL, WORKQ_SIZE_DMA_RECV_CMPL);

EMEM_CONTEXTQ_DECLARE(context_chain_1_t, context_chain_pool, 2048);
MEM_RING_INIT(context_chain_ring, 2048);

__forceinline static void init_context_chain_ring() {
	unsigned int idx, rnum, raddr_hi, init_range;
	init_range = IF_SIMULATION ? 10 : 2048;
	if (ctx() == 0) {
		rnum = MEM_RING_GET_NUM(context_chain_ring);
		raddr_hi = MEM_RING_GET_MEMADDR(context_chain_ring);
		for (idx=1; idx<init_range; idx++) mem_ring_journal_fast(rnum, raddr_hi, idx);
	}
	for (idx=0; idx<2048; ++idx) context_chain_pool[idx].ctx_id = idx;
}

__forceinline static struct context_chain_1_t* alloc_context_chain_ring_entry() {
	__xread unsigned int context_idx;
	unsigned int rnum, raddr_hi;
	rnum = MEM_RING_GET_NUM(context_chain_ring);
	raddr_hi = MEM_RING_GET_MEMADDR(context_chain_ring);
	while (mem_ring_get(rnum, raddr_hi, &context_idx, sizeof(context_idx)) != 0);
	return &context_chain_pool[context_idx];
}

#endif
