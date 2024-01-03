#ifndef _PROG_HDR_H_
#define _PROG_HDR_H_

#include "nfplib.h"
#include <nfp/mem_ring.h>
#include "extern/extern_net_meta.h"

typedef __packed struct __int48 {
	int8_t storage[6];
} int48_t;

__packed struct __wrapper_arg_t {
	int32_t f0;
	char* f1;
};

__packed struct __buf_t {
	char* buf;
	unsigned offs;
	unsigned sz;
};

__packed struct pkt_info_t {
	int32_t f0;
	int32_t f1;
	int32_t f2;
};

__packed struct eth_header_t {
	int48_t f0;
	int48_t f1;
	int16_t f2;
	int8_t pad0[2];
};

__packed struct ip_header_t {
	int16_t f0;
	int16_t f1;
	int16_t f2;
	int16_t f3;
	int16_t f4;
	int16_t f5;
	int32_t f6;
	int32_t f7;
	int32_t f8;
};

__packed struct tcp_header_t {
	int16_t f0;
	int16_t f1;
	int32_t f2;
	int32_t f3;
	int8_t f4;
	int8_t f5;
	int16_t f6;
	int16_t f7;
	int16_t f8;
};

__packed struct flow_state_t {
	int32_t f0;
	int16_t f1;
	int16_t f2;
	int32_t f3;
	int32_t f4;
	int32_t f5;
	int32_t f6;
	int32_t f7;
	int32_t f8;
};

__packed struct dma_write_cmd_t {
	int32_t f0;
	int32_t f1;
};

__packed struct ack_info_t {
	int32_t f0;
	int32_t f1;
	int32_t f2;
};

__packed struct context_chain_1_t {
	struct eth_header_t f0;
	int8_t pad0[2];
	struct ip_header_t f1;
	struct tcp_header_t f2;
	struct __buf_t f3;
	int32_t ctx_id;
};

__packed struct event_param_NET_RECV {
	struct __buf_t f0;
	struct recv_meta_t meta;
	struct context_chain_1_t* ctx;
};

__packed struct event_param_OoO_DETECT {
	struct pkt_info_t f0;
	struct context_chain_1_t* ctx;
};

__packed struct event_param_DMA_WRITE_REQ {
	struct __buf_t f0;
	struct dma_write_cmd_t f1;
	struct context_chain_1_t* ctx;
};

__packed struct event_param_ACK_GEN {
	struct ack_info_t f0;
	struct context_chain_1_t* ctx;
};

__packed struct event_param_NET_SEND {
	struct __buf_t f0;
	struct send_meta_t meta;
	struct context_chain_1_t* ctx;
};

#define WORKQ_SIZE_ACK_GEN 256
#define WORKQ_ID_ACK_GEN 10
#define WORKQ_TYPE_ACK_GEN MEM_TYEP_CLS
CLS_WORKQ_DECLARE(workq_ACK_GEN, WORKQ_SIZE_ACK_GEN);

#define WORKQ_SIZE_OoO_DETECT 256
#define WORKQ_ID_OoO_DETECT 11
#define WORKQ_TYPE_OoO_DETECT MEM_TYEP_CLS
CLS_WORKQ_DECLARE(workq_OoO_DETECT, WORKQ_SIZE_OoO_DETECT);

__packed struct table_i16_flow_state_t_16_t {
	struct flow_state_t table[16];
};
__shared struct table_i16_flow_state_t_16_t table_7;

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
	for (idx=0; idx<init_range; ++idx) context_chain_pool[idx].ctx_id = idx;
}

__forceinline static struct context_chain_1_t* alloc_context_chain_ring_entry() {
	__xread unsigned int context_idx;
	unsigned int rnum, raddr_hi;
	rnum = MEM_RING_GET_NUM(context_chain_ring);
	raddr_hi = MEM_RING_GET_MEMADDR(context_chain_ring);
	while (mem_ring_get(rnum, raddr_hi, &context_idx, sizeof(context_idx)) != 0);
	return &context_chain_pool[context_idx];
}

__forceinline static struct __buf_t alloc_packet_buf() {
	struct __buf_t buf;
	buf.buf = alloc_packet_buffer();
	buf.offs = 0;
	buf.sz = 0;
	return buf;
}

#endif
