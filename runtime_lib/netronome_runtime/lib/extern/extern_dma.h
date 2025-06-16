
#ifndef _EXTERN_DMA_H_
#define _EXTERN_DMA_H_

#include "nfplib.h"
#include "prog_hdr.h"

// 32B received descriptor
// Should not defined here, only used for sim
__packed struct recv_desc_sim_t
{
    uint32_t flow_id;
    uint32_t bump_seq;
    uint32_t flags;
    uint32_t flow_grp;
};


// how many send recv queue, usually one pair of queue per application
#define DMA_SEND_RECV_QUEUE_COUNT 1

// event for fetching a descriptor from the host, described as receive request
struct extern_event_param_DMA_RECV_REQ
{
    uint64_t host_buf_addr;
    uint64_t nic_buf_addr;
};

#define WORKQ_ID_DMA_RECV_REQ 0
#define WORKQ_SIZE_DMA_RECV_REQ 256
#define WORKQ_TYPE_DMA_RECV_REQ MEM_TYEP_CLS

CLS_WORKQ_DECLARE(workq_DMA_RECV_REQ, WORKQ_SIZE_DMA_RECV_REQ);

__packed struct extern_event_param_DMA_RECV_CMPL {
	char* data_ptr;
  char* ctx;
};

#define WORKQ_ID_DMA_RECV_CMPL 9
#define WORKQ_SIZE_DMA_RECV_CMPL 256
#define WORKQ_TYPE_DMA_RECV_CMPL MEM_TYEP_CLS

CLS_WORKQ_DECLARE(workq_DMA_RECV_CMPL, WORKQ_SIZE_DMA_RECV_CMPL);

__packed struct extern_event_param_DMA_WRITE_REQ {
  uint32_t addr;
  uint32_t size;
};

#define WORKQ_ID_DMA_WRITE_REQ 1
#define WORKQ_SIZE_DMA_WRITE_REQ 256
#define WORKQ_TYPE_DMA_WRITE_REQ MEM_TYEP_CLS

CLS_WORKQ_DECLARE(workq_DMA_WRITE_REQ, WORKQ_SIZE_DMA_WRITE_REQ);

// Send/Recv queue register array.
// configured by host -- not used in simulator ()
// the struct that describe the send recv queue start address, send, len, and the entry size (desc_size) inside the queue.
struct dma_send_recv_queue_reg_t
{
    uint32_t index;
    uint32_t desc_size; // Send/Recv size unit
    uint32_t len;
    uint64_t queue_base;
};

__shared __lmem struct dma_send_recv_queue_reg_t dma_send_queue_reg[DMA_SEND_RECV_QUEUE_COUNT];
__shared __lmem struct dma_send_recv_queue_reg_t dma_send_queue_reg[DMA_SEND_RECV_QUEUE_COUNT];

#define RECV_DESC_SIZE 16 // in byte
#define RECV_DESC_POOL_NUM 2048
// Descriptor pool that actually stores the descriptor
EMEM_CONTEXTQ_DECLARE(recv_desc_sim_t, recv_desc_pool, RECV_DESC_POOL_NUM);
MEM_RING_INIT(recv_desc_ring, RECV_DESC_POOL_NUM);

__forceinline void init_recv_desc_ring()
{
    unsigned int idx;
    unsigned int rnum, raddr_hi;
    unsigned int init_range;

    if (IF_SIMULATION)
        init_range = 10;
    else
        init_range = RECV_DESC_POOL_NUM;

    if (ctx() == 0) // initializtion only happend on the first thread in the core.
    {
        rnum = MEM_RING_GET_NUM(recv_desc_ring);
        raddr_hi = MEM_RING_GET_MEMADDR(recv_desc_ring);
        // Fill in the context buffer pool index
        for (idx = 1; idx < init_range; idx++)
        {
            mem_ring_journal_fast(rnum, raddr_hi, idx);
        }
    }
}

__forceinline unsigned int allocate_recv_desc_ring_entry()
{
    __xread unsigned int entry_idx;
    unsigned int rnum, raddr_hi;

    rnum = MEM_RING_GET_NUM(recv_desc_ring);
    raddr_hi = MEM_RING_GET_MEMADDR(recv_desc_ring);
    while (mem_ring_get(rnum, raddr_hi, &entry_idx, sizeof(entry_idx)) != 0)
    {
    }

    return entry_idx;
}

#endif
