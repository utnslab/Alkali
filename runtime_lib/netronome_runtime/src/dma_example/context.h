#ifndef _CONTEXT_H_
#define _CONTEXT_H_

#include "nfplib.h"
#include "struct.h"
#include <nfp/mem_ring.h>

// the pool size is determined by how many inflight packets are allowed in the NIC.
#define CONTEXT_CHAIN1_POOL_NUM 2048

// the context is defined as per-chain context
struct context_chain1_t
{
    struct recv_desc_t desc;
};
// If the desc in context is a reference pointer, it should look like something like this:
// struct context_chain1_t
// {
//     char* desc_ptr;
// };

// the conext is allocated/deallocated from the conext buffer pool. Can be EMEM/CTM/CLS
EMEM_CONTEXTQ_DECLARE(context_chain1_t, context_chain1_pool, CONTEXT_CHAIN1_POOL_NUM);

// the MEM ring sits in EMEM and stores the indexs for each CONTEXT queue enrty.
// TODO: Whether this could use CTM/CLS
MEM_RING_INIT(context_chain1_ring, CONTEXT_CHAIN1_POOL_NUM);

__forceinline void init_context_chain1_ring()
{
    unsigned int idx;
    unsigned int rnum, raddr_hi;
    unsigned int init_range;

    // The only purpose for override the init_range for simulation is to reduce the simulation time

    if (IF_SIMULATION)
        init_range = 10;
    else
        init_range = CONTEXT_CHAIN1_POOL_NUM;

    if (ctx() == 0) // initializtion only happend on the first thread in the core.
    {
        rnum = MEM_RING_GET_NUM(context_chain1_ring);
        raddr_hi = MEM_RING_GET_MEMADDR(context_chain1_ring);

        // Fill in the context buffer pool index
        for (idx = 1; idx < init_range; idx++)
        {
            mem_ring_journal_fast(rnum, raddr_hi, idx);
        }
    }
}

// This is a thread safe data structure.
__forceinline unsigned int allocate_context_chain1_ring_entry()
{
    __xread unsigned int context_idx;
    unsigned int rnum, raddr_hi;

    rnum = MEM_RING_GET_NUM(context_chain1_ring);
    raddr_hi = MEM_RING_GET_MEMADDR(context_chain1_ring);
    while (mem_ring_get(rnum, raddr_hi, &context_idx, sizeof(context_idx)) != 0)
    {
    }

    return context_idx;
}

#endif