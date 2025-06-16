
#include "nfplib.h"
#include "context.h"
#include "eventq_config.h"
#include "extern/extern_dma.h"
#include "struct.h"
// handler DMA_RECV_CMPL:receive_desc (context ctx, buf recv_buf) {
//     Desc_Hdr desc;
//     recv_buf.extract(desc); # Parse descriptor header
//     ctx.desc = desc; # Save Context
//     generate DMA_READ_REQ:receive_payload_1 {ctx, desc.data_addr, 100};
// }

int main(void)
{
    __xrw struct event_param_DMA_RECV_CMPL work;
    int extract_offset;
    __mem40 char *ctx_ptr;
    __mem40 struct recv_desc_t * desc_ptr;
    unsigned int context_idx;

    __gpr struct event_param_USER_EVENT1 next_work;
    __xwrite struct event_param_USER_EVENT1 next_work_ref;

    // TODO: When there are multiple threads/core running the the same stage, initialization should only happen once. Need to implement the synchroniaton primitive here.

    // If this is the first stage of the pipeline chain, initialize the context chain ring.
    init_context_chain1_ring();

    // initial event queue for this pipeline stage. the queue can be instanted at different memory hierarchy (mem/ctm/cls)
    init_recv_event_workq(WORKQ_ID_DMA_RECV_CMPL, workq_DMA_RECV_CMPL, WORKQ_TYPE_DMA_RECV_CMPL, WORKQ_SIZE_DMA_RECV_CMPL, 8);

    for (;;)
    {

        // DMA_RECV_CMPL:receive_desc (context ctx, buf recv_buf)
        // get an event from the work_DMA_RECV_CMPL queue.

        // Change based on memory type:
        // mem_workq_add_thread / ctm_ring_get /  cls_workq_add_thread
        cls_workq_add_thread(WORKQ_ID_DMA_RECV_CMPL, &work, sizeof(work));

        // Since this is the first stage of the chain, allocate an entry from the context_chain1_ring, the entry is the offset for this context in the context_chain1_pool.
        context_idx = allocate_context_chain1_ring_entry();
        // If it is not the first stage:
        // context_idx = work.context_idx;

        // get reference PTR for descriptor struct inside conext
        ctx_ptr = (__mem40 void *)&context_chain1_pool[context_idx];

        // data.extract(desc);
        desc_ptr = work.data_ptr;
        
        // ctx.desc = desc;
        // TODO: pass by value or pass by reference, here is pass by value (ua_memcpy/bulk_memcpy)
        bulk_memcpy(ctx_ptr, desc_ptr, sizeof(struct recv_desc_t));

        // constrcut new event command for next stage
        next_work.context_idx = context_idx;
        next_work.param = 100;

        // fire this event to the next stage event queue
        next_work_ref = next_work;
        cls_workq_add_work(WORKQ_ID_USER_EVENT1, &next_work_ref, sizeof(next_work));
    }

    return 0;
}