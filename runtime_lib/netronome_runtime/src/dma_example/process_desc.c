

#include "nfplib.h"
#include "context.h"
#include "eventq_config.h"

int main(void)
{
    int tmp;
    __xread struct event_param_USER_EVENT1 work_r;
    __xwrite struct event_param_USER_EVENT1 work_w;

    __mem40 char *ctx_ptr;
    unsigned int context_idx;

    __gpr struct recv_desc_t test_desc;
    __xread struct recv_desc_t test_desc_ref;

    // TODO: When there are multiple threads/core running the the same stage, initialization should only happen once. Need to implement the synchroniaton primitive here.

    // initial event queue for this pipeline stage. the queue can be instanted at different memory hierarchy (mem/ctm/cls)
    
    init_recv_event_workq(WORKQ_ID_USER_EVENT1, workq_USER_EVENT1, WORKQ_TYPE_USER_EVENT1, WORKQ_SIZE_USER_EVENT1, 8);

    for (;;)
    {

        // handler USER_EVENT1:receive_payload_1 (context ctx, long addr, buf payload)
        // get an event from the work_USER_EVENT1 queue.

        // Change based on memory type:
        // mem_workq_add_thread / ctm_ring_get /  cls_workq_add_thread
        cls_workq_add_thread(WORKQ_ID_USER_EVENT1, &work_r, sizeof(work_r));

        // get reference PTR for descriptor struct inside conext
        context_idx = work_r.context_idx;
        ctx_ptr = (__mem40 void *)&context_chain1_pool[context_idx];

        // TODO: Double check whether need sync write
        work_w = work_r;
        mem_write32(&(work_w.param), ctx_ptr + offsetof(struct recv_desc_t, flow_grp), sizeof(work_w.param));


        // below is for debug purpose
        mem_read64(&(test_desc_ref), (__mem40 void *)ctx_ptr, sizeof(struct recv_desc_t));
        local_csr_write(local_csr_mailbox_0, test_desc_ref.bump_seq);
        local_csr_write(local_csr_mailbox_1, test_desc_ref.flow_grp);

    }

    return 0;
}