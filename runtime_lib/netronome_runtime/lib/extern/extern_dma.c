// This is the external dma engine that handles dma read/write send/recv  event request and generate request completion it to the processing pipeline
__declspec(export emem scope(global)) int global_start = 0;
#define _NO_IMPORT
#include "nfplib.h"
#undef _NO_IMPORT
#include "prog_hdr.h"
#include "extern_dma.h"

__forceinline void handle_dma_recv()
{
    //// for simulation
    __gpr struct recv_desc_sim_t sim_template_desc;
    __xwrite struct recv_desc_sim_t sim_template_desc_write;

    __xread struct extern_event_param_DMA_RECV_REQ work;
    __gpr struct extern_event_param_DMA_RECV_CMPL next_work;

    __xwrite struct extern_event_param_DMA_RECV_CMPL next_work_w;

    int tmp_seq = 0;
    sim_template_desc.flow_id = 64;
    sim_template_desc.flow_grp = 1;
    sim_template_desc.bump_seq = tmp_seq;
    sim_template_desc.flags = 16;

    for (;;)
    {
        cls_workq_add_thread(WORKQ_ID_DMA_RECV_REQ, &work, sizeof(work));

        // allocate buffer for the descriptor
        if (IF_SIMULATION)
        {
            // assume issue desc dma, and immediately get response, fill in the nic desc pool buffer
            sim_template_desc.bump_seq = tmp_seq;

            sim_template_desc_write = sim_template_desc;
            // Double check whether need sync write
            mem_write64(&sim_template_desc_write, (__mem40 void *)work.nic_buf_addr, sizeof(sim_template_desc));

            local_csr_write(local_csr_mailbox_1, tmp_seq);
            tmp_seq++;

            next_work.data_ptr = (char*)work.nic_buf_addr;
            next_work_w = next_work;
            cls_workq_add_work(WORKQ_ID_DMA_RECV_CMPL, &next_work_w, sizeof(next_work));
        }
        else
        {
            // Currently inflight is 1, need multiple inflight in future.
            // __xread struct dma_pkt_cmd_t tx_cmd[1];
            // __xwrite struct nfp_pcie_dma_cmd dma_cmd[1];
            // // for recv, alloc a a buffer in NIC, and genearte fetch dma command to fetch the data from the host command queue.
            // issue_pkt_dma(DMA_TX_PCIe_QUEUE, &tx_cmd[X], &dma_cmd[X], dma_cmd_word1, dma_cmd_word3, &dma_sig##X)
        }
    }
}

__forceinline void handle_dma_send()
{
}

__forceinline void handle_dma_read_req()
{
}

__forceinline void handle_dma_read_cmpl()
{
}

__forceinline void handle_dma_write_req()
{
    __xread struct extern_event_param_DMA_WRITE_REQ work;
    int temp_seq = 0;
    

     //// for simulation
    for (;;)
    {
        cls_workq_add_thread(WORKQ_ID_DMA_WRITE_REQ, &work, sizeof(work));

        // allocate buffer for the descriptor
        if (IF_SIMULATION)
        {
            local_csr_write(local_csr_mailbox_2, temp_seq);
            temp_seq ++;
        }
        else
        {
        }
    }
}

__forceinline void handle_dma_write_cmpl()
{
}

__forceinline void handle_host_control_signal()
{
    // TODO
}

__forceinline void handle_sim_control_signal()
{
    unsigned int entry_idx;
    int i =0;
    // __xrw struct extern_event_param_DMA_RECV_REQ work;
    __gpr struct extern_event_param_DMA_RECV_REQ work;
    __xwrite struct extern_event_param_DMA_RECV_REQ work_write;
    for (;;)
    {
        work.host_buf_addr = 0;

        entry_idx = allocate_recv_desc_ring_entry();

        work.nic_buf_addr = (uint64_t)(__mem40 void *)&recv_desc_pool[entry_idx];

        //  (uint64_t)((__mem40 void *)&recv_desc_pool[0]) + sizeof(struct recv_desc_sim_t) * entry_idx;
        work_write = work;
        cls_workq_add_work(WORKQ_ID_DMA_RECV_REQ, &work_write, sizeof(work));

        sleep(50);
    }
}

int main(void)
{
    __xrw tmp_xfer;
    uint32_t tmp;
    /////////initialization//////
    init_recv_desc_ring();
    init_recv_event_workq(WORKQ_ID_DMA_RECV_REQ, workq_DMA_RECV_REQ, WORKQ_TYPE_DMA_RECV_REQ, WORKQ_SIZE_DMA_RECV_REQ, 8);
    init_recv_event_workq(WORKQ_ID_DMA_WRITE_REQ, workq_DMA_WRITE_REQ, WORKQ_TYPE_DMA_WRITE_REQ, WORKQ_SIZE_DMA_WRITE_REQ, 8);

    // TODO: Currently we assume 1000 is enough in simulation
    sleep(1000);
    tmp = 1;
    tmp_xfer = tmp;
    mem_write32(&tmp_xfer, &global_start, sizeof(tmp_xfer));
    local_csr_write(local_csr_mailbox_3, 3);

    //// start run /////
    if (ctx() == 0)
    {
        handle_dma_recv();
    }
    else if (ctx() == 1)
    {
        handle_dma_send();
    }
    else if (ctx() == 2)
    {
        handle_dma_read_req();
    }
    else if (ctx() == 3)
    {
        handle_dma_read_cmpl();
    }
    else if (ctx() == 4)
    {
        handle_dma_write_req();
    }
    else if (ctx() == 5)
    {
        handle_dma_write_cmpl();
    }
    else if (ctx() == 6)
    {
        if (IF_GENERATE_FAKE_DMA_RECV_REQ)
        {
            handle_sim_control_signal();
        }
        else
        {
            handle_host_control_signal();
        }
    }
}
