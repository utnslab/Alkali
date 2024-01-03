#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

struct recv_desc_t _loc_buf_0;
__xrw struct recv_desc_t _loc_buf_0_xfer;
__declspec(aligned(4)) struct event_param_DMA_RECV_CMPL work;
__xrw struct event_param_DMA_RECV_CMPL work_ref;
__declspec(aligned(4)) struct event_param_USER_EVENT1 next_work_USER_EVENT1;
__xrw struct event_param_USER_EVENT1 next_work_ref_USER_EVENT1;

__forceinline
void __event___handler_DMA_RECV_CMPL_receive_desc() {
  int32_t v1;
  int64_t v2;
  __declspec(aligned(4)) struct event_param_DMA_RECV_CMPL* v3;
  __xrw struct event_param_DMA_RECV_CMPL* v4;
  struct context_chain_1_t* v5;
  struct context_chain_1_t* v6;
  struct __buf_t v7;
  struct recv_desc_t* v8;
  __xrw struct recv_desc_t* v9;
  __declspec(aligned(4)) struct event_param_USER_EVENT1* v10;
  __xrw struct event_param_USER_EVENT1* v11;
  v1 = 1;
  v2 = 100;
  v3 = &work;
  v4 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_DMA_RECV_CMPL, v4, sizeof(*v4));
  *(v3) = *(v4);
  v5 = alloc_context_chain_ring_entry();
  v3->ctx = v5;
  v6 = v3->ctx;
  v7 = v3->f0;
  v8 = &_loc_buf_0;
  v9 = &_loc_buf_0_xfer;
  mem_read32(&v9->f0, v7.buf + v7.offs, 16);
  v7.offs += 16;
  *(v8) = *(v9);
  v6->f0 = *v8;
  v10 = &next_work_USER_EVENT1;
  v10->ctx = v6;
  v10->f0 = v2;
  v11 = &next_work_ref_USER_EVENT1;
  *(v11) = *(v10);
  cls_workq_add_work(WORKQ_ID_USER_EVENT1, v11, sizeof(*v11));
  return;
}


int main(void) {
	init_context_chain_ring();
	init_recv_event_workq(WORKQ_ID_DMA_RECV_CMPL, workq_DMA_RECV_CMPL, WORKQ_TYPE_DMA_RECV_CMPL, WORKQ_SIZE_DMA_RECV_CMPL, 8);
	for (;;) {
		__event___handler_DMA_RECV_CMPL_receive_desc();
	}
}
