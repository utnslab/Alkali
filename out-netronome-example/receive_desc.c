#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

struct recv_desc_t _loc_buf_0;
__xrw struct recv_desc_t _loc_buf_0_xfer;
unsigned extr_offset = 0;
unsigned emit_offset = 0;
__declspec(aligned(4)) struct event_param_DMA_RECV_CMPL work;
__xrw struct event_param_DMA_RECV_CMPL work_ref;
struct __wrapper_arg_t wrap_in;
__declspec(aligned(4)) struct event_param_USER_EVENT1 next_work;
__xrw struct event_param_USER_EVENT1 next_work_ref;
struct __wrapper_arg_t wrap_out;

__forceinline
void __event___handler_DMA_RECV_CMPL_receive_desc(struct __wrapper_arg_t* v1, struct __wrapper_arg_t* v2) {
  int32_t v3;
  int64_t v4;
  struct event_param_DMA_RECV_CMPL* v5;
  struct context_chain_1_t* v6;
  char* v7;
  struct recv_desc_t* v8;
  __xrw struct recv_desc_t* v9;
  int32_t v10;
  int32_t v11;
  int32_t v12;
  int32_t v13;
  struct event_param_USER_EVENT1* v14;
  v3 = 1;
  v4 = 100;
  v5 = v1->f1;
  v6 = v5->ctx;
  v7 = v5->f0;
  v8 = &_loc_buf_0;
  v9 = &_loc_buf_0_xfer;
  mem_read32(&v9->f0, v7+extr_offset, 16);
  extr_offset += 16;
  *(v8) = *(v9);
  v10 = v8->f0;
  v6->f0 = v10;
  v11 = v8->f1;
  v6->f1 = v11;
  v12 = v8->f2;
  v6->f2 = v12;
  v13 = v8->f3;
  v6->f3 = v13;
  v14 = &next_work;
  v2->f0 = v3;
  v2->f1 = v14;
  v14->ctx = v6;
  v14->f0 = v4;
  return;
}


int main(void) {
	init_context_chain_ring();
	init_recv_event_workq(WORKQ_ID_DMA_RECV_CMPL, workq_DMA_RECV_CMPL, WORKQ_TYPE_DMA_RECV_CMPL, WORKQ_SIZE_DMA_RECV_CMPL, 8);
	for (;;) {
		cls_workq_add_thread(WORKQ_ID_DMA_RECV_CMPL, &work_ref, sizeof(work_ref));
		work = work_ref;
		wrap_in.f1 = &work;
		work.ctx = alloc_context_chain_ring_entry();
		__event___handler_DMA_RECV_CMPL_receive_desc(&wrap_in, &wrap_out);
		next_work_ref = next_work;
		cls_workq_add_work(WORKQ_ID_USER_EVENT1, &next_work_ref, sizeof(next_work_ref));
	}
}
