#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

struct recv_desc_t _loc_buf_0;
__xrw struct recv_desc_t _loc_buf_0_xfer;
__declspec(aligned(4)) struct event_param_DMA_RECV_CMPL work;
__xrw struct event_param_DMA_RECV_CMPL work_ref;
struct __wrapper_arg_t wrap_in;
__declspec(aligned(4)) struct event_param_USER_EVENT1 next_work_USER_EVENT1;
__xrw struct event_param_USER_EVENT1 next_work_ref_USER_EVENT1;
__declspec(aligned(4)) struct event_param_USER_EVENT2 next_work_USER_EVENT2;
__xrw struct event_param_USER_EVENT2 next_work_ref_USER_EVENT2;
struct __wrapper_arg_t wrap_out;

__forceinline
void __event___handler_DMA_RECV_CMPL_receive_desc(struct __wrapper_arg_t* v1, struct __wrapper_arg_t* v2) {
  int64_t v3;
  int32_t v4;
  int64_t v5;
  int32_t v6;
  struct event_param_DMA_RECV_CMPL* v7;
  struct context_chain_1_t* v8;
  struct __buf_t v9;
  struct recv_desc_t* v10;
  __xrw struct recv_desc_t* v11;
  int32_t v12;
  int32_t v13;
  int32_t v14;
  int32_t v15;
  bool v16;
  struct event_param_USER_EVENT1* v17;
  struct event_param_USER_EVENT2* v18;
  v3 = 1;
  v4 = 1;
  v5 = 100;
  v6 = 2;
  v7 = v1->f1;
  v8 = v7->ctx;
  v9 = v7->f0;
  v10 = &_loc_buf_0;
  v11 = &_loc_buf_0_xfer;
  mem_read32(&v11->f0, v9+extr_offset, 16);
  extr_offset += 16;
  *(v10) = *(v11);
  v12 = v10->f0;
  v8->f0 = v12;
  v13 = v10->f1;
  v8->f1 = v13;
  v14 = v10->f2;
  v8->f2 = v14;
  v15 = v10->f3;
  v8->f3 = v15;
  v16 = v14 == v3;
  if (v16) {
    v17 = &next_work_USER_EVENT1;
    v2->f0 = v4;
    v2->f1 = v17;
    v17->ctx = v8;
    v17->f0 = v5;
  } else {
    v18 = &next_work_USER_EVENT2;
    v2->f0 = v6;
    v2->f1 = v18;
    v18->ctx = v8;
    v18->f0 = v5;
  }
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
		switch (wrap_out.f0) {
		case 2: {
			next_work_ref_USER_EVENT2 = next_work_USER_EVENT2;
			cls_workq_add_work(WORKQ_ID_USER_EVENT2, &next_work_ref_USER_EVENT2, sizeof(next_work_ref_USER_EVENT2));
			break;
		}
		case 1: {
			next_work_ref_USER_EVENT1 = next_work_USER_EVENT1;
			cls_workq_add_work(WORKQ_ID_USER_EVENT1, &next_work_ref_USER_EVENT1, sizeof(next_work_ref_USER_EVENT1));
			break;
		}
		}
	}
}
