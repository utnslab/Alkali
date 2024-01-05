#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

__declspec(aligned(4)) struct event_param_DMA_WRITE_REQ work;
__xrw struct event_param_DMA_WRITE_REQ work_ref;

__forceinline
void __event___handler_DMA_WRITE_REQ_dma_write() {
  __declspec(aligned(4)) struct event_param_DMA_WRITE_REQ* v1;
  __xrw struct event_param_DMA_WRITE_REQ* v2;
  struct context_chain_1_t* v3;
  struct __buf_t v4;
  struct dma_write_cmd_t* v5;
  v1 = &work;
  v2 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_DMA_WRITE_REQ, v2, sizeof(*v2));
  *(v1) = *(v2);
  v3 = v1->ctx;
  v4 = v1->f0;
  v5 = &v1->f1;
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_DMA_WRITE_REQ, workq_DMA_WRITE_REQ, WORKQ_TYPE_DMA_WRITE_REQ, WORKQ_SIZE_DMA_WRITE_REQ, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_DMA_WRITE_REQ_dma_write();
	}
}
