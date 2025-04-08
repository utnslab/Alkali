#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

__declspec(aligned(4)) struct event_param_USER_EVENT1 work;
__xrw struct event_param_USER_EVENT1 work_ref;

__forceinline
void __event___handler_USER_EVENT1_process_desc() {
  __declspec(aligned(4)) struct event_param_USER_EVENT1* v1;
  __xrw struct event_param_USER_EVENT1* v2;
  struct context_chain_1_t* v3;
  int32_t v4;
  v1 = &work;
  v2 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_USER_EVENT1, v2, sizeof(*v2));
  *(v1) = *(v2);
  v3 = v1->ctx;
  v4 = v1->f0;
  v3->f1 = v4;
  return;
}


int main(void) {
	init_recv_event_workq(WORKQ_ID_USER_EVENT1, workq_USER_EVENT1, WORKQ_TYPE_USER_EVENT1, WORKQ_SIZE_USER_EVENT1, 8);
	for (;;) {
		__event___handler_USER_EVENT1_process_desc();
	}
}
