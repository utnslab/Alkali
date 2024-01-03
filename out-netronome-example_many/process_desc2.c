#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

__declspec(aligned(4)) struct event_param_USER_EVENT2 work;
__xrw struct event_param_USER_EVENT2 work_ref;
struct __wrapper_arg_t wrap_in;
__forceinline
void __event___handler_USER_EVENT2_process_desc2(struct __wrapper_arg_t* v1) {
  struct event_param_USER_EVENT2* v2;
  struct context_chain_1_t* v3;
  int32_t v4;
  v2 = v1->f1;
  v3 = v2->ctx;
  v4 = v2->f0;
  v3->f3 = v4;
  return;
}


int main(void) {
	init_recv_event_workq(WORKQ_ID_USER_EVENT2, workq_USER_EVENT2, WORKQ_TYPE_USER_EVENT2, WORKQ_SIZE_USER_EVENT2, 8);
	for (;;) {
		cls_workq_add_thread(WORKQ_ID_USER_EVENT2, &work_ref, sizeof(work_ref));
		work = work_ref;
		wrap_in.f1 = &work;
		__event___handler_USER_EVENT2_process_desc2(&wrap_in);
	}
}
