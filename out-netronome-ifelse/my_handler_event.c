#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

__declspec(aligned(4)) struct event_param_MY_EVENT work;
__xrw struct event_param_MY_EVENT work_ref;
struct __wrapper_arg_t wrap_in;
__forceinline
void __event___handler_MY_EVENT_my_handler_event(struct __wrapper_arg_t* v1) {
  int64_t v2;
  int64_t v3;
  int32_t v4;
  int32_t v5;
  struct event_param_MY_EVENT* v6;
  struct context_chain_1_t* v7;
  int32_t v8;
  int32_t v9;
  bool v10;
  bool v11;
  int32_t v12;
  int32_t v13;
  v2 = 1;
  v3 = 2;
  v4 = 1;
  v5 = 233;
  v6 = v1->f1;
  v7 = v6->ctx;
  v8 = v6->f0;
  v9 = v6->f1;
  v10 = v8 == v2;
  v11 = v9 == v2;
  v12 = (v10 ? v4 : v5);
  v13 = v12 + v3;
  if (v11) {
    v7->f0 = v3;
    v7->f1 = v13;
  }
  return;
}


int main(void) {
	init_context_chain_ring();
	init_recv_event_workq(WORKQ_ID_MY_EVENT, workq_MY_EVENT, WORKQ_TYPE_MY_EVENT, WORKQ_SIZE_MY_EVENT, 8);
	for (;;) {
		cls_workq_add_thread(WORKQ_ID_MY_EVENT, &work_ref, sizeof(work_ref));
		work = work_ref;
		wrap_in.f1 = &work;
		work.ctx = alloc_context_chain_ring_entry();
		__event___handler_MY_EVENT_my_handler_event(&wrap_in);
	}
}
