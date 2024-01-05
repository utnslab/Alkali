#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

__declspec(aligned(4)) struct event_param_NET_SEND work;
__xrw struct event_param_NET_SEND work_ref;

__forceinline
void __event___handler_NET_SEND_net_send() {
  __declspec(aligned(4)) struct event_param_NET_SEND* v1;
  __xrw struct event_param_NET_SEND* v2;
  struct context_chain_1_t* v3;
  struct __buf_t v4;
  v1 = &work;
  v2 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_NET_SEND, v2, sizeof(*v2));
  *(v1) = *(v2);
  v3 = v1->ctx;
  v4 = v1->f0;
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_NET_SEND, workq_NET_SEND, WORKQ_TYPE_NET_SEND, WORKQ_SIZE_NET_SEND, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_NET_SEND_net_send();
	}
}
