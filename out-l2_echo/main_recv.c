#include "nfplib.h"
#include "prog_hdr.h"

struct eth_header_t _loc_buf_0;
struct event_param_NET_RECV work;
__xrw struct event_param_NET_RECV work_ref;
struct __wrapper_arg_t wrap_in;
struct event_param_NET_SEND next_work;
__xrw struct event_param_NET_SEND next_work_ref;
struct __wrapper_arg_t wrap_out;

void __event___handler_NET_RECV_main_recv(struct __wrapper_arg_t* v1, struct __wrapper_arg_t* v2) {
  int32_t v3;
  int32_t v4;
  int32_t v5;
  struct event_param_NET_RECV* v6;
  struct context_chain_1_t* v7;
  char* v8;
  char* v9;
  struct eth_header_t* v10;
  int48_t v11;
  int48_t v12;
  struct eth_header_t* v13;
  struct eth_header_t* v14;
  struct event_param_NET_SEND* v15;
  v3 = 14;
  v4 = 1486;
  v5 = 0;
  v6 = v1->f1;
  v7 = v6->ctx;
  v8 = v6->f0;
  v9 = __ep2_rt_alloc_buf();
  v10 = &_loc_buf_0;
  bulk_memcpy(v10, v8, v3);
  v8 += 14;
  v11 = v10->f1;
  v12 = v10->f0;
  v10->f1 = v12;
  v13->f0 = v11;
  bulk_memcpy(v9, v14, v3);
  v9 += 14;
  bulk_memcpy(v9, v8, v4);
  v9 += 1486;
  v15 = &next_work;
  v2->f0 = v5;
  v2->f1 = v15;
  v15->ctx = v7;
  v15->f0 = v9;
  return;
}


int main(void) {
	init_context_chain_ring();
	init_recv_event_workq(WORKQ_ID_NET_RECV, workq_NET_RECV, WORKQ_TYPE_NET_RECV, WORKQ_SIZE_NET_RECV, 8);
	for (;;) {
		cls_workq_add_thread(WORKQ_ID_NET_RECV, &work_ref, sizeof(work_ref));
		work = work_ref;
		wrap_in.f1 = &work;
		work.ctx = alloc_context_chain_ring_entry();
		__event___handler_NET_RECV_main_recv(&wrap_in, &wrap_out);
		next_work_ref = next_work;
		cls_workq_add_work(WORKQ_ID_NET_SEND, &next_work_ref, sizeof(next_work_ref));
	}
}
