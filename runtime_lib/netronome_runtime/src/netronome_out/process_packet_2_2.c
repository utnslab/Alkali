#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct tcp_tracker_t _loc_buf_21;
__xrw static struct tcp_tracker_t _loc_buf_21_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_NET_RECV_2 work;
__xrw struct event_param_NET_RECV_2 work_ref;
__declspec(aligned(4)) struct event_param_NET_RECV_3 next_work_NET_RECV_3;
__xrw struct event_param_NET_RECV_3 next_work_ref_NET_RECV_3;

__forceinline static void dispatch1 () {
	switch (rr_ctr) {
	case 0:
		cls_workq_add_work(WORKQ_ID_NET_RECV_3_1, &next_work_ref_NET_RECV_3, sizeof(next_work_ref_NET_RECV_3));
		break;
	case 1:
		cls_workq_add_work(WORKQ_ID_NET_RECV_3_2, &next_work_ref_NET_RECV_3, sizeof(next_work_ref_NET_RECV_3));
		break;
	case 2:
		cls_workq_add_work(WORKQ_ID_NET_RECV_3_3, &next_work_ref_NET_RECV_3, sizeof(next_work_ref_NET_RECV_3));
		break;
	}
	rr_ctr = rr_ctr == 2 ? 0 : (rr_ctr + 1);
}

__forceinline
void __event___handler_NET_RECV_2_process_packet_2_2() {
  uint32_t v1;
  __declspec(aligned(4)) struct event_param_NET_RECV_2* v2;
  __xrw struct event_param_NET_RECV_2* v3;
  __shared __cls struct context_chain_1_t* v4;
  struct __buf_t v5;
  uint32_t v6;
  uint32_t v7;
  __export __shared __cls struct table_i32_tcp_tracker_t_64_t* v8;
  struct tcp_tracker_t* v9;
  uint32_t v10;
  uint32_t v11;
  struct tcp_tracker_t* v12;
  __shared __cls struct tcp_tracker_t* v13;
  struct tcp_tracker_t* v14;
  __xrw struct tcp_tracker_t* v15;
  __declspec(aligned(4)) struct event_param_NET_RECV_3* v16;
  __xrw struct event_param_NET_RECV_3* v17;
  v1 = 5;
  v2 = &work;
  v3 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_NET_RECV_2_2, v3, sizeof(*v3));
  *(v2) = *(v3);
  v4 = v2->ctx;
  v5 = v2->f0;
  v6 = v4->f8;
  v7 = v4->f2;
  v8 = &tcp_tracker_table;
  v9 = &_loc_buf_21;
  *v9 = v8->table[lmem_cam_lookup(tcp_tracker_table_index, v6, 64)];
  v10 = v9->f0;
  v11 = v10 - v7;
  v9->f0 = v11;
  v8->table[lmem_cam_update(tcp_tracker_table_index, v6, 64)] = *v9;
  v13 = &v4->f6;
  v14 = &_loc_buf_21;
  v15 = &_loc_buf_21_xfer;
  *(v15) = *(v14);
  cls_write(&v15->f0, &v13->f0, 12);
  v16 = &next_work_NET_RECV_3;
  v16->ctx = v4;
  v16->f0 = v5;
  v17 = &next_work_ref_NET_RECV_3;
  *(v17) = *(v16);
  dispatch1();
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_NET_RECV_2_2, workq_NET_RECV_2_2, WORKQ_TYPE_NET_RECV_2, WORKQ_SIZE_NET_RECV_2, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_NET_RECV_2_process_packet_2_2();
	}
}
