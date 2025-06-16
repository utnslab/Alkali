#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct tcp_tracker_t _loc_buf_25;
__xrw static struct tcp_tracker_t _loc_buf_25_xfer;
static struct err_tracker_t _loc_buf_26;
__xrw static struct err_tracker_t _loc_buf_26_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_NET_RECV_3 work;
__xrw struct event_param_NET_RECV_3 work_ref;
__declspec(aligned(4)) struct event_param_NET_RECV_4 next_work_NET_RECV_4;
__xrw struct event_param_NET_RECV_4 next_work_ref_NET_RECV_4;

__forceinline static void dispatch1 () {
	switch (rr_ctr) {
	case 0:
		cls_workq_add_work(WORKQ_ID_NET_RECV_4_1, &next_work_ref_NET_RECV_4, sizeof(next_work_ref_NET_RECV_4));
		break;
	case 1:
		cls_workq_add_work(WORKQ_ID_NET_RECV_4_2, &next_work_ref_NET_RECV_4, sizeof(next_work_ref_NET_RECV_4));
		break;
	case 2:
		cls_workq_add_work(WORKQ_ID_NET_RECV_4_3, &next_work_ref_NET_RECV_4, sizeof(next_work_ref_NET_RECV_4));
		break;
	}
	rr_ctr = rr_ctr == 2 ? 0 : (rr_ctr + 1);
}

__forceinline
void __event___handler_NET_RECV_3_process_packet_3_1() {
  uint32_t v1;
  uint32_t v2;
  uint32_t v3;
  __declspec(aligned(4)) struct event_param_NET_RECV_3* v4;
  __xrw struct event_param_NET_RECV_3* v5;
  __shared __cls struct context_chain_1_t* v6;
  struct __buf_t v7;
  uint32_t v8;
  uint32_t v9;
  uint16_t v10;
  uint32_t v11;
  __shared __cls struct tcp_tracker_t* v12;
  struct tcp_tracker_t* v13;
  __xrw struct tcp_tracker_t* v14;
  __export __shared __cls struct table_i32_err_tracker_t_64_t* v15;
  struct err_tracker_t* v16;
  uint32_t v17;
  uint32_t v18;
  uint32_t v19;
  struct err_tracker_t* v20;
  uint32_t v21;
  uint32_t v22;
  uint32_t v23;
  uint32_t v24;
  struct err_tracker_t* v25;
  uint32_t v26;
  uint32_t v27;
  uint32_t v28;
  uint32_t v29;
  uint32_t v30;
  uint32_t v31;
  struct tcp_tracker_t* v32;
  uint32_t v33;
  uint32_t v34;
  __declspec(aligned(4)) struct event_param_NET_RECV_4* v35;
  __xrw struct event_param_NET_RECV_4* v36;
  v1 = 1;
  v2 = 256;
  v3 = 6;
  v4 = &work;
  v5 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_NET_RECV_3_1, v5, sizeof(*v5));
  *(v4) = *(v5);
  v6 = v4->ctx;
  v7 = v4->f0;
  v8 = v6->f4;
  v9 = v6->f7;
  v10 = v6->f5;
  v11 = v6->f2;
  v12 = &v6->f6;
  v13 = &_loc_buf_25;
  v14 = &_loc_buf_25_xfer;
  cls_read(&v14->f0, &v12->f0, 12);
  *(v13) = *(v14);
  v15 = &err_tracker_table;
  v16 = &_loc_buf_26;
  *v16 = v15->table[lmem_cam_lookup(err_tracker_table_index, v8, 64)];
  v17 = v16->f0;
  v18 = v17 + v1;
  v19 = v18 - v11;
  v16->f0 = v19;
  v21 = v16->f2;
  v22 = v21 + v2;
  v23 = v22 + v9;
  v24 = v23 + v10;
  v16->f2 = v24;
  v26 = v16->f0;
  v27 = v16->f2;
  v28 = v26 + v27;
  v29 = v28 + v10;
  v15->table[lmem_cam_update(err_tracker_table_index, v8, 64)] = *v16;
  v30 = v13->f2;
  v31 = v30 + v29;
  v13->f2 = v31;
  v33 = v13->f0;
  v34 = v13->f2;
  v35 = &next_work_NET_RECV_4;
  v35->ctx = v6;
  v35->f0 = v1;
  v35->f1 = v33;
  v35->f2 = v7;
  v35->f3 = v34;
  v36 = &next_work_ref_NET_RECV_4;
  *(v36) = *(v35);
  dispatch1();
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_NET_RECV_3_1, workq_NET_RECV_3_1, WORKQ_TYPE_NET_RECV_3, WORKQ_SIZE_NET_RECV_3, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_NET_RECV_3_process_packet_3_1();
	}
}
