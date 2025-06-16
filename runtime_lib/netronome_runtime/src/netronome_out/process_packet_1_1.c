#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct firewall_ip_entries_t _loc_buf_16;
__xrw static struct firewall_ip_entries_t _loc_buf_16_xfer;
static struct tcp_header_t _loc_buf_15;
__xrw static struct tcp_header_t _loc_buf_15_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_NET_RECV_1 work;
__xrw struct event_param_NET_RECV_1 work_ref;
__declspec(aligned(4)) struct event_param_NET_RECV_2 next_work_NET_RECV_2;
__xrw struct event_param_NET_RECV_2 next_work_ref_NET_RECV_2;

__forceinline static void dispatch1 () {
	switch (rr_ctr) {
	case 0:
		cls_workq_add_work(WORKQ_ID_NET_RECV_2_1, &next_work_ref_NET_RECV_2, sizeof(next_work_ref_NET_RECV_2));
		break;
	case 1:
		cls_workq_add_work(WORKQ_ID_NET_RECV_2_2, &next_work_ref_NET_RECV_2, sizeof(next_work_ref_NET_RECV_2));
		break;
	case 2:
		cls_workq_add_work(WORKQ_ID_NET_RECV_2_3, &next_work_ref_NET_RECV_2, sizeof(next_work_ref_NET_RECV_2));
		break;
	}
	rr_ctr = rr_ctr == 2 ? 0 : (rr_ctr + 1);
}

__forceinline
void __event___handler_NET_RECV_1_process_packet_1_1() {
  uint32_t v1;
  __declspec(aligned(4)) struct event_param_NET_RECV_1* v2;
  __xrw struct event_param_NET_RECV_1* v3;
  __shared __cls struct context_chain_1_t* v4;
  uint32_t v5;
  uint32_t v6;
  struct __buf_t v7;
  __shared __cls struct tcp_header_t* v8;
  struct tcp_header_t* v9;
  __xrw struct tcp_header_t* v10;
  uint32_t v11;
  __export __shared __cls struct table_i32_firewall_ip_entries_t_64_t* v12;
  struct firewall_ip_entries_t* v13;
  uint32_t v14;
  uint32_t v15;
  uint32_t v16;
  uint32_t v17;
  uint32_t v18;
  uint16_t v19;
  __declspec(aligned(4)) struct event_param_NET_RECV_2* v20;
  __xrw struct event_param_NET_RECV_2* v21;
  v1 = 4;
  v2 = &work;
  v3 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_NET_RECV_1_1, v3, sizeof(*v3));
  *(v2) = *(v3);
  v4 = v2->ctx;
  v5 = v2->f0;
  v6 = v2->f1;
  v7 = v2->f2;
  v8 = &v4->f9;
  v9 = &_loc_buf_15;
  v10 = &_loc_buf_15_xfer;
  cls_read(&v10->f0, &v8->f0, 20);
  *(v9) = *(v10);
  v11 = v4->f4;
  v12 = &firewall_ip_table;
  v13 = &_loc_buf_16;
  *v13 = v12->table[lmem_cam_lookup(firewall_ip_table_index, v11, 64)];
  v14 = v13->f4;
  v15 = v5 + v14;
  v16 = v13->f3;
  v17 = v16 + v6;
  v18 = v9->f2;
  v19 = v9->f7;
  v4->f1 = v17;
  v4->f5 = v19;
  v4->f2 = v15;
  v4->f7 = v18;
  v20 = &next_work_NET_RECV_2;
  v20->ctx = v4;
  v20->f0 = v7;
  v21 = &next_work_ref_NET_RECV_2;
  *(v21) = *(v20);
  dispatch1();
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_NET_RECV_1_1, workq_NET_RECV_1_1, WORKQ_TYPE_NET_RECV_1, WORKQ_SIZE_NET_RECV_1, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_NET_RECV_1_process_packet_1_1();
	}
}
