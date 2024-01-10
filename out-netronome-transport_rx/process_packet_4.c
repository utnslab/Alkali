#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct ip_header_t _loc_buf_10;
__xrw static struct ip_header_t _loc_buf_10_xfer;
static struct eth_header_t _loc_buf_9;
__xrw static struct eth_header_t _loc_buf_9_xfer;
static struct tcp_header_t _loc_buf_11;
__xrw static struct tcp_header_t _loc_buf_11_xfer;
static struct pkt_info_t _loc_buf_21;
__xrw static struct pkt_info_t _loc_buf_21_xfer;
__declspec(aligned(4)) struct event_param_NET_RECV work;
__xrw struct event_param_NET_RECV work_ref;
__declspec(aligned(4)) struct event_param_OoO_DETECT next_work_OoO_DETECT;
__xrw struct event_param_OoO_DETECT next_work_ref_OoO_DETECT;

__forceinline static void dispatch1 () {
	switch (hash(work.ctx->f4) % 2) {
	case 0:
		cls_workq_add_work(WORKQ_ID_OoO_DETECT_1, &next_work_ref_OoO_DETECT, sizeof(next_work_ref_OoO_DETECT));
		break;
	case 1:
		cls_workq_add_work(WORKQ_ID_OoO_DETECT_2, &next_work_ref_OoO_DETECT, sizeof(next_work_ref_OoO_DETECT));
		break;
	}
}

__forceinline
void __event___handler_NET_RECV_process_packet_4() {
  int64_t v1;
  int32_t v2;
  __declspec(aligned(4)) struct event_param_NET_RECV* v3;
  struct context_chain_1_t* v4;
  struct context_chain_1_t* v5;
  struct __buf_t v6;
  struct pkt_info_t* v7;
  struct eth_header_t* v8;
  __xrw struct eth_header_t* v9;
  struct ip_header_t* v10;
  __xrw struct ip_header_t* v11;
  struct tcp_header_t* v12;
  __xrw struct tcp_header_t* v13;
  int16_t v14;
  int16_t v15;
  int32_t v16;
  struct pkt_info_t* v17;
  int32_t v18;
  struct pkt_info_t* v19;
  int16_t v20;
  int32_t v21;
  struct pkt_info_t* v22;
  __declspec(aligned(4)) struct event_param_OoO_DETECT* v23;
  __xrw struct event_param_OoO_DETECT* v24;
  v1 = 14;
  v2 = 3;
  v3 = &work;
  inlined_net_recv(v3);
  v4 = alloc_context_chain_ring_entry();
  v3->ctx = v4;
  v5 = v3->ctx;
  v6 = v3->f0;
  v7 = &_loc_buf_21;
  v8 = &_loc_buf_9;
  v9 = &_loc_buf_9_xfer;
  mem_read32(&v9->f0, v6.buf + v6.offs, 12);
  v6.offs += 12;
  mem_read8(&v9->f2, v6.buf + v6.offs, 2);
  v6.offs += 2;
  *(v8) = *(v9);
  v10 = &_loc_buf_10;
  v11 = &_loc_buf_10_xfer;
  mem_read32(&v11->f0, v6.buf + v6.offs, 24);
  v6.offs += 24;
  *(v10) = *(v11);
  v12 = &_loc_buf_11;
  v13 = &_loc_buf_11_xfer;
  mem_read32(&v13->f0, v6.buf + v6.offs, 20);
  v6.offs += 20;
  *(v12) = *(v13);
  v5->f1 = *v8;
  v5->f2 = *v10;
  v5->f3 = *v12;
  v5->f0 = v6;
  v14 = v10->f1;
  v15 = v14 + v1;
  v16 = (int32_t) v15;
  v7->f1 = v16;
  v18 = v12->f2;
  v7->f2 = v18;
  v20 = v12->f1;
  v21 = (int32_t) v20;
  v7->f0 = v21;
  v5->f4 = v20;
  v23 = &next_work_OoO_DETECT;
  v23->ctx = v5;
  v23->f0 = *v7;
  v24 = &next_work_ref_OoO_DETECT;
  *(v24) = *(v23);
  dispatch1();
  return;
}


int main(void) {
	init_me_cam(16);
	init_context_chain_ring();
	wait_global_start_();
	for (;;) {
		__event___handler_NET_RECV_process_packet_4();
	}
}
