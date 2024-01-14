#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct pkt_info_t _loc_buf_30;
__xrw static struct pkt_info_t _loc_buf_30_xfer;
static struct tcp_header_t _loc_buf_20;
__xrw static struct tcp_header_t _loc_buf_20_xfer;
static struct eth_header_t _loc_buf_0;
__xrw static struct eth_header_t _loc_buf_0_xfer;
static struct eth_header_t _loc_buf_18;
__xrw static struct eth_header_t _loc_buf_18_xfer;
static struct ip_header_t _loc_buf_19;
__xrw static struct ip_header_t _loc_buf_19_xfer;
static struct ip_header_t _loc_buf_1;
__xrw static struct ip_header_t _loc_buf_1_xfer;
static struct tcp_header_t _loc_buf_2;
__xrw static struct tcp_header_t _loc_buf_2_xfer;
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
void __event___handler_NET_RECV_process_packet_1() {
  uint64_t v1;
  uint32_t v2;
  __declspec(aligned(4)) struct event_param_NET_RECV* v3;
  __shared __cls struct context_chain_1_t* v4;
  __shared __cls struct context_chain_1_t* v5;
  struct __buf_t v6;
  struct pkt_info_t* v7;
  struct eth_header_t* v8;
  __xrw struct eth_header_t* v9;
  struct ip_header_t* v10;
  __xrw struct ip_header_t* v11;
  struct tcp_header_t* v12;
  __xrw struct tcp_header_t* v13;
  __shared __cls struct eth_header_t* v14;
  struct eth_header_t* v15;
  __xrw struct eth_header_t* v16;
  __shared __cls struct ip_header_t* v17;
  struct ip_header_t* v18;
  __xrw struct ip_header_t* v19;
  __shared __cls struct tcp_header_t* v20;
  struct tcp_header_t* v21;
  __xrw struct tcp_header_t* v22;
  uint16_t v23;
  uint16_t v24;
  uint32_t v25;
  struct pkt_info_t* v26;
  uint32_t v27;
  struct pkt_info_t* v28;
  uint16_t v29;
  uint32_t v30;
  struct pkt_info_t* v31;
  __declspec(aligned(4)) struct event_param_OoO_DETECT* v32;
  __xrw struct event_param_OoO_DETECT* v33;
  v1 = 14;
  v2 = 3;
  v3 = &work;
  inlined_net_recv(v3);
  v4 = alloc_context_chain_ring_entry();
  v3->ctx = v4;
  v5 = v3->ctx;
  v6 = v3->f0;
  v7 = &_loc_buf_30;
  v8 = &_loc_buf_0;
  v9 = &_loc_buf_0_xfer;
  mem_read32(&v9->f0, v6.buf + v6.offs, 16);
  v6.offs += 14;
  *(v8) = *(v9);
  v10 = &_loc_buf_1;
  v11 = &_loc_buf_1_xfer;
  mem_read32(&v11->f0, v6.buf + v6.offs, 24);
  v6.offs += 24;
  *(v10) = *(v11);
  v12 = &_loc_buf_2;
  v13 = &_loc_buf_2_xfer;
  mem_read32(&v13->f0, v6.buf + v6.offs, 20);
  v6.offs += 20;
  *(v12) = *(v13);
  v14 = &v5->f0;
  v15 = &_loc_buf_18;
  v16 = &_loc_buf_18_xfer;
  *(v16) = *(v15);
  cls_write(&v16->f0, &v14->f0, 12);
  v14->f2 = v16->f2;
  v17 = &v5->f1;
  v18 = &_loc_buf_19;
  v19 = &_loc_buf_19_xfer;
  *(v19) = *(v18);
  cls_write(&v19->f0, &v17->f0, 24);
  v20 = &v5->f2;
  v21 = &_loc_buf_20;
  v22 = &_loc_buf_20_xfer;
  *(v22) = *(v21);
  cls_write(&v22->f0, &v20->f0, 20);
  v5->f3 = v6;
  v23 = v10->f1;
  v24 = v23 + v1;
  v25 = (uint32_t) v24;
  v7->f1 = v25;
  v27 = v12->f2;
  v7->f2 = v27;
  v29 = v12->f1;
  v30 = (uint32_t) v29;
  v7->f0 = v30;
  v5->f4 = v29;
  v32 = &next_work_OoO_DETECT;
  v32->ctx = v5;
  v32->f0 = *v7;
  v33 = &next_work_ref_OoO_DETECT;
  *(v33) = *(v32);
  dispatch1();
  return;
}


int main(void) {
	init_me_cam(16);
	init_context_chain_ring();
	wait_global_start_();
	for (;;) {
		__event___handler_NET_RECV_process_packet_1();
	}
}
