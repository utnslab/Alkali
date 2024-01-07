#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

struct pkt_info_t _loc_buf_18;
__xrw struct pkt_info_t _loc_buf_18_xfer;
struct tcp_header_t _loc_buf_2;
__xrw struct tcp_header_t _loc_buf_2_xfer;
struct eth_header_t _loc_buf_0;
__xrw struct eth_header_t _loc_buf_0_xfer;
struct ip_header_t _loc_buf_1;
__xrw struct ip_header_t _loc_buf_1_xfer;
__declspec(aligned(4)) struct event_param_NET_RECV work;
__xrw struct event_param_NET_RECV work_ref;
__declspec(aligned(4)) struct event_param_OoO_DETECT next_work_OoO_DETECT;
__xrw struct event_param_OoO_DETECT next_work_ref_OoO_DETECT;

__forceinline
void __event___handler_NET_RECV_process_packet_1() {
  int64_t v1;
  int32_t v2;
  int32_t v3;
  __declspec(aligned(4)) struct event_param_NET_RECV* v4;
  struct context_chain_1_t* v5;
  struct context_chain_1_t* v6;
  struct __buf_t v7;
  struct pkt_info_t* v8;
  struct eth_header_t* v9;
  __xrw struct eth_header_t* v10;
  struct ip_header_t* v11;
  __xrw struct ip_header_t* v12;
  struct tcp_header_t* v13;
  __xrw struct tcp_header_t* v14;
  int16_t v15;
  int16_t v16;
  int32_t v17;
  struct pkt_info_t* v18;
  int32_t v19;
  struct pkt_info_t* v20;
  struct pkt_info_t* v21;
  __declspec(aligned(4)) struct event_param_OoO_DETECT* v22;
  __xrw struct event_param_OoO_DETECT* v23;
  v1 = 14;
  v2 = 3;
  v3 = 0;
  v4 = &work;
  inlined_net_recv(v4);
  v5 = alloc_context_chain_ring_entry();
  v4->ctx = v5;
  v6 = v4->ctx;
  v7 = v4->f0;
  v8 = &_loc_buf_18;
  v9 = &_loc_buf_0;
  v10 = &_loc_buf_0_xfer;
  mem_read32(&v10->f0, v7.buf + v7.offs, 12);
  v7.offs += 12;
  mem_read8(&v10->f2, v7.buf + v7.offs, 2);
  v7.offs += 2;
  *(v9) = *(v10);
  v11 = &_loc_buf_1;
  v12 = &_loc_buf_1_xfer;
  mem_read32(&v12->f0, v7.buf + v7.offs, 24);
  v7.offs += 24;
  *(v11) = *(v12);
  v13 = &_loc_buf_2;
  v14 = &_loc_buf_2_xfer;
  mem_read32(&v14->f0, v7.buf + v7.offs, 20);
  v7.offs += 20;
  *(v13) = *(v14);
  v6->f0 = *v9;
  v6->f1 = *v11;
  v6->f2 = *v13;
  v6->f3 = v7;
  v15 = v11->f1;
  v16 = v15 + v1;
  v17 = (int32_t) v16;
  v8->f1 = v17;
  v19 = v13->f2;
  v8->f2 = v19;
  v8->f0 = v3;
  v22 = &next_work_OoO_DETECT;
  v22->ctx = v6;
  v22->f0 = *v8;
  v23 = &next_work_ref_OoO_DETECT;
  *(v23) = *(v22);
  cls_workq_add_work(WORKQ_ID_OoO_DETECT_1, v23, sizeof(*v23));
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
