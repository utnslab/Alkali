#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct pkt_info_t _loc_buf_0;
__xrw static struct pkt_info_t _loc_buf_0_xfer;
static struct ip_header_t_sub_1 _loc_buf_3;
__xrw static struct ip_header_t_sub_1 _loc_buf_3_xfer;
static struct tcp_header_t_sub_0 _loc_buf_4;
__xrw static struct tcp_header_t_sub_0 _loc_buf_4_xfer;
static struct eth_header_t_sub_0 _loc_buf_1;
__xrw static struct eth_header_t_sub_0 _loc_buf_1_xfer;
static struct repack_type_0 _loc_buf_2;
__xrw static struct repack_type_0 _loc_buf_2_xfer;
static int rr_ctr = 0;
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
	rr_ctr = (rr_ctr + 1) & 1;
}

__forceinline
void __event___handler_NET_RECV_process_packet_1() {
  uint16_t v1;
  uint32_t v2;
  __declspec(aligned(4)) struct event_param_NET_RECV* v3;
  __shared __cls struct context_chain_1_t* v4;
  __shared __cls struct context_chain_1_t* v5;
  struct __buf_t v6;
  struct pkt_info_t* v7;
  struct eth_header_t_sub_0* _loc_buf_1;
  __xrw struct eth_header_t_sub_0* v8;
  struct repack_type_0* _loc_buf_2;
  __xrw struct repack_type_0* v9;
  uint16_t v10;
  struct ip_header_t_sub_1* _loc_buf_3;
  __xrw struct ip_header_t_sub_1* v11;
  struct tcp_header_t_sub_0* _loc_buf_4;
  __xrw struct tcp_header_t_sub_0* v12;
  uint16_t v13;
  uint32_t v14;
  struct pkt_info_t* v15;
  uint32_t v16;
  struct pkt_info_t* v17;
  uint16_t v18;
  uint32_t v19;
  struct pkt_info_t* v20;
  __shared __cls struct ip_header_t_sub_1* v21;
  struct ip_header_t_sub_1* _loc_buf_3;
  __xrw struct ip_header_t_sub_1* v22;
  __shared __cls struct eth_header_t_sub_0* v23;
  struct eth_header_t_sub_0* _loc_buf_1;
  __xrw struct eth_header_t_sub_0* v24;
  __shared __cls struct tcp_header_t_sub_0* v25;
  struct tcp_header_t_sub_0* _loc_buf_4;
  __xrw struct tcp_header_t_sub_0* v26;
  __declspec(aligned(4)) struct event_param_OoO_DETECT* v27;
  __xrw struct event_param_OoO_DETECT* v28;
  v1 = 14;
  v2 = 3;
  v3 = &work;
  inlined_net_recv(v3);
  v4 = alloc_context_chain_ring_entry();
  v3->ctx = v4;
  v5 = v3->ctx;
  v6 = v3->f0;
  v7 = &_loc_buf_0;
  ;
  v8 = &_loc_buf_1_xfer;
  v6.offs = 0;
  mem_read32(&v8->f0, v6.buf + v6.offs, 12);
  _loc_buf_1 = *(v8);
  ;
  v9 = &_loc_buf_2_xfer;
  v6.offs = 16;
  mem_read32(&v9->f0, v6.buf + v6.offs, 4);
  _loc_buf_2 = *(v9);
  v10 = _loc_buf_2.f0;
  ;
  v11 = &_loc_buf_3_xfer;
  v6.offs = 26;
  mem_read32(&v11->f0, v6.buf + v6.offs, 8);
  _loc_buf_3 = *(v11);
  ;
  v12 = &_loc_buf_4_xfer;
  v6.offs = 38;
  mem_read32(&v12->f0, v6.buf + v6.offs, 12);
  _loc_buf_4 = *(v12);
  v13 = v10 + v1;
  v14 = (uint32_t) v13;
  v7->f1 = v14;
  v16 = _loc_buf_4.f2;
  v7->f2 = v16;
  v18 = _loc_buf_4.f1;
  v19 = (uint32_t) v18;
  v7->f0 = v19;
  v5->f0 = v6;
  v5->f4 = v18;
  v21 = &v5->f1;
  ;
  v22 = &_loc_buf_3_xfer;
  *(v22) = _loc_buf_3;
  cls_write(&v22->f0, &v21->f0, 8);
  v23 = &v5->f2;
  ;
  v24 = &_loc_buf_1_xfer;
  *(v24) = _loc_buf_1;
  cls_write(&v24->f0, &v23->f0, 12);
  v25 = &v5->f3;
  ;
  v26 = &_loc_buf_4_xfer;
  *(v26) = _loc_buf_4;
  cls_write(&v26->f0, &v25->f0, 12);
  v27 = &next_work_OoO_DETECT;
  v27->ctx = v5;
  v27->f0 = *v7;
  v28 = &next_work_ref_OoO_DETECT;
  *(v28) = *(v27);
  dispatch1();
  return;
}


int main(void) {
	init_me_cam(16);
	wait_global_start_();
	for (;;) {
		__event___handler_NET_RECV_process_packet_1();
	}
}
