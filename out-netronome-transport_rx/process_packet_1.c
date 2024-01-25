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
static struct repack_type_0 _loc_buf_2;
__xrw static struct repack_type_0 _loc_buf_2_xfer;
static struct eth_header_t_sub_0 _loc_buf_1;
__xrw static struct eth_header_t_sub_0 _loc_buf_1_xfer;
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
  struct eth_header_t_sub_0* v8;
  __xrw struct eth_header_t_sub_0* v9;
  struct repack_type_0* v10;
  __xrw struct repack_type_0* v11;
  uint16_t v12;
  struct ip_header_t_sub_1* v13;
  __xrw struct ip_header_t_sub_1* v14;
  struct tcp_header_t_sub_0* v15;
  __xrw struct tcp_header_t_sub_0* v16;
  uint16_t v17;
  uint32_t v18;
  struct pkt_info_t* v19;
  uint32_t v20;
  struct pkt_info_t* v21;
  uint16_t v22;
  uint32_t v23;
  struct pkt_info_t* v24;
  __shared __cls struct ip_header_t_sub_1* v25;
  struct ip_header_t_sub_1* v26;
  __xrw struct ip_header_t_sub_1* v27;
  __shared __cls struct eth_header_t_sub_0* v28;
  struct eth_header_t_sub_0* v29;
  __xrw struct eth_header_t_sub_0* v30;
  __shared __cls struct tcp_header_t_sub_0* v31;
  struct tcp_header_t_sub_0* v32;
  __xrw struct tcp_header_t_sub_0* v33;
  __declspec(aligned(4)) struct event_param_OoO_DETECT* v34;
  __xrw struct event_param_OoO_DETECT* v35;
  v1 = 14;
  v2 = 3;
  v3 = &work;
  inlined_net_recv(v3);
  v4 = alloc_context_chain_ring_entry();
  v3->ctx = v4;
  v5 = v3->ctx;
  v6 = v3->f0;
  v7 = &_loc_buf_0;
  v8 = &_loc_buf_1;
  v9 = &_loc_buf_1_xfer;
  v6.offs = 0;
  mem_read32(&v9->f0, v6.buf + v6.offs, 12);
  *(v8) = *(v9);
  v10 = &_loc_buf_2;
  v11 = &_loc_buf_2_xfer;
  v6.offs = 16;
  mem_read32(&v11->f0, v6.buf + v6.offs, 4);
  *(v10) = *(v11);
  v12 = v10->f0;
  v13 = &_loc_buf_3;
  v14 = &_loc_buf_3_xfer;
  v6.offs = 26;
  mem_read32(&v14->f0, v6.buf + v6.offs, 8);
  *(v13) = *(v14);
  v15 = &_loc_buf_4;
  v16 = &_loc_buf_4_xfer;
  v6.offs = 38;
  mem_read32(&v16->f0, v6.buf + v6.offs, 12);
  *(v15) = *(v16);
  v17 = v12 + v1;
  v18 = (uint32_t) v17;
  v7->f1 = v18;
  v20 = v15->f2;
  v7->f2 = v20;
  v22 = v15->f1;
  v23 = (uint32_t) v22;
  v7->f0 = v23;
  v5->f0 = v6;
  v5->f4 = v22;
  v25 = &v5->f1;
  v26 = &_loc_buf_3;
  v27 = &_loc_buf_3_xfer;
  *(v27) = *(v26);
  cls_write(&v27->f0, &v25->f0, 8);
  v28 = &v5->f2;
  v29 = &_loc_buf_1;
  v30 = &_loc_buf_1_xfer;
  *(v30) = *(v29);
  cls_write(&v30->f0, &v28->f0, 12);
  v31 = &v5->f3;
  v32 = &_loc_buf_4;
  v33 = &_loc_buf_4_xfer;
  *(v33) = *(v32);
  cls_write(&v33->f0, &v31->f0, 12);
  v34 = &next_work_OoO_DETECT;
  v34->ctx = v5;
  v34->f0 = *v7;
  v35 = &next_work_ref_OoO_DETECT;
  *(v35) = *(v34);
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
