#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct pkt_info_t _loc_buf_0;
__xrw static struct pkt_info_t _loc_buf_0_xfer;
static struct ip_header_t _loc_buf_2;
__xrw static struct ip_header_t _loc_buf_2_xfer;
static struct tcp_header_t _loc_buf_3;
__xrw static struct tcp_header_t _loc_buf_3_xfer;
static struct eth_header_t _loc_buf_1;
__xrw static struct eth_header_t _loc_buf_1_xfer;
static struct ip_header_t _loc_buf_2;
__xrw static struct ip_header_t _loc_buf_2_xfer;
static struct eth_header_t _loc_buf_1;
__xrw static struct eth_header_t _loc_buf_1_xfer;
static struct tcp_header_t _loc_buf_3;
__xrw static struct tcp_header_t _loc_buf_3_xfer;
__declspec(aligned(4)) struct event_param_NET_RECV work;
__xrw struct event_param_NET_RECV work_ref;
__declspec(aligned(4)) struct event_param_OoO_DETECT next_work_OoO_DETECT;
__xrw struct event_param_OoO_DETECT next_work_ref_OoO_DETECT;

__forceinline
void __event___handler_NET_RECV_process_packet_1() {
  uint32_t v1;
  uint64_t v2;
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
  uint16_t v14;
  uint16_t v15;
  uint32_t v16;
  struct pkt_info_t* v17;
  uint32_t v18;
  struct pkt_info_t* v19;
  uint16_t v20;
  uint32_t v21;
  struct pkt_info_t* v22;
  __shared __cls struct ip_header_t* v23;
  struct ip_header_t* v24;
  __xrw struct ip_header_t* v25;
  __shared __cls struct eth_header_t* v26;
  struct eth_header_t* v27;
  __xrw struct eth_header_t* v28;
  __shared __cls struct tcp_header_t* v29;
  struct tcp_header_t* v30;
  __xrw struct tcp_header_t* v31;
  __declspec(aligned(4)) struct event_param_OoO_DETECT* v32;
  __xrw struct event_param_OoO_DETECT* v33;
  v1 = 3;
  v2 = 14;
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
  mem_read32(&v9->f0, v6.buf + v6.offs, 16);
  *(v8) = *(v9);
  v10 = &_loc_buf_2;
  v11 = &_loc_buf_2_xfer;
  v6.offs = 14;
  mem_read32(&v11->f0, v6.buf + v6.offs, 24);
  *(v10) = *(v11);
  v12 = &_loc_buf_3;
  v13 = &_loc_buf_3_xfer;
  v6.offs = 38;
  mem_read32(&v13->f0, v6.buf + v6.offs, 20);
  *(v12) = *(v13);
  v14 = v10->f1;
  v15 = v14 + v2;
  v16 = (uint32_t) v15;
  v7->f1 = v16;
  v18 = v12->f2;
  v7->f2 = v18;
  v20 = v12->f1;
  v21 = (uint32_t) v20;
  v7->f0 = v21;
  v5->f0 = v6;
  v5->f1 = v20;
  v23 = &v5->f2;
  v24 = &_loc_buf_2;
  v25 = &_loc_buf_2_xfer;
  *(v25) = *(v24);
  cls_write(&v25->f0, &v23->f0, 24);
  v26 = &v5->f3;
  v27 = &_loc_buf_1;
  v28 = &_loc_buf_1_xfer;
  *(v28) = *(v27);
  cls_write(&v28->f0, &v26->f0, 16);
  v29 = &v5->f4;
  v30 = &_loc_buf_3;
  v31 = &_loc_buf_3_xfer;
  *(v31) = *(v30);
  cls_write(&v31->f0, &v29->f0, 20);
  v32 = &next_work_OoO_DETECT;
  v32->ctx = v5;
  v32->f0 = *v7;
  v33 = &next_work_ref_OoO_DETECT;
  *(v33) = *(v32);
  cls_workq_add_work(WORKQ_ID_OoO_DETECT_1, v33, sizeof(*v33));
  return;
}


int main(void) {
	init_me_cam(16);
	wait_global_start_();
	for (;;) {
		__event___handler_NET_RECV_process_packet_1();
	}
}
