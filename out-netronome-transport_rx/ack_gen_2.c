#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct tcp_header_t _loc_buf_27;
__xrw static struct tcp_header_t _loc_buf_27_xfer;
static struct eth_header_t _loc_buf_26;
__xrw static struct eth_header_t _loc_buf_26_xfer;
static struct ip_header_t _loc_buf_25;
__xrw static struct ip_header_t _loc_buf_25_xfer;
__declspec(aligned(4)) struct event_param_ACK_GEN work;
__xrw struct event_param_ACK_GEN work_ref;
__declspec(aligned(4)) struct event_param_NET_SEND next_work_NET_SEND;
__xrw struct event_param_NET_SEND next_work_ref_NET_SEND;

__forceinline
void __event___handler_ACK_GEN_ack_gen_2() {
  uint32_t v1;
  uint16_t v2;
  __declspec(aligned(4)) struct event_param_ACK_GEN* v3;
  __xrw struct event_param_ACK_GEN* v4;
  __shared __cls struct context_chain_1_t* v5;
  struct ack_info_t* v6;
  __shared __cls struct ip_header_t* v7;
  struct ip_header_t* v8;
  __xrw struct ip_header_t* v9;
  __shared __cls struct eth_header_t* v10;
  struct eth_header_t* v11;
  __xrw struct eth_header_t* v12;
  struct __buf_t v13;
  __shared __cls struct tcp_header_t* v14;
  struct tcp_header_t* v15;
  __xrw struct tcp_header_t* v16;
  uint48_t v17;
  uint48_t v18;
  struct eth_header_t* v19;
  struct eth_header_t* v20;
  uint32_t v21;
  uint32_t v22;
  struct ip_header_t* v23;
  struct ip_header_t* v24;
  struct ip_header_t* v25;
  uint16_t v26;
  uint16_t v27;
  struct tcp_header_t* v28;
  struct tcp_header_t* v29;
  uint32_t v30;
  struct tcp_header_t* v31;
  uint32_t v32;
  struct tcp_header_t* v33;
  __xrw struct eth_header_t* v34;
  __xrw struct ip_header_t* v35;
  __xrw struct tcp_header_t* v36;
  __declspec(aligned(4)) struct event_param_NET_SEND* v37;
  v1 = 0;
  v2 = 64;
  v3 = &work;
  v4 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_ACK_GEN_2, v4, sizeof(*v4));
  *(v3) = *(v4);
  v5 = v3->ctx;
  v6 = &v3->f0;
  v7 = &v5->f2;
  v8 = &_loc_buf_25;
  v9 = &_loc_buf_25_xfer;
  cls_read(&v9->f0, &v7->f0, 24);
  *(v8) = *(v9);
  v10 = &v5->f3;
  v11 = &_loc_buf_26;
  v12 = &_loc_buf_26_xfer;
  cls_read(&v12->f0, &v10->f0, 16);
  *(v11) = *(v12);
  v13 = v5->f0;
  v14 = &v5->f1;
  v15 = &_loc_buf_27;
  v16 = &_loc_buf_27_xfer;
  cls_read(&v16->f0, &v14->f0, 20);
  *(v15) = *(v16);
  v17 = v11->f0;
  v18 = v11->f1;
  v11->f0 = v18;
  v11->f1 = v17;
  v21 = v8->f6;
  v22 = v8->f7;
  v8->f6 = v22;
  v8->f7 = v21;
  v8->f1 = v2;
  v26 = v15->f0;
  v27 = v15->f1;
  v15->f0 = v27;
  v15->f1 = v26;
  v30 = v6->f0;
  v15->f2 = v30;
  v32 = v6->f1;
  v15->f3 = v32;
  v34 = &_loc_buf_26_xfer;
  *(v34) = *(v11);
  v13.offs = 0;
  mem_write32(&v34->f0, v13.buf + v13.offs, 12);
  v13.offs = 12;
  mem_write8(&v34->f2, v13.buf + v13.offs, 2);
  v35 = &_loc_buf_25_xfer;
  *(v35) = *(v8);
  v13.offs = 14;
  mem_write32(&v35->f0, v13.buf + v13.offs, 24);
  v36 = &_loc_buf_27_xfer;
  *(v36) = *(v15);
  v13.offs = 38;
  mem_write32(&v36->f0, v13.buf + v13.offs, 20);
  v37 = &next_work_NET_SEND;
  v37->ctx = v5;
  v37->f0 = v13;
  next_work_NET_SEND.meta.len = next_work_NET_SEND.f0.sz;
  inlined_net_send(v37);
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_ACK_GEN_2, workq_ACK_GEN_2, WORKQ_TYPE_ACK_GEN, WORKQ_SIZE_ACK_GEN, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_ACK_GEN_ack_gen_2();
	}
}
