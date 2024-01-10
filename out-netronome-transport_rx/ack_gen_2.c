#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct eth_header_t _loc_buf_15;
__xrw static struct eth_header_t _loc_buf_15_xfer;
static struct ip_header_t _loc_buf_16;
__xrw static struct ip_header_t _loc_buf_16_xfer;
static struct tcp_header_t _loc_buf_17;
__xrw static struct tcp_header_t _loc_buf_17_xfer;
__declspec(aligned(4)) struct event_param_ACK_GEN work;
__xrw struct event_param_ACK_GEN work_ref;
__declspec(aligned(4)) struct event_param_NET_SEND next_work_NET_SEND;
__xrw struct event_param_NET_SEND next_work_ref_NET_SEND;

__forceinline
void __event___handler_ACK_GEN_ack_gen_2() {
  int32_t v1;
  int16_t v2;
  __declspec(aligned(4)) struct event_param_ACK_GEN* v3;
  __xrw struct event_param_ACK_GEN* v4;
  struct context_chain_1_t* v5;
  struct ack_info_t* v6;
  struct __buf_t v7;
  struct eth_header_t* v8;
  struct eth_header_t* v9;
  int48_t v10;
  int48_t v11;
  struct eth_header_t* v12;
  struct eth_header_t* v13;
  struct ip_header_t* v14;
  struct ip_header_t* v15;
  int32_t v16;
  int32_t v17;
  struct ip_header_t* v18;
  struct ip_header_t* v19;
  struct ip_header_t* v20;
  struct tcp_header_t* v21;
  struct tcp_header_t* v22;
  int16_t v23;
  int16_t v24;
  struct tcp_header_t* v25;
  struct tcp_header_t* v26;
  int32_t v27;
  struct tcp_header_t* v28;
  int32_t v29;
  struct tcp_header_t* v30;
  __xrw struct eth_header_t* v31;
  __xrw struct ip_header_t* v32;
  __xrw struct tcp_header_t* v33;
  struct __buf_t v34;
  __declspec(aligned(4)) struct event_param_NET_SEND* v35;
  v1 = 0;
  v2 = 64;
  v3 = &work;
  v4 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_ACK_GEN_2, v4, sizeof(*v4));
  *(v3) = *(v4);
  v5 = v3->ctx;
  v6 = &v3->f0;
  v7 = alloc_packet_buf();
  v8 = &v5->f1;
  v9 = &_loc_buf_15;
  *(v9) = *(v8);
  v10 = v9->f0;
  v11 = v9->f1;
  v9->f0 = v11;
  v9->f1 = v10;
  v14 = &v5->f2;
  v15 = &_loc_buf_16;
  *(v15) = *(v14);
  v16 = v15->f6;
  v17 = v15->f7;
  v15->f6 = v17;
  v15->f7 = v16;
  v15->f1 = v2;
  v21 = &v5->f3;
  v22 = &_loc_buf_17;
  *(v22) = *(v21);
  v23 = v22->f0;
  v24 = v22->f1;
  v22->f0 = v24;
  v22->f1 = v23;
  v27 = v6->f0;
  v22->f2 = v27;
  v29 = v6->f1;
  v22->f3 = v29;
  v31 = &_loc_buf_15_xfer;
  *(v31) = *(v9);
  mem_write32(&v31->f0, v7.buf + v7.offs, 12);
  v7.offs += 12;
  mem_write8(&v31->f2, v7.buf + v7.offs, 2);
  v7.offs += 2;
  v32 = &_loc_buf_16_xfer;
  *(v32) = *(v15);
  mem_write32(&v32->f0, v7.buf + v7.offs, 24);
  v7.offs += 24;
  v33 = &_loc_buf_17_xfer;
  *(v33) = *(v22);
  mem_write32(&v33->f0, v7.buf + v7.offs, 20);
  v7.offs += 20;
  v34 = v5->f0;
  bulk_memcpy(v7.buf + v7.offs, v34.buf + v34.offs, v34.sz - v34.offs);
  v7.offs += v34.sz - v34.offs;
  v35 = &next_work_NET_SEND;
  v35->ctx = v5;
  v35->f0 = v7;
  next_work_NET_SEND.meta.len = next_work_NET_SEND.f0.offs;
  inlined_net_send(v35);
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
