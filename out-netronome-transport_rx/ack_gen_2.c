#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct ip_header_t _loc_buf_16;
__xrw static struct ip_header_t _loc_buf_16_xfer;
static struct eth_header_t _loc_buf_15;
__xrw static struct eth_header_t _loc_buf_15_xfer;
static struct tcp_header_t _loc_buf_17;
__xrw static struct tcp_header_t _loc_buf_17_xfer;
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
  struct __buf_t v7;
  __shared __cls struct eth_header_t* v8;
  struct eth_header_t* v9;
  __xrw struct eth_header_t* v10;
  uint48_t v11;
  uint48_t v12;
  struct eth_header_t* v13;
  struct eth_header_t* v14;
  __shared __cls struct ip_header_t* v15;
  struct ip_header_t* v16;
  __xrw struct ip_header_t* v17;
  uint32_t v18;
  uint32_t v19;
  struct ip_header_t* v20;
  struct ip_header_t* v21;
  struct ip_header_t* v22;
  __shared __cls struct tcp_header_t* v23;
  struct tcp_header_t* v24;
  __xrw struct tcp_header_t* v25;
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
  struct __buf_t v37;
  __declspec(aligned(4)) struct event_param_NET_SEND* v38;
  v1 = 0;
  v2 = 64;
  v3 = &work;
  v4 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_ACK_GEN_2, v4, sizeof(*v4));
  *(v3) = *(v4);
  v5 = v3->ctx;
  v6 = &v3->f0;
  v7 = alloc_packet_buf();
  v8 = &v5->f0;
  v9 = &_loc_buf_15;
  v10 = &_loc_buf_15_xfer;
  cls_read(&v10->f0, &v8->f0, 16);
  *(v9) = *(v10);
  v11 = v9->f0;
  v12 = v9->f1;
  v9->f0 = v12;
  v9->f1 = v11;
  v15 = &v5->f1;
  v16 = &_loc_buf_16;
  v17 = &_loc_buf_16_xfer;
  cls_read(&v17->f0, &v15->f0, 24);
  *(v16) = *(v17);
  v18 = v16->f6;
  v19 = v16->f7;
  v16->f6 = v19;
  v16->f7 = v18;
  v16->f1 = v2;
  v23 = &v5->f2;
  v24 = &_loc_buf_17;
  v25 = &_loc_buf_17_xfer;
  cls_read(&v25->f0, &v23->f0, 20);
  *(v24) = *(v25);
  v26 = v24->f0;
  v27 = v24->f1;
  v24->f0 = v27;
  v24->f1 = v26;
  v30 = v6->f0;
  v24->f2 = v30;
  v32 = v6->f1;
  v24->f3 = v32;
  v34 = &_loc_buf_15_xfer;
  *(v34) = *(v9);
  mem_write32(&v34->f0, v7.buf + v7.offs, 12);
  v7.offs += 12;
  mem_write8(&v34->f2, v7.buf + v7.offs, 2);
  v7.offs += 2;
  v35 = &_loc_buf_16_xfer;
  *(v35) = *(v16);
  mem_write32(&v35->f0, v7.buf + v7.offs, 24);
  v7.offs += 24;
  v36 = &_loc_buf_17_xfer;
  *(v36) = *(v24);
  mem_write32(&v36->f0, v7.buf + v7.offs, 20);
  v7.offs += 20;
  v37 = v5->f3;
  bulk_memcpy(v7.buf + v7.offs, v37.buf + v37.offs, v37.sz - v37.offs);
  v7.offs += v37.sz - v37.offs;
  v38 = &next_work_NET_SEND;
  v38->ctx = v5;
  v38->f0 = v7;
  next_work_NET_SEND.meta.len = next_work_NET_SEND.f0.offs;
  inlined_net_send(v38);
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
