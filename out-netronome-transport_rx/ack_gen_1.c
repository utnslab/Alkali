#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct repack_type_1 _loc_buf_29;
__xrw static struct repack_type_1 _loc_buf_29_xfer;
static struct ip_header_t_sub_1 _loc_buf_26;
__xrw static struct ip_header_t_sub_1 _loc_buf_26_xfer;
static struct eth_header_t_sub_0 _loc_buf_27;
__xrw static struct eth_header_t_sub_0 _loc_buf_27_xfer;
static struct tcp_header_t_sub_0 _loc_buf_28;
__xrw static struct tcp_header_t_sub_0 _loc_buf_28_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_ACK_GEN work;
__xrw struct event_param_ACK_GEN work_ref;
__declspec(aligned(4)) struct event_param_NET_SEND next_work_NET_SEND;
__xrw struct event_param_NET_SEND next_work_ref_NET_SEND;

__forceinline
void __event___handler_ACK_GEN_ack_gen_1() {
  uint32_t v1;
  uint16_t v2;
  __declspec(aligned(4)) struct event_param_ACK_GEN* v3;
  __xrw struct event_param_ACK_GEN* v4;
  __shared __cls struct context_chain_1_t* v5;
  struct ack_info_t* v6;
  __shared __cls struct ip_header_t_sub_1* v7;
  struct ip_header_t_sub_1* v8;
  __xrw struct ip_header_t_sub_1* v9;
  __shared __cls struct eth_header_t_sub_0* v10;
  struct eth_header_t_sub_0* v11;
  __xrw struct eth_header_t_sub_0* v12;
  __shared __cls struct tcp_header_t_sub_0* v13;
  struct tcp_header_t_sub_0* v14;
  __xrw struct tcp_header_t_sub_0* v15;
  struct __buf_t v16;
  uint48_t v17;
  uint48_t v18;
  struct eth_header_t_sub_0* v19;
  struct eth_header_t_sub_0* v20;
  uint32_t v21;
  uint32_t v22;
  struct ip_header_t_sub_1* v23;
  struct ip_header_t_sub_1* v24;
  uint16_t v25;
  uint16_t v26;
  struct tcp_header_t_sub_0* v27;
  struct tcp_header_t_sub_0* v28;
  uint32_t v29;
  struct tcp_header_t_sub_0* v30;
  uint32_t v31;
  struct tcp_header_t_sub_0* v32;
  __xrw struct eth_header_t_sub_0* v33;
  struct repack_type_1* v34;
  __xrw struct repack_type_1* v35;
  __xrw struct ip_header_t_sub_1* v36;
  __xrw struct tcp_header_t_sub_0* v37;
  __declspec(aligned(4)) struct event_param_NET_SEND* v38;
  v1 = 0;
  v2 = 64;
  v3 = &work;
  v4 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_ACK_GEN_1, v4, sizeof(*v4));
  *(v3) = *(v4);
  v5 = v3->ctx;
  v6 = &v3->f0;
  v7 = &v5->f1;
  v8 = &_loc_buf_26;
  v9 = &_loc_buf_26_xfer;
  cls_read(&v9->f0, &v7->f0, 8);
  *(v8) = *(v9);
  v10 = &v5->f2;
  v11 = &_loc_buf_27;
  v12 = &_loc_buf_27_xfer;
  cls_read(&v12->f0, &v10->f0, 12);
  *(v11) = *(v12);
  v13 = &v5->f3;
  v14 = &_loc_buf_28;
  v15 = &_loc_buf_28_xfer;
  cls_read(&v15->f0, &v13->f0, 12);
  *(v14) = *(v15);
  v16 = v5->f0;
  v17 = v11->f0;
  v18 = v11->f1;
  v11->f0 = v18;
  v11->f1 = v17;
  v21 = v8->f0;
  v22 = v8->f1;
  v8->f0 = v22;
  v8->f1 = v21;
  v25 = v14->f0;
  v26 = v14->f1;
  v14->f0 = v26;
  v14->f1 = v25;
  v29 = v6->f0;
  v14->f2 = v29;
  v31 = v6->f1;
  v14->f3 = v31;
  v33 = &_loc_buf_27_xfer;
  *(v33) = *(v11);
  v16.offs = 0;
  mem_write32(&v33->f0, v16.buf + v16.offs, 12);
  v34 = &_loc_buf_29;
  v34->f0 = v2;
  v35 = &_loc_buf_29_xfer;
  *(v35) = *(v34);
  v16.offs = 16;
  mem_write8(&v35->f0, v16.buf + v16.offs, 2);
  v36 = &_loc_buf_26_xfer;
  *(v36) = *(v8);
  v16.offs = 26;
  mem_write32(&v36->f0, v16.buf + v16.offs, 8);
  v37 = &_loc_buf_28_xfer;
  *(v37) = *(v14);
  v16.offs = 38;
  mem_write32(&v37->f0, v16.buf + v16.offs, 12);
  v38 = &next_work_NET_SEND;
  v38->ctx = v5;
  v38->f0 = v16;
  next_work_NET_SEND.meta.len = next_work_NET_SEND.f0.sz;
  inlined_net_send(v38);
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_ACK_GEN_1, workq_ACK_GEN_1, WORKQ_TYPE_ACK_GEN, WORKQ_SIZE_ACK_GEN, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_ACK_GEN_ack_gen_1();
	}
}
