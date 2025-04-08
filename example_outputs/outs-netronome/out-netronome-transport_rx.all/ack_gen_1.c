#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct eth_header_t_sub_0 _loc_buf_16;
__xrw static struct eth_header_t_sub_0 _loc_buf_16_xfer;
static struct tcp_header_t_sub_0 _loc_buf_14;
__xrw static struct tcp_header_t_sub_0 _loc_buf_14_xfer;
static struct repack_type_1 _loc_buf_17;
__xrw static struct repack_type_1 _loc_buf_17_xfer;
static struct ip_header_t_sub_1 _loc_buf_15;
__xrw static struct ip_header_t_sub_1 _loc_buf_15_xfer;
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
  __shared __cls struct tcp_header_t_sub_0* v7;
  struct tcp_header_t_sub_0* v8;
  __xrw struct tcp_header_t_sub_0* v9;
  __shared __cls struct ip_header_t_sub_1* v10;
  struct ip_header_t_sub_1* v11;
  __xrw struct ip_header_t_sub_1* v12;
  __shared __cls struct eth_header_t_sub_0* v13;
  struct eth_header_t_sub_0* v14;
  __xrw struct eth_header_t_sub_0* v15;
  struct __buf_t v16;
  uint32_t v17;
  uint16_t v18;
  uint32_t v19;
  struct eth_header_t_sub_0* v20;
  uint16_t v21;
  struct eth_header_t_sub_0* v22;
  struct eth_header_t_sub_0* v23;
  struct eth_header_t_sub_0* v24;
  uint32_t v25;
  uint32_t v26;
  struct ip_header_t_sub_1* v27;
  struct ip_header_t_sub_1* v28;
  uint16_t v29;
  uint16_t v30;
  struct tcp_header_t_sub_0* v31;
  struct tcp_header_t_sub_0* v32;
  uint32_t v33;
  struct tcp_header_t_sub_0* v34;
  uint32_t v35;
  struct tcp_header_t_sub_0* v36;
  __xrw struct eth_header_t_sub_0* v37;
  struct repack_type_1* v38;
  __xrw struct repack_type_1* v39;
  __xrw struct ip_header_t_sub_1* v40;
  __xrw struct tcp_header_t_sub_0* v41;
  __declspec(aligned(4)) struct event_param_NET_SEND* v42;
  v1 = 0;
  v2 = 64;
  v3 = &work;
  v4 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_ACK_GEN_1, v4, sizeof(*v4));
  *(v3) = *(v4);
  v5 = v3->ctx;
  v6 = &v3->f0;
  v7 = &v5->f4;
  v8 = &_loc_buf_14;
  v9 = &_loc_buf_14_xfer;
  cls_read(&v9->f0, &v7->f0, 12);
  *(v8) = *(v9);
  v10 = &v5->f2;
  v11 = &_loc_buf_15;
  v12 = &_loc_buf_15_xfer;
  cls_read(&v12->f0, &v10->f0, 8);
  *(v11) = *(v12);
  v13 = &v5->f3;
  v14 = &_loc_buf_16;
  v15 = &_loc_buf_16_xfer;
  cls_read(&v15->f0, &v13->f0, 12);
  *(v14) = *(v15);
  v16 = v5->f0;
  v17 = v14->f0;
  v18 = v14->f1;
  v19 = v14->f2;
  v14->f0 = v19;
  v21 = v14->f3;
  v14->f1 = v21;
  v14->f2 = v17;
  v14->f3 = v18;
  v25 = v11->f0;
  v26 = v11->f1;
  v11->f0 = v26;
  v11->f1 = v25;
  v29 = v8->f0;
  v30 = v8->f1;
  v8->f0 = v30;
  v8->f1 = v29;
  v33 = v6->f0;
  v8->f2 = v33;
  v35 = v6->f1;
  v8->f3 = v35;
  v37 = &_loc_buf_16_xfer;
  *(v37) = *(v14);
  v16.offs = 0;
  mem_write32(&v37->f0, v16.buf + v16.offs, 12);
  v38 = &_loc_buf_17;
  v38->f0 = v2;
  v39 = &_loc_buf_17_xfer;
  *(v39) = *(v38);
  v16.offs = 16;
  mem_write8(&v39->f0, v16.buf + v16.offs, 2);
  v40 = &_loc_buf_15_xfer;
  *(v40) = *(v11);
  v16.offs = 26;
  mem_write32(&v40->f0, v16.buf + v16.offs, 8);
  v41 = &_loc_buf_14_xfer;
  *(v41) = *(v8);
  v16.offs = 38;
  mem_write32(&v41->f0, v16.buf + v16.offs, 12);
  v42 = &next_work_NET_SEND;
  v42->ctx = v5;
  v42->f0 = v16;
  next_work_NET_SEND.meta.len = next_work_NET_SEND.f0.sz;
  inlined_net_send(v42);
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
