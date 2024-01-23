#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct eth_header_t_sub_0 _loc_buf_31;
__xrw static struct eth_header_t_sub_0 _loc_buf_31_xfer;
static struct ip_header_t_sub_1 _loc_buf_32;
__xrw static struct ip_header_t_sub_1 _loc_buf_32_xfer;
static struct tcp_header_t_sub_0 _loc_buf_30;
__xrw static struct tcp_header_t_sub_0 _loc_buf_30_xfer;
static struct repack_type_1 _loc_buf_33;
__xrw static struct repack_type_1 _loc_buf_33_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_ACK_GEN work;
__xrw struct event_param_ACK_GEN work_ref;
__declspec(aligned(4)) struct event_param_NET_SEND next_work_NET_SEND;
__xrw struct event_param_NET_SEND next_work_ref_NET_SEND;

__forceinline
void __event___handler_ACK_GEN_ack_gen_2() {
  uint16_t v1;
  uint32_t v2;
  __declspec(aligned(4)) struct event_param_ACK_GEN* v3;
  __xrw struct event_param_ACK_GEN* v4;
  __shared __cls struct context_chain_1_t* v5;
  struct ack_info_t* v6;
  __shared __cls struct tcp_header_t_sub_0* v7;
  struct tcp_header_t_sub_0* _loc_buf_30;
  __xrw struct tcp_header_t_sub_0* v8;
  __shared __cls struct eth_header_t_sub_0* v9;
  struct eth_header_t_sub_0* _loc_buf_31;
  __xrw struct eth_header_t_sub_0* v10;
  struct __buf_t v11;
  __shared __cls struct ip_header_t_sub_1* v12;
  struct ip_header_t_sub_1* _loc_buf_32;
  __xrw struct ip_header_t_sub_1* v13;
  uint48_t v14;
  uint48_t v15;
  struct eth_header_t_sub_0* v16;
  struct eth_header_t_sub_0* v17;
  uint32_t v18;
  uint32_t v19;
  struct ip_header_t_sub_1* v20;
  struct ip_header_t_sub_1* v21;
  uint16_t v22;
  uint16_t v23;
  struct tcp_header_t_sub_0* v24;
  struct tcp_header_t_sub_0* v25;
  uint32_t v26;
  struct tcp_header_t_sub_0* v27;
  uint32_t v28;
  struct tcp_header_t_sub_0* v29;
  __xrw struct eth_header_t_sub_0* v30;
  struct repack_type_1* _loc_buf_33;
  __xrw struct repack_type_1* v31;
  __xrw struct ip_header_t_sub_1* v32;
  __xrw struct tcp_header_t_sub_0* v33;
  __declspec(aligned(4)) struct event_param_NET_SEND* v34;
  v1 = 64;
  v2 = 0;
  v3 = &work;
  v4 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_ACK_GEN_2, v4, sizeof(*v4));
  *(v3) = *(v4);
  v5 = v3->ctx;
  v6 = &v3->f0;
  v7 = &v5->f3;
  ;
  v8 = &_loc_buf_30_xfer;
  cls_read(&v8->f0, &v7->f0, 12);
  _loc_buf_30 = *(v8);
  v9 = &v5->f2;
  ;
  v10 = &_loc_buf_31_xfer;
  cls_read(&v10->f0, &v9->f0, 12);
  _loc_buf_31 = *(v10);
  v11 = v5->f0;
  v12 = &v5->f1;
  ;
  v13 = &_loc_buf_32_xfer;
  cls_read(&v13->f0, &v12->f0, 8);
  _loc_buf_32 = *(v13);
  v14 = _loc_buf_31.f0;
  v15 = _loc_buf_31.f1;
  _loc_buf_31.f0 = v15;
  _loc_buf_31.f1 = v14;
  v18 = _loc_buf_32.f0;
  v19 = _loc_buf_32.f1;
  _loc_buf_32.f0 = v19;
  _loc_buf_32.f1 = v18;
  v22 = _loc_buf_30.f0;
  v23 = _loc_buf_30.f1;
  _loc_buf_30.f0 = v23;
  _loc_buf_30.f1 = v22;
  v26 = v6->f0;
  _loc_buf_30.f2 = v26;
  v28 = v6->f1;
  _loc_buf_30.f3 = v28;
  v30 = &_loc_buf_31_xfer;
  *(v30) = _loc_buf_31;
  v11.offs = 0;
  mem_write32(&v30->f0, v11.buf + v11.offs, 12);
  ;
  _loc_buf_33.f0 = v1;
  v31 = &_loc_buf_33_xfer;
  *(v31) = _loc_buf_33;
  v11.offs = 16;
  mem_write8(&v31->f0, v11.buf + v11.offs, 2);
  v32 = &_loc_buf_32_xfer;
  *(v32) = _loc_buf_32;
  v11.offs = 26;
  mem_write32(&v32->f0, v11.buf + v11.offs, 8);
  v33 = &_loc_buf_30_xfer;
  *(v33) = _loc_buf_30;
  v11.offs = 38;
  mem_write32(&v33->f0, v11.buf + v11.offs, 12);
  v34 = &next_work_NET_SEND;
  v34->ctx = v5;
  v34->f0 = v11;
  next_work_NET_SEND.meta.len = next_work_NET_SEND.f0.sz;
  inlined_net_send(v34);
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
