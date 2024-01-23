#define DO_CTXQ_INIT

#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct rpc_header_t _loc_buf_3;
__xrw static struct rpc_header_t _loc_buf_3_xfer;
static struct udp_header_t _loc_buf_2;
__xrw static struct udp_header_t _loc_buf_2_xfer;
static struct ip_header_t _loc_buf_1;
__xrw static struct ip_header_t _loc_buf_1_xfer;
static struct eth_header_t _loc_buf_0;
__xrw static struct eth_header_t _loc_buf_0_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_NET_RECV work;
__xrw struct event_param_NET_RECV work_ref;
__declspec(aligned(4)) struct event_param_MSG_REASSEMBLE next_work_MSG_REASSEMBLE;
__xrw struct event_param_MSG_REASSEMBLE next_work_ref_MSG_REASSEMBLE;

__forceinline
void __event___handler_NET_RECV_process_packet_1() {
  uint32_t v1;
  __declspec(aligned(4)) struct event_param_NET_RECV* v2;
  __shared __cls struct context_chain_1_t* v3;
  __shared __cls struct context_chain_1_t* v4;
  struct __buf_t v5;
  struct eth_header_t* v6;
  __xrw struct eth_header_t* v7;
  struct ip_header_t* v8;
  __xrw struct ip_header_t* v9;
  struct udp_header_t* v10;
  __xrw struct udp_header_t* v11;
  struct rpc_header_t* v12;
  __xrw struct rpc_header_t* v13;
  __declspec(aligned(4)) struct event_param_MSG_REASSEMBLE* v14;
  __xrw struct event_param_MSG_REASSEMBLE* v15;
  v1 = 2;
  v2 = &work;
  inlined_net_recv(v2);
  v3 = alloc_context_chain_ring_entry();
  v2->ctx = v3;
  v4 = v2->ctx;
  v5 = v2->f0;
  v6 = &_loc_buf_0;
  v7 = &_loc_buf_0_xfer;
  mem_read32(&v7->f0, v5.buf + v5.offs, 16);
  v5.offs += 14;
  *(v6) = *(v7);
  v8 = &_loc_buf_1;
  v9 = &_loc_buf_1_xfer;
  mem_read32(&v9->f0, v5.buf + v5.offs, 24);
  v5.offs += 24;
  *(v8) = *(v9);
  v10 = &_loc_buf_2;
  v11 = &_loc_buf_2_xfer;
  mem_read32(&v11->f0, v5.buf + v5.offs, 8);
  v5.offs += 8;
  *(v10) = *(v11);
  v12 = &_loc_buf_3;
  v13 = &_loc_buf_3_xfer;
  mem_read32(&v13->f0, v5.buf + v5.offs, 16);
  v5.offs += 16;
  *(v12) = *(v13);
  v14 = &next_work_MSG_REASSEMBLE;
  v14->ctx = v4;
  v14->f0 = v5;
  v14->f1 = *v12;
  v15 = &next_work_ref_MSG_REASSEMBLE;
  *(v15) = *(v14);
  cls_workq_add_work(WORKQ_ID_MSG_REASSEMBLE_1, v15, sizeof(*v15));
  return;
}


int main(void) {
	init_me_cam(16);
	wait_global_start_();
	for (;;) {
		__event___handler_NET_RECV_process_packet_1();
	}
}
