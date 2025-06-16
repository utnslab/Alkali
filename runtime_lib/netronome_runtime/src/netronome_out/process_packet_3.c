#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct ip_header_t _loc_buf_11;
__xrw static struct ip_header_t _loc_buf_11_xfer;
static struct eth_header_t _loc_buf_10;
__xrw static struct eth_header_t _loc_buf_10_xfer;
static struct priority_entries_t _loc_buf_14;
__xrw static struct priority_entries_t _loc_buf_14_xfer;
static struct tcp_header_t _loc_buf_12;
__xrw static struct tcp_header_t _loc_buf_12_xfer;
static struct firewall_tcpport_entries_t _loc_buf_13;
__xrw static struct firewall_tcpport_entries_t _loc_buf_13_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_NET_RECV work;
__xrw struct event_param_NET_RECV work_ref;
__declspec(aligned(4)) struct event_param_NET_RECV_1 next_work_NET_RECV_1;
__xrw struct event_param_NET_RECV_1 next_work_ref_NET_RECV_1;

__forceinline static void dispatch1 () {
	switch (rr_ctr) {
	case 0:
		cls_workq_add_work(WORKQ_ID_NET_RECV_1_1, &next_work_ref_NET_RECV_1, sizeof(next_work_ref_NET_RECV_1));
		break;
	case 1:
		cls_workq_add_work(WORKQ_ID_NET_RECV_1_2, &next_work_ref_NET_RECV_1, sizeof(next_work_ref_NET_RECV_1));
		break;
	}
	rr_ctr = (rr_ctr + 1) & 1;
}

__forceinline
void __event___handler_NET_RECV_process_packet_3() {
  uint32_t v1;
  __declspec(aligned(4)) struct event_param_NET_RECV* v2;
  __shared __cls struct context_chain_1_t* v3;
  __shared __cls struct context_chain_1_t* v4;
  struct __buf_t v5;
  __export __shared __cls struct table_i32_firewall_tcpport_entries_t_64_t* v6;
  __export __shared __cls struct table_i32_priority_entries_t_64_t* v7;
  struct eth_header_t* v8;
  __xrw struct eth_header_t* v9;
  struct ip_header_t* v10;
  __xrw struct ip_header_t* v11;
  struct tcp_header_t* v12;
  __xrw struct tcp_header_t* v13;
  uint32_t v14;
  uint32_t v15;
  uint32_t v16;
  uint16_t v17;
  uint32_t v18;
  uint16_t v19;
  uint32_t v20;
  struct firewall_tcpport_entries_t* v21;
  struct priority_entries_t* v22;
  uint32_t v23;
  uint32_t v24;
  uint32_t v25;
  uint32_t v26;
  __shared __cls struct tcp_header_t* v27;
  struct tcp_header_t* v28;
  __xrw struct tcp_header_t* v29;
  __declspec(aligned(4)) struct event_param_NET_RECV_1* v30;
  __xrw struct event_param_NET_RECV_1* v31;
  v1 = 3;
  v2 = &work;
  inlined_net_recv(v2);
  v3 = alloc_context_chain_ring_entry();
  v2->ctx = v3;
  v4 = v2->ctx;
  v5 = v2->f0;
  v6 = &firewall_tcpport_table;
  v7 = &priority_table;
  v8 = &_loc_buf_10;
  v9 = &_loc_buf_10_xfer;
  mem_read32(&v9->f0, v5.buf + v5.offs, 16);
  v5.offs += 14;
  *(v8) = *(v9);
  v10 = &_loc_buf_11;
  v11 = &_loc_buf_11_xfer;
  mem_read32(&v11->f0, v5.buf + v5.offs, 24);
  v5.offs += 24;
  *(v10) = *(v11);
  v12 = &_loc_buf_12;
  v13 = &_loc_buf_12_xfer;
  mem_read32(&v13->f0, v5.buf + v5.offs, 20);
  v5.offs += 20;
  *(v12) = *(v13);
  v14 = v10->f6;
  v15 = v10->f7;
  v16 = v14 + v15;
  v17 = v12->f0;
  v18 = v16 + v17;
  v19 = v12->f1;
  v20 = v18 + v19;
  v21 = &_loc_buf_13;
  *v21 = v6->table[lmem_cam_lookup(firewall_tcpport_table_index, v20, 64)];
  v22 = &_loc_buf_14;
  *v22 = v7->table[lmem_cam_lookup(priority_table_index, v20, 64)];
  v23 = v21->f4;
  v24 = v22->f0;
  v25 = v24 + v24;
  v26 = v21->f3;
  v27 = &v4->f9;
  v28 = &_loc_buf_12;
  v29 = &_loc_buf_12_xfer;
  *(v29) = *(v28);
  cls_write(&v29->f0, &v27->f0, 20);
  v4->f8 = v14;
  v4->f4 = v20;
  v4->f3 = v25;
  v4->f0 = v17;
  v30 = &next_work_NET_RECV_1;
  v30->ctx = v4;
  v30->f0 = v23;
  v30->f1 = v26;
  v30->f2 = v5;
  v31 = &next_work_ref_NET_RECV_1;
  *(v31) = *(v30);
  dispatch1();
  return;
}


int main(void) {
	init_me_cam(16);
	wait_global_start_();
	for (;;) {
		__event___handler_NET_RECV_process_packet_3();
	}
}
