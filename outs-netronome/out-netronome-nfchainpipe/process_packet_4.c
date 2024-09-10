#define DO_CTXQ_INIT

#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct firewall_meta_header_t _loc_buf_31;
__xrw static struct firewall_meta_header_t _loc_buf_31_xfer;
static struct priority_entries_t _loc_buf_30;
__xrw static struct priority_entries_t _loc_buf_30_xfer;
static struct priority_entries_t _loc_buf_29;
__xrw static struct priority_entries_t _loc_buf_29_xfer;
static struct eth_header_t _loc_buf_24;
__xrw static struct eth_header_t _loc_buf_24_xfer;
static struct ip_header_t _loc_buf_25;
__xrw static struct ip_header_t _loc_buf_25_xfer;
static struct tcp_header_t _loc_buf_26;
__xrw static struct tcp_header_t _loc_buf_26_xfer;
static struct firewall_ip_entries_t _loc_buf_27;
__xrw static struct firewall_ip_entries_t _loc_buf_27_xfer;
static struct firewall_tcpport_entries_t _loc_buf_28;
__xrw static struct firewall_tcpport_entries_t _loc_buf_28_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_NET_RECV work;
__xrw struct event_param_NET_RECV work_ref;
__declspec(aligned(4)) struct event_param_OoO_DETECT1 next_work_OoO_DETECT1;
__xrw struct event_param_OoO_DETECT1 next_work_ref_OoO_DETECT1;

__forceinline
void __event___handler_NET_RECV_process_packet_4() {
  uint32_t v1;
  __declspec(aligned(4)) struct event_param_NET_RECV* v2;
  __shared __cls struct context_chain_1_t* v3;
  __shared __cls struct context_chain_1_t* v4;
  struct __buf_t v5;
  struct eth_header_t* v6;
  __xrw struct eth_header_t* v7;
  struct ip_header_t* v8;
  __xrw struct ip_header_t* v9;
  struct tcp_header_t* v10;
  __xrw struct tcp_header_t* v11;
  uint32_t v12;
  uint32_t v13;
  uint32_t v14;
  uint16_t v15;
  uint32_t v16;
  uint16_t v17;
  uint32_t v18;
  __shared __lmem struct table_i32_firewall_ip_entries_t_64_t* v19;
  struct firewall_ip_entries_t* v20;
  __export __shared __cls struct table_i16_firewall_tcpport_entries_t_64_t* v21;
  struct firewall_tcpport_entries_t* v22;
  uint32_t v23;
  __export __shared __cls struct table_i32_priority_entries_t_64_t* v24;
  struct priority_entries_t* v25;
  uint32_t v26;
  struct priority_entries_t* v27;
  struct firewall_meta_header_t* v28;
  uint32_t v29;
  uint32_t v30;
  uint32_t v31;
  struct firewall_meta_header_t* v32;
  uint32_t v33;
  uint32_t v34;
  uint32_t v35;
  struct firewall_meta_header_t* v36;
  __xrw struct firewall_meta_header_t* v37;
  __declspec(aligned(4)) struct event_param_OoO_DETECT1* v38;
  __xrw struct event_param_OoO_DETECT1* v39;
  v1 = 3;
  v2 = &work;
  inlined_net_recv(v2);
  v3 = alloc_context_chain_ring_entry();
  v2->ctx = v3;
  v4 = v2->ctx;
  v5 = v2->f0;
  v6 = &_loc_buf_24;
  v7 = &_loc_buf_24_xfer;
  mem_read32(&v7->f0, v5.buf + v5.offs, 16);
  v5.offs += 14;
  *(v6) = *(v7);
  v8 = &_loc_buf_25;
  v9 = &_loc_buf_25_xfer;
  mem_read32(&v9->f0, v5.buf + v5.offs, 24);
  v5.offs += 24;
  *(v8) = *(v9);
  v10 = &_loc_buf_26;
  v11 = &_loc_buf_26_xfer;
  mem_read32(&v11->f0, v5.buf + v5.offs, 20);
  v5.offs += 20;
  *(v10) = *(v11);
  v12 = v8->f6;
  v13 = v8->f7;
  v14 = v12 + v13;
  v15 = v10->f0;
  v16 = v14 + v15;
  v17 = v10->f1;
  v18 = v16 + v17;
  v19 = &firewall_ip_table;
  v20 = &_loc_buf_27;
  *v20 = v19->table[lmem_cam_lookup(firewall_ip_table_index, v12, 64)];
  v21 = &firewall_tcpport_table;
  v22 = &_loc_buf_28;
  *v22 = v21->table[lmem_cam_lookup(firewall_tcpport_table_index, v15, 64)];
  v23 = v22->f2;
  v24 = &priority_table;
  v25 = &_loc_buf_29;
  *v25 = v24->table[lmem_cam_lookup(priority_table_index, v23, 64)];
  v26 = v20->f2;
  v27 = &_loc_buf_30;
  *v27 = v24->table[lmem_cam_lookup(priority_table_index, v26, 64)];
  v28 = &_loc_buf_31;
  v29 = v22->f4;
  v30 = v20->f4;
  v31 = v29 + v30;
  v28->f1 = v31;
  v33 = v27->f0;
  v34 = v25->f0;
  v35 = v33 + v34;
  v28->f2 = v35;
  v37 = &_loc_buf_31_xfer;
  *(v37) = *(v28);
  mem_write32(&v37->f0, v5.buf + v5.offs, 12);
  v5.offs += 12;
  v38 = &next_work_OoO_DETECT1;
  v38->ctx = v4;
  v38->f0 = v5;
  v38->f1 = v18;
  v38->f2 = v12;
  v38->f3 = v15;
  v39 = &next_work_ref_OoO_DETECT1;
  *(v39) = *(v38);
  cls_workq_add_work(WORKQ_ID_OoO_DETECT1_4, v39, sizeof(*v39));
  return;
}


int main(void) {
	init_me_cam(16);
	wait_global_start_();
	for (;;) {
		__event___handler_NET_RECV_process_packet_4();
	}
}
