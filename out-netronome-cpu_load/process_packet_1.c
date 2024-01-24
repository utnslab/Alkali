#define DO_CTXQ_INIT

#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct coremap_t _loc_buf_0;
__xrw static struct coremap_t _loc_buf_0_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_NET_RECV work;
__xrw struct event_param_NET_RECV work_ref;
__declspec(aligned(4)) struct event_param_LOAD_TABLE_ADD next_work_LOAD_TABLE_ADD;
__xrw struct event_param_LOAD_TABLE_ADD next_work_ref_LOAD_TABLE_ADD;

__forceinline
void __event___handler_NET_RECV_process_packet_1() {
  uint32_t v1;
  uint16_t v2;
  uint16_t v3;
  uint16_t v4;
  uint16_t v5;
  uint16_t v6;
  uint32_t v7;
  __declspec(aligned(4)) struct event_param_NET_RECV* v8;
  __shared __cls struct context_chain_1_t* v9;
  __shared __cls struct context_chain_1_t* v10;
  struct __buf_t v11;
  __export __shared __cls struct table_i32_coremap_t_16_t* v12;
  struct coremap_t* v13;
  uint16_t v14;
  uint16_t v15;
  char v16;
  uint16_t v17;
  uint16_t v18;
  uint32_t v19;
  uint16_t v20;
  uint16_t v21;
  char v22;
  uint16_t v23;
  uint32_t v24;
  char v25;
  uint16_t v26;
  __declspec(aligned(4)) struct event_param_LOAD_TABLE_ADD* v27;
  __xrw struct event_param_LOAD_TABLE_ADD* v28;
  v1 = 3;
  v2 = 4;
  v3 = 3;
  v4 = 2;
  v5 = 1;
  v6 = 0;
  v7 = 0;
  v8 = &work;
  inlined_net_recv(v8);
  v9 = alloc_context_chain_ring_entry();
  v8->ctx = v9;
  v10 = v8->ctx;
  v11 = v8->f0;
  v12 = &service_load;
  v13 = &_loc_buf_0;
  *v13 = v12->table[me_cam_lookup(v7)];
  v14 = v13->f0;
  v15 = v13->f1;
  v16 = v14 < v15;
  v17 = (v16 ? v5 : v4);
  v18 = (v16 ? v14 : v15);
  v19 = (uint32_t) v18;
  v20 = v13->f2;
  v21 = v13->f3;
  v22 = v20 < v21;
  v23 = (v22 ? v3 : v2);
  v24 = (uint32_t) v21;
  v25 = v19 < v24;
  v26 = (v25 ? v17 : v23);
  v27 = &next_work_LOAD_TABLE_ADD;
  v27->ctx = v10;
  v27->f0 = v6;
  v27->f1 = v26;
  v28 = &next_work_ref_LOAD_TABLE_ADD;
  *(v28) = *(v27);
  cls_workq_add_work(WORKQ_ID_LOAD_TABLE_ADD_1, v28, sizeof(*v28));
  return;
}


int main(void) {
	init_me_cam(16);
	wait_global_start_();
	for (;;) {
		__event___handler_NET_RECV_process_packet_1();
	}
}
