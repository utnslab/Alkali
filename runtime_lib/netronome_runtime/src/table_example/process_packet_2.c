#define DO_CTXQ_INIT

#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct flow_state_t _loc_buf_1;
__xrw static struct flow_state_t _loc_buf_1_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_NET_RECV work;
__xrw struct event_param_NET_RECV work_ref;
__declspec(aligned(4)) struct event_param_NET_SEND next_work_NET_SEND;
__xrw struct event_param_NET_SEND next_work_ref_NET_SEND;

__shared __lmem struct table_i16_flow_state_t_16_2t flow_table; // the table value can be in lmem or cls (lmem in this example)
__shared __lmem struct flowht_entry_t flowhash_cache[SIZE_table_i16_flow_state_t_16_2t]; // index now we assume always in lmem
__forceinline
void __event___handler_NET_RECV_process_packet_2() {
  uint32_t v1;
  uint16_t v2;
  __declspec(aligned(4)) struct event_param_NET_RECV* v3;
  __shared __cls struct context_chain_1_t* v4;
  __shared __cls struct context_chain_1_t* v5;
  struct __buf_t v6;
  __shared __lmem struct table_i16_flow_state_t_16_2t* v7;
  struct flow_state_t* v8;
  __xrw struct flow_state_t* v9;
  __declspec(aligned(4)) struct event_param_NET_SEND* v10;
  v1 = 0;
  v2 = 1;
  v3 = &work;
  inlined_net_recv(v3);
  v4 = alloc_context_chain_ring_entry();
  v3->ctx = v4;
  v5 = v3->ctx;
  v6 = v3->f0;
  v7 = &flow_table;
  v8 = &_loc_buf_1;
  *v8 = v7->table[lmem_cam_lookup(flowhash_cache, v2, SIZE_table_i16_flow_state_t_16_2t)];
  v9 = &_loc_buf_1_xfer;
  *(v9) = *(v8);
  mem_write32(&v9->f0, v6.buf + v6.offs, 68);
  v6.offs += 68;
  v10 = &next_work_NET_SEND;
  v10->ctx = v5;
  v10->f0 = v6;
  next_work_NET_SEND.meta.len = next_work_NET_SEND.f0.sz;
  inlined_net_send(v10);
  return;
}


int main(void) {
	init_me_cam(16);
	wait_global_start_();
	for (;;) {
		__event___handler_NET_RECV_process_packet_2();
	}
}
