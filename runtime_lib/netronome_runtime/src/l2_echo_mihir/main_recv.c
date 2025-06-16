#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

struct eth_header_t _loc_buf_0;
__xrw struct eth_header_t _loc_buf_0_xfer;
__declspec(aligned(4)) struct event_param_NET_RECV work;
__xrw struct event_param_NET_RECV work_ref;
struct __wrapper_arg_t wrap_in;
__declspec(aligned(4)) struct event_param_NET_SEND next_work;
__xrw struct event_param_NET_SEND next_work_ref;
struct __wrapper_arg_t wrap_out;

__forceinline
void __event___handler_NET_RECV_main_recv(struct __wrapper_arg_t* v1, struct __wrapper_arg_t* v2) {
  int32_t v3;
  struct event_param_NET_RECV* v4;
  struct context_chain_1_t* v5;
  char* v6;
  char* v7;
  struct eth_header_t* v8;
  __xrw struct eth_header_t* v9;
  int48_t v10;
  int48_t v11;
  struct eth_header_t* v12;
  struct eth_header_t* v13;
  __xrw struct eth_header_t* v14;
  struct event_param_NET_SEND* v15;
  v3 = 0;
  v4 = v1->f1;
  v5 = v4->ctx;
  v6 = v4->f0;
  v7 = alloc_packet_buffer();
  v8 = &_loc_buf_0;
  v9 = &_loc_buf_0_xfer;
  mem_read32(&v9->f0, v6+0, 12);
  mem_read8(&v9->f2, v6+12, 2);
  *(v8) = *(v9);
  v10 = v8->f1;
  v11 = v8->f0;
  v8->f1 = v11;
  v8->f0 = v10;
  v14 = &_loc_buf_0_xfer;
  *(v14) = *(v8);
  mem_write32(&v14->f0, v7+0, 12);
  mem_write8(&v14->f2, v7+12, 2);
  bulk_memcpy(v7+14, v6+14, (work.meta.len)-14);
  v15 = &next_work;
  v2->f0 = v3;
  v2->f1 = v15;
  v15->ctx = v5;
  v15->f0 = v7;
  return;
}


int main(void) {
	init_context_chain_ring();
  wait_global_start_();
	for (;;) {
		inlined_net_recv(&work);
		wrap_in.f1 = &work;
		work.ctx = alloc_context_chain_ring_entry();
		__event___handler_NET_RECV_main_recv(&wrap_in, &wrap_out);
		next_work.meta.len = work.meta.len+0;
		inlined_net_send(&next_work);
	}
}
