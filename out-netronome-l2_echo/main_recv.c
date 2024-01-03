#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

struct eth_header_t _loc_buf_0;
__xrw struct eth_header_t _loc_buf_0_xfer;
__declspec(aligned(4)) struct event_param_NET_RECV work;
__xrw struct event_param_NET_RECV work_ref;
__declspec(aligned(4)) struct event_param_NET_SEND next_work_NET_SEND;
__xrw struct event_param_NET_SEND next_work_ref_NET_SEND;

__forceinline
void __event___handler_NET_RECV_main_recv() {
  int32_t v1;
  __declspec(aligned(4)) struct event_param_NET_RECV* v2;
  struct context_chain_1_t* v3;
  struct context_chain_1_t* v4;
  struct __buf_t v5;
  struct __buf_t v6;
  struct eth_header_t* v7;
  __xrw struct eth_header_t* v8;
  int48_t v9;
  int48_t v10;
  struct eth_header_t* v11;
  struct eth_header_t* v12;
  __xrw struct eth_header_t* v13;
  __declspec(aligned(4)) struct event_param_NET_SEND* v14;
  v1 = 0;
  v2 = &work;
  inlined_net_recv(v2);
  v3 = alloc_context_chain_ring_entry();
  v2->ctx = v3;
  v4 = v2->ctx;
  v5 = v2->f0;
  v6 = alloc_packet_buf();
  v7 = &_loc_buf_0;
  v8 = &_loc_buf_0_xfer;
  mem_read32(&v8->f0, v5.buf + v5.offs, 12);
  v5.offs += 12;
  mem_read8(&v8->f2, v5.buf + v5.offs, 2);
  v5.offs += 2;
  *(v7) = *(v8);
  v9 = v7->f1;
  v10 = v7->f0;
  v7->f1 = v10;
  v7->f0 = v9;
  v13 = &_loc_buf_0_xfer;
  *(v13) = *(v7);
  mem_write32(&v13->f0, v6.buf + v6.offs, 12);
  v6.offs += 12;
  mem_write8(&v13->f2, v6.buf + v6.offs, 2);
  v6.offs += 2;
  bulk_memcpy(v6.buf + v6.offs, v5.buf + v5.offs, work.meta.len - v5.offs);
  v6.offs += work.meta.len - v5.offs;
  v14 = &next_work_NET_SEND;
  v14->ctx = v4;
  v14->f0 = v6;
  next_work_NET_SEND.meta.len = next_work_NET_SEND.f0.offs;
  inlined_net_send(v14);
  return;
}


int main(void) {
	init_context_chain_ring();
	for (;;) {
		__event___handler_NET_RECV_main_recv();
	}
}
