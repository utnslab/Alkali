#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct agg_t _loc_buf_7;
__xrw static struct agg_t _loc_buf_7_xfer;
static struct agg_t _loc_buf_6;
__xrw static struct agg_t _loc_buf_6_xfer;
static struct ip_header_t _loc_buf_5;
__xrw static struct ip_header_t _loc_buf_5_xfer;
static struct eth_header_t _loc_buf_4;
__xrw static struct eth_header_t _loc_buf_4_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_MSG_REASSEMBLE work;
__xrw struct event_param_MSG_REASSEMBLE work_ref;
__declspec(aligned(4)) struct event_param_NET_SEND next_work_NET_SEND;
__xrw struct event_param_NET_SEND next_work_ref_NET_SEND;

__forceinline
void __event___handler_MSG_REASSEMBLE_msg_reassemble_1() {
  uint32_t v1;
  uint32_t v2;
  __declspec(aligned(4)) struct event_param_MSG_REASSEMBLE* v3;
  __xrw struct event_param_MSG_REASSEMBLE* v4;
  __shared __cls struct context_chain_1_t* v5;
  struct __buf_t v6;
  struct rpc_header_t* v7;
  __shared __cls struct eth_header_t* v8;
  struct eth_header_t* v9;
  __xrw struct eth_header_t* v10;
  __shared __cls struct ip_header_t* v11;
  struct ip_header_t* v12;
  __xrw struct ip_header_t* v13;
  __shared __lmem struct table_i16_agg_t_16_t* v14;
  __shared __lmem struct table_i16___buf_t_16_t* v15;
  struct agg_t* v16;
  uint32_t v17;
  uint16_t v18;
  struct agg_t* v19;
  struct __buf_t v20;
  uint32_t v21;
  char v22;
  uint32_t v23;
  struct agg_t* v24;
  __xrw struct eth_header_t* v25;
  __xrw struct ip_header_t* v26;
  uint32_t v27;
  uint32_t v28;
  struct agg_t* v29;
  uint32_t v30;
  uint32_t v31;
  uint32_t v32;
  struct agg_t* v33;
  uint32_t v34;
  char v35;
  struct __buf_t v36;
  struct agg_t* v37;
  __declspec(aligned(4)) struct event_param_NET_SEND* v38;
  struct agg_t* v39;
  v1 = 1;
  v2 = 0;
  v3 = &work;
  v4 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_MSG_REASSEMBLE_1, v4, sizeof(*v4));
  *(v3) = *(v4);
  v5 = v3->ctx;
  v6 = v3->f0;
  v7 = &v3->f1;
  v8 = &v5->f1;
  v9 = &_loc_buf_4;
  v10 = &_loc_buf_4_xfer;
  cls_read(&v10->f0, &v8->f0, 16);
  *(v9) = *(v10);
  v11 = &v5->f0;
  v12 = &_loc_buf_5;
  v13 = &_loc_buf_5_xfer;
  cls_read(&v13->f0, &v11->f0, 20);
  *(v12) = *(v13);
  v14 = &table_8;
  v15 = &table_9;
  v16 = &_loc_buf_6;
  v17 = v7->f0;
  v18 = (uint16_t) v17;
  v19 = &_loc_buf_7;
  *v19 = v14->table[me_cam_lookup(v18)];
  v20 = v15->table[me_cam_lookup(v18)];
  v21 = v19->f0;
  v22 = v21 == v2;
  if (v22) {
    goto label2;
  } else {
    v39 = v19;
    goto label3;
  }
label2:
  v23 = v7->f3;
  v19->f0 = v23;
  v25 = &_loc_buf_4_xfer;
  *(v25) = *(v9);
  mem_write32(&v25->f0, v20.buf + v20.offs, 12);
  v20.offs += 12;
  mem_write8(&v25->f2, v20.buf + v20.offs, 2);
  v20.offs += 2;
  v26 = &_loc_buf_5_xfer;
  *(v26) = *(v12);
  mem_write32(&v26->f0, v20.buf + v20.offs, 20);
  v20.offs += 20;
  v39 = v19;
  goto label3;
label3:
  v27 = v39->f1;
  v28 = v27 + v1;
  v39->f1 = v28;
  bulk_memcpy(v20.buf + v20.offs, v6.buf + v6.offs, v6.sz - v6.offs);
  v20.offs += v6.sz - v6.offs;
  v30 = v39->f0;
  v31 = v7->f2;
  v32 = v30 - v31;
  v39->f0 = v32;
  v34 = v39->f0;
  v35 = v34 > v2;
  if (v35) {
    goto label4;
  } else {
    goto label5;
  }
label4:
  v14->table[me_cam_update(v18)] = *v39;
  v15->table[me_cam_update(v18)] = v20;
  goto label6;
label5:
  v36 = alloc_packet_buf();
  v16->f0 = v2;
  v14->table[me_cam_update(v18)] = *v16;
  v15->table[me_cam_update(v18)] = v36;
  v38 = &next_work_NET_SEND;
  v38->ctx = v5;
  v38->f0 = v20;
  next_work_NET_SEND.meta.len = next_work_NET_SEND.f0.sz;
  inlined_net_send(v38);
  goto label6;
label6:
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_MSG_REASSEMBLE_1, workq_MSG_REASSEMBLE_1, WORKQ_TYPE_MSG_REASSEMBLE, WORKQ_SIZE_MSG_REASSEMBLE, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_MSG_REASSEMBLE_msg_reassemble_1();
	}
}
