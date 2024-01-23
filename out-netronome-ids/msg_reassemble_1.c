#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct agg_t _loc_buf_4;
__xrw static struct agg_t _loc_buf_4_xfer;
static struct agg_t _loc_buf_5;
__xrw static struct agg_t _loc_buf_5_xfer;
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
  __shared __lmem struct table_i16_agg_t_16_t* v8;
  __shared __lmem struct table_i16___buf_t_16_t* v9;
  struct agg_t* v10;
  struct __buf_t v11;
  uint32_t v12;
  uint16_t v13;
  struct agg_t* v14;
  struct __buf_t v15;
  uint32_t v16;
  char v17;
  uint32_t v18;
  struct agg_t* v19;
  uint32_t v20;
  uint32_t v21;
  uint32_t v22;
  char v23;
  struct agg_t* v24;
  uint32_t v25;
  uint32_t v26;
  uint32_t v27;
  struct agg_t* v28;
  uint32_t v29;
  char v30;
  struct agg_t* v31;
  __declspec(aligned(4)) struct event_param_NET_SEND* v32;
  struct agg_t* v33;
  struct agg_t* v34;
  v1 = 1;
  v2 = 0;
  v3 = &work;
  v4 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_MSG_REASSEMBLE_1, v4, sizeof(*v4));
  *(v3) = *(v4);
  v5 = v3->ctx;
  v6 = v3->f0;
  v7 = &v3->f1;
  v8 = &table_6;
  v9 = &table_7;
  v10 = &_loc_buf_4;
  v11 = alloc_packet_buf();
  v12 = v7->f0;
  v13 = (uint16_t) v12;
  v14 = &_loc_buf_5;
  *v14 = v8->table[me_cam_lookup(v13)];
  v15 = v9->table[me_cam_lookup(v13)];
  v16 = v14->f0;
  v17 = v16 == v2;
  if (v17) {
    goto label2;
  } else {
    v33 = v14;
    goto label3;
  }
label2:
  v18 = v7->f3;
  v14->f0 = v18;
  v33 = v14;
  goto label3;
label3:
  v20 = v33->f1;
  v21 = v20 + v1;
  v22 = v7->f1;
  v23 = v21 == v22;
  if (v23) {
    goto label4;
  } else {
    v34 = v33;
    goto label5;
  }
label4:
  v33->f1 = v21;
  bulk_memcpy(v15.buf + v15.offs, v6.buf + v6.offs, v6.sz - v6.offs);
  v15.offs += v6.sz - v6.offs;
  v25 = v33->f0;
  v26 = v7->f2;
  v27 = v25 - v26;
  v33->f0 = v27;
  v34 = v33;
  goto label5;
label5:
  v29 = v34->f0;
  v30 = v29 > v2;
  if (v30) {
    goto label6;
  } else {
    goto label7;
  }
label6:
  v8->table[me_cam_update(v13)] = *v34;
  v9->table[me_cam_update(v13)] = v15;
  goto label8;
label7:
  v10->f0 = v2;
  v8->table[me_cam_update(v13)] = *v10;
  v9->table[me_cam_update(v13)] = v11;
  v32 = &next_work_NET_SEND;
  v32->ctx = v5;
  v32->f0 = v15;
  next_work_NET_SEND.meta.len = next_work_NET_SEND.f0.sz;
  inlined_net_send(v32);
  goto label8;
label8:
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
