#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct lb_fwd_tcp_hdr_t _loc_buf_55;
__xrw static struct lb_fwd_tcp_hdr_t _loc_buf_55_xfer;
static struct lb_DIP_entries_t _loc_buf_54;
__xrw static struct lb_DIP_entries_t _loc_buf_54_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_OoO_DETECT2 work;
__xrw struct event_param_OoO_DETECT2 work_ref;
__declspec(aligned(4)) struct event_param_NET_SEND next_work_NET_SEND;
__xrw struct event_param_NET_SEND next_work_ref_NET_SEND;

__forceinline
void __event___handler_OoO_DETECT2_OoO_detection2_4() {
  uint32_t v1;
  uint32_t v2;
  uint16_t v3;
  uint16_t v4;
  uint32_t v5;
  uint32_t v6;
  __declspec(aligned(4)) struct event_param_OoO_DETECT2* v7;
  __xrw struct event_param_OoO_DETECT2* v8;
  __shared __cls struct context_chain_1_t* v9;
  struct __buf_t v10;
  uint32_t v11;
  __shared __lmem struct table_i32_lb_DIP_entries_t_64_t* v12;
  struct lb_DIP_entries_t* v13;
  struct lb_DIP_entries_t* v14;
  uint64_t v15;
  uint64_t v16;
  struct lb_DIP_entries_t* v17;
  uint64_t v18;
  uint64_t v19;
  struct lb_DIP_entries_t* v20;
  uint32_t v21;
  struct lb_DIP_entries_t* v22;
  uint32_t v23;
  struct lb_DIP_entries_t* v24;
  uint16_t v25;
  struct lb_DIP_entries_t* v26;
  uint16_t v27;
  struct lb_DIP_entries_t* v28;
  uint32_t v29;
  uint32_t v30;
  uint32_t v31;
  uint16_t v32;
  uint32_t v33;
  uint16_t v34;
  uint32_t v35;
  __export __shared __cls struct table_i32_lb_fwd_tcp_hdr_t_64_t* v36;
  struct lb_fwd_tcp_hdr_t* v37;
  __xrw struct lb_fwd_tcp_hdr_t* v38;
  __declspec(aligned(4)) struct event_param_NET_SEND* v39;
  v1 = 0;
  v2 = 1;
  v3 = 60;
  v4 = 50;
  v5 = 134744071;
  v6 = 134744072;
  v7 = &work;
  v8 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_OoO_DETECT2_4, v8, sizeof(*v8));
  *(v7) = *(v8);
  v9 = v7->ctx;
  v10 = v7->f0;
  v11 = v7->f1;
  v12 = &lb_table;
  v13 = &_loc_buf_54;
  *v13 = v12->table[lmem_cam_lookup(lb_table_index, v11, 64)];
  v13->f6 = v2;
  v15 = v13->f0;
  v16 = v15 + v11;
  v13->f0 = v16;
  v18 = v13->f1;
  v19 = v18 + v11;
  v13->f1 = v19;
  v21 = v6 + v11;
  v13->f2 = v21;
  v23 = v5 + v11;
  v13->f3 = v23;
  v25 = v4 + v11;
  v13->f4 = v25;
  v27 = v3 + v11;
  v13->f5 = v27;
  v12->table[lmem_cam_update(lb_table_index, v11, 64)] = *v13;
  v29 = v13->f2;
  v30 = v13->f3;
  v31 = v29 + v30;
  v32 = v13->f4;
  v33 = v31 + v32;
  v34 = v13->f5;
  v35 = v33 + v34;
  v36 = &lb_fwd_table;
  v37 = &_loc_buf_55;
  *v37 = v36->table[lmem_cam_lookup(lb_fwd_table_index, v35, 64)];
  v38 = &_loc_buf_55_xfer;
  *(v38) = *(v37);
  mem_write32(&v38->f0, v10.buf + v10.offs, 20);
  v10.offs += 20;
  v39 = &next_work_NET_SEND;
  v39->ctx = v9;
  v39->f0 = v10;
  next_work_NET_SEND.meta.len = next_work_NET_SEND.f0.sz;
  inlined_net_send(v39);
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_OoO_DETECT2_4, workq_OoO_DETECT2_4, WORKQ_TYPE_OoO_DETECT2, WORKQ_SIZE_OoO_DETECT2, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_OoO_DETECT2_OoO_detection2_4();
	}
}
