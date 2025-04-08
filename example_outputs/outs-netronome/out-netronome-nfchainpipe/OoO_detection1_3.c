#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct tcp_tracker_t _loc_buf_41;
__xrw static struct tcp_tracker_t _loc_buf_41_xfer;
static struct flow_tracker_t _loc_buf_40;
__xrw static struct flow_tracker_t _loc_buf_40_xfer;
static struct err_tracker_t _loc_buf_43;
__xrw static struct err_tracker_t _loc_buf_43_xfer;
static struct ip_tracker_t _loc_buf_42;
__xrw static struct ip_tracker_t _loc_buf_42_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_OoO_DETECT1 work;
__xrw struct event_param_OoO_DETECT1 work_ref;
__declspec(aligned(4)) struct event_param_OoO_DETECT2 next_work_OoO_DETECT2;
__xrw struct event_param_OoO_DETECT2 next_work_ref_OoO_DETECT2;

__forceinline
void __event___handler_OoO_DETECT1_OoO_detection1_3() {
  uint32_t v1;
  uint32_t v2;
  __declspec(aligned(4)) struct event_param_OoO_DETECT1* v3;
  __xrw struct event_param_OoO_DETECT1* v4;
  __shared __cls struct context_chain_1_t* v5;
  struct __buf_t v6;
  uint32_t v7;
  uint32_t v8;
  uint16_t v9;
  __shared __lmem struct table_i32_flow_tracker_t_64_t* v10;
  struct flow_tracker_t* v11;
  uint32_t v12;
  uint32_t v13;
  struct flow_tracker_t* v14;
  __shared __lmem struct table_i32_tcp_tracker_t_64_t* v15;
  uint32_t v16;
  struct tcp_tracker_t* v17;
  uint32_t v18;
  uint32_t v19;
  struct tcp_tracker_t* v20;
  __export __shared __cls struct table_i32_ip_tracker_t_64_t* v21;
  struct ip_tracker_t* v22;
  uint32_t v23;
  uint32_t v24;
  struct ip_tracker_t* v25;
  __export __shared __cls struct table_i32_err_tracker_t_64_t* v26;
  struct err_tracker_t* v27;
  uint32_t v28;
  uint32_t v29;
  struct err_tracker_t* v30;
  __declspec(aligned(4)) struct event_param_OoO_DETECT2* v31;
  __xrw struct event_param_OoO_DETECT2* v32;
  v1 = 4;
  v2 = 1;
  v3 = &work;
  v4 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_OoO_DETECT1_3, v4, sizeof(*v4));
  *(v3) = *(v4);
  v5 = v3->ctx;
  v6 = v3->f0;
  v7 = v3->f1;
  v8 = v3->f2;
  v9 = v3->f3;
  v10 = &flow_tracker_table;
  v11 = &_loc_buf_40;
  *v11 = v10->table[lmem_cam_lookup(flow_tracker_table_index, v7, 64)];
  v12 = v11->f0;
  v13 = v12 + v2;
  v11->f0 = v13;
  v10->table[lmem_cam_update(flow_tracker_table_index, v7, 64)] = *v11;
  v15 = &tcp_tracker_table;
  v16 = (uint32_t) v9;
  v17 = &_loc_buf_41;
  *v17 = v15->table[lmem_cam_lookup(tcp_tracker_table_index, v16, 64)];
  v18 = v17->f0;
  v19 = v18 + v2;
  v17->f0 = v19;
  v15->table[lmem_cam_update(tcp_tracker_table_index, v16, 64)] = *v17;
  v21 = &ip_tracker_table;
  v22 = &_loc_buf_42;
  *v22 = v21->table[lmem_cam_lookup(ip_tracker_table_index, v8, 64)];
  v23 = v22->f0;
  v24 = v23 + v2;
  v22->f0 = v24;
  v21->table[lmem_cam_update(ip_tracker_table_index, v8, 64)] = *v22;
  v26 = &err_tracker_table;
  v27 = &_loc_buf_43;
  *v27 = v26->table[lmem_cam_lookup(err_tracker_table_index, v7, 64)];
  v28 = v27->f0;
  v29 = v28 + v2;
  v27->f0 = v29;
  v26->table[lmem_cam_update(err_tracker_table_index, v7, 64)] = *v27;
  v31 = &next_work_OoO_DETECT2;
  v31->ctx = v5;
  v31->f0 = v6;
  v31->f1 = v7;
  v32 = &next_work_ref_OoO_DETECT2;
  *(v32) = *(v31);
  cls_workq_add_work(WORKQ_ID_OoO_DETECT2_3, v32, sizeof(*v32));
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_OoO_DETECT1_3, workq_OoO_DETECT1_3, WORKQ_TYPE_OoO_DETECT1, WORKQ_SIZE_OoO_DETECT1, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_OoO_DETECT1_OoO_detection1_3();
	}
}
