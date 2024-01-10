#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct flow_state_t lookup_buf_29;
__xrw static struct flow_state_t lookup_buf_29_xfer;
static struct dma_write_cmd_t _loc_buf_26;
__xrw static struct dma_write_cmd_t _loc_buf_26_xfer;
static struct ack_info_t _loc_buf_27;
__xrw static struct ack_info_t _loc_buf_27_xfer;
__declspec(aligned(4)) struct event_param_OoO_DETECT work;
__xrw struct event_param_OoO_DETECT work_ref;
__declspec(aligned(4)) struct event_param_ACK_GEN next_work_ACK_GEN;
__xrw struct event_param_ACK_GEN next_work_ref_ACK_GEN;
__declspec(aligned(4)) struct event_param_DMA_WRITE_REQ next_work_DMA_WRITE_REQ;
__xrw struct event_param_DMA_WRITE_REQ next_work_ref_DMA_WRITE_REQ;

__forceinline
void __event___handler_OoO_DETECT_OoO_detection_2() {
  int64_t v1;
  int32_t v2;
  int32_t v3;
  int32_t v4;
  __declspec(aligned(4)) struct event_param_OoO_DETECT* v5;
  __xrw struct event_param_OoO_DETECT* v6;
  struct context_chain_1_t* v7;
  struct pkt_info_t* v8;
  __shared __lmem struct table_i16_flow_state_t_16_t* v9;
  int32_t v10;
  int16_t v11;
  struct flow_state_t* v12;
  struct dma_write_cmd_t* v13;
  struct ack_info_t* v14;
  int32_t v15;
  int32_t v16;
  int32_t v17;
  int32_t v18;
  int32_t v19;
  int32_t v20;
  char v21;
  int32_t v22;
  int32_t v23;
  int32_t v24;
  int32_t v25;
  char v26;
  char v27;
  struct flow_state_t* v28;
  int32_t v29;
  int32_t v30;
  struct flow_state_t* v31;
  int32_t v32;
  int32_t v33;
  struct flow_state_t* v34;
  int32_t v35;
  int32_t v36;
  struct flow_state_t* v37;
  struct dma_write_cmd_t* v38;
  struct dma_write_cmd_t* v39;
  struct __buf_t v40;
  __declspec(aligned(4)) struct event_param_DMA_WRITE_REQ* v41;
  __xrw struct event_param_DMA_WRITE_REQ* v42;
  int32_t v43;
  struct ack_info_t* v44;
  int32_t v45;
  struct ack_info_t* v46;
  int32_t v47;
  struct ack_info_t* v48;
  __declspec(aligned(4)) struct event_param_ACK_GEN* v49;
  __xrw struct event_param_ACK_GEN* v50;
  v1 = 0;
  v2 = 1;
  v3 = 4;
  v4 = 0;
  v5 = &work;
  v6 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_OoO_DETECT_2, v6, sizeof(*v6));
  *(v5) = *(v6);
  v7 = v5->ctx;
  v8 = &v5->f0;
  v9 = &table_25;
  v10 = v8->f0;
  v11 = (int16_t) v10;
  v12 = &lookup_buf_29;
  *v12 = v9->table[me_cam_lookup(v11)];
  v13 = &_loc_buf_26;
  v14 = &_loc_buf_27;
  v15 = v12->f5;
  v16 = v8->f2;
  v17 = v15 - v16;
  v18 = v8->f1;
  v19 = v18 - v17;
  v20 = v12->f4;
  v21 = v19 < v20;
  ;
  if (v21) {
    v22 = v4;
  } else {
    v23 = v19 - v20;
    v22 = v23;
  }
  v24 = v17 + v22;
  v25 = v18 - v24;
  v26 = v17 <= v18;
  if (v26) {
    v27 = v25 > v1;
    ;
    if (v27) {
      v29 = v12->f6;
      v30 = v20 - v25;
      v12->f4 = v30;
      v32 = v12->f5;
      v33 = v32 + v25;
      v12->f5 = v33;
      v35 = v12->f6;
      v36 = v35 + v25;
      v12->f6 = v36;
      v13->f0 = v29;
      v13->f1 = v25;
      v40 = v7->f0;
      v41 = &next_work_DMA_WRITE_REQ;
      v41->ctx = v7;
      v41->f0 = v40;
      v41->f1 = *v13;
      v42 = &next_work_ref_DMA_WRITE_REQ;
      *(v42) = *(v41);
      cls_workq_add_work(WORKQ_ID_DMA_WRITE_REQ, v42, sizeof(*v42));
      v28 = v12;
    } else {
      v28 = v12;
    };
    v9->table[me_cam_update(v11)] = *v28;
    v43 = v28->f0;
    v14->f0 = v43;
    v45 = v28->f5;
    v14->f1 = v45;
    v47 = v28->f4;
    v14->f2 = v47;
    v49 = &next_work_ACK_GEN;
    v49->ctx = v7;
    v49->f0 = *v14;
    v50 = &next_work_ref_ACK_GEN;
    *(v50) = *(v49);
    cls_workq_add_work(WORKQ_ID_ACK_GEN_2, v50, sizeof(*v50));
  }
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_OoO_DETECT_2, workq_OoO_DETECT_2, WORKQ_TYPE_OoO_DETECT, WORKQ_SIZE_OoO_DETECT, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_OoO_DETECT_OoO_detection_2();
	}
}
