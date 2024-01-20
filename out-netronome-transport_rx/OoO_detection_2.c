#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct dma_write_cmd_t _loc_buf_24;
__xrw static struct dma_write_cmd_t _loc_buf_24_xfer;
static struct ack_info_t _loc_buf_25;
__xrw static struct ack_info_t _loc_buf_25_xfer;
static struct flow_state_t _loc_buf_23;
__xrw static struct flow_state_t _loc_buf_23_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_OoO_DETECT work;
__xrw struct event_param_OoO_DETECT work_ref;
__declspec(aligned(4)) struct event_param_ACK_GEN next_work_ACK_GEN;
__xrw struct event_param_ACK_GEN next_work_ref_ACK_GEN;
__declspec(aligned(4)) struct event_param_DMA_WRITE_REQ next_work_DMA_WRITE_REQ;
__xrw struct event_param_DMA_WRITE_REQ next_work_ref_DMA_WRITE_REQ;

__forceinline
void __event___handler_OoO_DETECT_OoO_detection_2() {
  uint64_t v1;
  uint32_t v2;
  uint32_t v3;
  uint32_t v4;
  __declspec(aligned(4)) struct event_param_OoO_DETECT* v5;
  __xrw struct event_param_OoO_DETECT* v6;
  __shared __cls struct context_chain_1_t* v7;
  struct pkt_info_t* v8;
  struct __buf_t v9;
  __shared __lmem struct table_i16_flow_state_t_16_t* v10;
  uint32_t v11;
  uint16_t v12;
  struct flow_state_t* v13;
  struct dma_write_cmd_t* v14;
  struct ack_info_t* v15;
  uint32_t v16;
  uint32_t v17;
  uint32_t v18;
  uint32_t v19;
  uint32_t v20;
  uint32_t v21;
  char v22;
  uint32_t v23;
  uint32_t v24;
  uint32_t v25;
  char v26;
  char v27;
  uint32_t v28;
  uint32_t v29;
  struct flow_state_t* v30;
  uint32_t v31;
  uint32_t v32;
  struct flow_state_t* v33;
  uint32_t v34;
  uint32_t v35;
  struct flow_state_t* v36;
  struct dma_write_cmd_t* v37;
  struct dma_write_cmd_t* v38;
  __declspec(aligned(4)) struct event_param_DMA_WRITE_REQ* v39;
  __xrw struct event_param_DMA_WRITE_REQ* v40;
  uint32_t v41;
  struct ack_info_t* v42;
  uint32_t v43;
  struct ack_info_t* v44;
  uint32_t v45;
  struct ack_info_t* v46;
  __declspec(aligned(4)) struct event_param_ACK_GEN* v47;
  __xrw struct event_param_ACK_GEN* v48;
  uint32_t v49;
  struct flow_state_t* v50;
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
  v9 = v7->f0;
  v10 = &table_35;
  v11 = v8->f0;
  v12 = (uint16_t) v11;
  v13 = &_loc_buf_23;
  *v13 = v10->table[me_cam_lookup(v12)];
  v14 = &_loc_buf_24;
  v15 = &_loc_buf_25;
  v16 = v13->f5;
  v17 = v8->f2;
  v18 = v16 - v17;
  v19 = v8->f1;
  v20 = v19 - v18;
  v21 = v13->f4;
  v22 = v20 < v21;
  if (v22) {
    v49 = v4;
    goto label3;
  } else {
    goto label2;
  }
label2:
  v23 = v20 - v21;
  v49 = v23;
  goto label3;
label3:
  v24 = v18 + v49;
  v25 = v19 - v24;
  v26 = v18 <= v19;
  if (v26) {
    goto label4;
  } else {
    goto label7;
  }
label4:
  v27 = v25 > v1;
  if (v27) {
    goto label5;
  } else {
    v50 = v13;
    goto label6;
  }
label5:
  v28 = v13->f6;
  v29 = v21 - v25;
  v13->f4 = v29;
  v31 = v13->f5;
  v32 = v31 + v25;
  v13->f5 = v32;
  v34 = v13->f6;
  v35 = v34 + v25;
  v13->f6 = v35;
  v14->f0 = v28;
  v14->f1 = v25;
  v39 = &next_work_DMA_WRITE_REQ;
  v39->ctx = v7;
  v39->f0 = v9;
  v39->f1 = *v14;
  v40 = &next_work_ref_DMA_WRITE_REQ;
  *(v40) = *(v39);
  cls_workq_add_work(WORKQ_ID_DMA_WRITE_REQ, v40, sizeof(*v40));
  v50 = v13;
  goto label6;
label6:
  v10->table[me_cam_update(v12)] = *v50;
  v41 = v50->f0;
  v15->f0 = v41;
  v43 = v50->f5;
  v15->f1 = v43;
  v45 = v50->f4;
  v15->f2 = v45;
  v47 = &next_work_ACK_GEN;
  v47->ctx = v7;
  v47->f0 = *v15;
  v48 = &next_work_ref_ACK_GEN;
  *(v48) = *(v47);
  cls_workq_add_work(WORKQ_ID_ACK_GEN_2, v48, sizeof(*v48));
  goto label7;
label7:
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
