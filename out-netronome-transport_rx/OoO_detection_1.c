#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct ack_info_t _loc_buf_36;
__xrw static struct ack_info_t _loc_buf_36_xfer;
static struct dma_write_cmd_t _loc_buf_35;
__xrw static struct dma_write_cmd_t _loc_buf_35_xfer;
static struct flow_state_t lookup_buf_40;
__xrw static struct flow_state_t lookup_buf_40_xfer;
__declspec(aligned(4)) struct event_param_OoO_DETECT work;
__xrw struct event_param_OoO_DETECT work_ref;
__declspec(aligned(4)) struct event_param_ACK_GEN next_work_ACK_GEN;
__xrw struct event_param_ACK_GEN next_work_ref_ACK_GEN;
__declspec(aligned(4)) struct event_param_DMA_WRITE_REQ next_work_DMA_WRITE_REQ;
__xrw struct event_param_DMA_WRITE_REQ next_work_ref_DMA_WRITE_REQ;

__forceinline
void __event___handler_OoO_DETECT_OoO_detection_1() {
  uint64_t v1;
  uint32_t v2;
  uint32_t v3;
  uint32_t v4;
  __declspec(aligned(4)) struct event_param_OoO_DETECT* v5;
  __xrw struct event_param_OoO_DETECT* v6;
  __shared __cls struct context_chain_1_t* v7;
  struct pkt_info_t* v8;
  __shared __lmem struct table_i16_flow_state_t_16_t* v9;
  uint32_t v10;
  uint16_t v11;
  struct flow_state_t* v12;
  struct dma_write_cmd_t* v13;
  struct ack_info_t* v14;
  uint32_t v15;
  uint32_t v16;
  uint32_t v17;
  uint32_t v18;
  uint32_t v19;
  uint32_t v20;
  char v21;
  uint32_t v22;
  uint32_t v23;
  uint32_t v24;
  char v25;
  char v26;
  uint32_t v27;
  uint32_t v28;
  struct flow_state_t* v29;
  uint32_t v30;
  uint32_t v31;
  struct flow_state_t* v32;
  uint32_t v33;
  uint32_t v34;
  struct flow_state_t* v35;
  struct dma_write_cmd_t* v36;
  struct dma_write_cmd_t* v37;
  struct __buf_t v38;
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
  cls_workq_add_thread(WORKQ_ID_OoO_DETECT_1, v6, sizeof(*v6));
  *(v5) = *(v6);
  v7 = v5->ctx;
  v8 = &v5->f0;
  v9 = &table_34;
  v10 = v8->f0;
  v11 = (uint16_t) v10;
  v12 = &lookup_buf_40;
  *v12 = v9->table[me_cam_lookup(v11)];
  v13 = &_loc_buf_35;
  v14 = &_loc_buf_36;
  v15 = v12->f5;
  v16 = v8->f2;
  v17 = v15 - v16;
  v18 = v8->f1;
  v19 = v18 - v17;
  v20 = v12->f4;
  v21 = v19 < v20;
  if (v21) {
    goto label2;
  } else {
    goto label3;
  }
label2:
  v49 = v4;
  goto label4;
label3:
  v22 = v19 - v20;
  v49 = v22;
  goto label4;
label4:
  goto label5;
label5:
  v23 = v17 + v49;
  v24 = v18 - v23;
  v25 = v17 <= v18;
  if (v25) {
    goto label6;
  } else {
    goto label11;
  }
label6:
  v26 = v24 > v1;
  if (v26) {
    goto label7;
  } else {
    goto label8;
  }
label7:
  v27 = v12->f6;
  v28 = v20 - v24;
  v12->f4 = v28;
  v30 = v12->f5;
  v31 = v30 + v24;
  v12->f5 = v31;
  v33 = v12->f6;
  v34 = v33 + v24;
  v12->f6 = v34;
  v13->f0 = v27;
  v13->f1 = v24;
  v38 = v7->f3;
  v39 = &next_work_DMA_WRITE_REQ;
  v39->ctx = v7;
  v39->f0 = v38;
  v39->f1 = *v13;
  v40 = &next_work_ref_DMA_WRITE_REQ;
  *(v40) = *(v39);
  cls_workq_add_work(WORKQ_ID_DMA_WRITE_REQ, v40, sizeof(*v40));
  v50 = v12;
  goto label9;
label8:
  v50 = v12;
  goto label9;
label9:
  goto label10;
label10:
  v9->table[me_cam_update(v11)] = *v50;
  v41 = v50->f0;
  v14->f0 = v41;
  v43 = v50->f5;
  v14->f1 = v43;
  v45 = v50->f4;
  v14->f2 = v45;
  v47 = &next_work_ACK_GEN;
  v47->ctx = v7;
  v47->f0 = *v14;
  v48 = &next_work_ref_ACK_GEN;
  *(v48) = *(v47);
  cls_workq_add_work(WORKQ_ID_ACK_GEN_1, v48, sizeof(*v48));
  goto label11;
label11:
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_OoO_DETECT_1, workq_OoO_DETECT_1, WORKQ_TYPE_OoO_DETECT, WORKQ_SIZE_OoO_DETECT, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_OoO_DETECT_OoO_detection_1();
	}
}
