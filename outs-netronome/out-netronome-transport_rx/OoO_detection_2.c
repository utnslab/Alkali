#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct ack_info_t _loc_buf_25;
__xrw static struct ack_info_t _loc_buf_25_xfer;
static struct flow_state_t _loc_buf_23;
__xrw static struct flow_state_t _loc_buf_23_xfer;
static struct dma_write_cmd_t _loc_buf_24;
__xrw static struct dma_write_cmd_t _loc_buf_24_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_OoO_DETECT work;
__xrw struct event_param_OoO_DETECT work_ref;
__declspec(aligned(4)) struct event_param_ACK_GEN next_work_ACK_GEN;
__xrw struct event_param_ACK_GEN next_work_ref_ACK_GEN;
__declspec(aligned(4)) struct event_param_DMA_WRITE_REQ next_work_DMA_WRITE_REQ;
__xrw struct event_param_DMA_WRITE_REQ next_work_ref_DMA_WRITE_REQ;

__forceinline
void __event___handler_OoO_DETECT_OoO_detection_2() {
  uint32_t v1;
  uint32_t v2;
  uint32_t v3;
  __declspec(aligned(4)) struct event_param_OoO_DETECT* v4;
  __xrw struct event_param_OoO_DETECT* v5;
  __shared __cls struct context_chain_1_t* v6;
  struct pkt_info_t* v7;
  struct __buf_t v8;
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
  __declspec(aligned(4)) struct event_param_DMA_WRITE_REQ* v38;
  __xrw struct event_param_DMA_WRITE_REQ* v39;
  uint32_t v40;
  struct ack_info_t* v41;
  uint32_t v42;
  struct ack_info_t* v43;
  uint32_t v44;
  struct ack_info_t* v45;
  __declspec(aligned(4)) struct event_param_ACK_GEN* v46;
  __xrw struct event_param_ACK_GEN* v47;
  uint32_t v48;
  struct flow_state_t* v49;
  v1 = 1;
  v2 = 4;
  v3 = 0;
  v4 = &work;
  v5 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_OoO_DETECT_2, v5, sizeof(*v5));
  *(v4) = *(v5);
  v6 = v4->ctx;
  v7 = &v4->f0;
  v8 = v6->f0;
  v9 = &table_35;
  v10 = v7->f0;
  v11 = (uint16_t) v10;
  v12 = &_loc_buf_23;
  *v12 = v9->table[me_cam_lookup(v11)];
  v13 = &_loc_buf_24;
  v14 = &_loc_buf_25;
  v15 = v12->f5;
  v16 = v7->f2;
  v17 = v15 - v16;
  v18 = v7->f1;
  v19 = v18 - v17;
  v20 = v12->f4;
  v21 = v19 < v20;
  if (v21) {
    v48 = v3;
    goto label3;
  } else {
    goto label2;
  }
label2:
  v22 = v19 - v20;
  v48 = v22;
  goto label3;
label3:
  v23 = v17 + v48;
  v24 = v18 - v23;
  v25 = v17 <= v18;
  if (v25) {
    goto label4;
  } else {
    goto label7;
  }
label4:
  v26 = v24 > v3;
  if (v26) {
    goto label5;
  } else {
    v49 = v12;
    goto label6;
  }
label5:
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
  v38 = &next_work_DMA_WRITE_REQ;
  v38->ctx = v6;
  v38->f0 = v8;
  v38->f1 = *v13;
  v39 = &next_work_ref_DMA_WRITE_REQ;
  *(v39) = *(v38);
  cls_workq_add_work(WORKQ_ID_DMA_WRITE_REQ, v39, sizeof(*v39));
  v49 = v12;
  goto label6;
label6:
  v9->table[me_cam_update(v11)] = *v49;
  v40 = v49->f0;
  v14->f0 = v40;
  v42 = v49->f5;
  v14->f1 = v42;
  v44 = v49->f4;
  v14->f2 = v44;
  v46 = &next_work_ACK_GEN;
  v46->ctx = v6;
  v46->f0 = *v14;
  v47 = &next_work_ref_ACK_GEN;
  *(v47) = *(v46);
  cls_workq_add_work(WORKQ_ID_ACK_GEN_2, v47, sizeof(*v47));
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
