#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct ack_info_t _loc_buf_7;
__xrw static struct ack_info_t _loc_buf_7_xfer;
static struct flow_state_t _loc_buf_5;
__xrw static struct flow_state_t _loc_buf_5_xfer;
static struct dma_write_cmd_t _loc_buf_6;
__xrw static struct dma_write_cmd_t _loc_buf_6_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_OoO_DETECT work;
__xrw struct event_param_OoO_DETECT work_ref;
__declspec(aligned(4)) struct event_param_ACK_GEN next_work_ACK_GEN;
__xrw struct event_param_ACK_GEN next_work_ref_ACK_GEN;
__declspec(aligned(4)) struct event_param_DMA_WRITE_REQ next_work_DMA_WRITE_REQ;
__xrw struct event_param_DMA_WRITE_REQ next_work_ref_DMA_WRITE_REQ;

__forceinline
void __event___handler_OoO_DETECT_OoO_detection_1() {
  uint32_t v1;
  uint32_t v2;
  uint32_t v3;
  __declspec(aligned(4)) struct event_param_OoO_DETECT* v4;
  __xrw struct event_param_OoO_DETECT* v5;
  __shared __cls struct context_chain_1_t* v6;
  struct pkt_info_t* v7;
  struct __buf_t v8;
  uint32_t v9;
  __shared __lmem struct table_i16_flow_state_t_16_t* v10;
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
  uint32_t v26;
  uint32_t v27;
  struct flow_state_t* v28;
  uint32_t v29;
  uint32_t v30;
  struct flow_state_t* v31;
  uint32_t v32;
  uint32_t v33;
  struct flow_state_t* v34;
  struct dma_write_cmd_t* v35;
  struct dma_write_cmd_t* v36;
  __declspec(aligned(4)) struct event_param_DMA_WRITE_REQ* v37;
  __xrw struct event_param_DMA_WRITE_REQ* v38;
  uint32_t v39;
  struct ack_info_t* v40;
  uint32_t v41;
  struct ack_info_t* v42;
  uint32_t v43;
  struct ack_info_t* v44;
  __declspec(aligned(4)) struct event_param_ACK_GEN* v45;
  __xrw struct event_param_ACK_GEN* v46;
  uint32_t v47;
  struct flow_state_t* v48;
  v1 = 1;
  v2 = 4;
  v3 = 0;
  v4 = &work;
  v5 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_OoO_DETECT_1, v5, sizeof(*v5));
  *(v4) = *(v5);
  v6 = v4->ctx;
  v7 = &v4->f0;
  v8 = v6->f0;
  v9 = v7->f0;
  v10 = &table_18;
  v11 = (uint16_t) v9;
  v12 = &_loc_buf_5;
  *v12 = v10->table[me_cam_lookup(v11)];
  v13 = &_loc_buf_6;
  v14 = &_loc_buf_7;
  v15 = v12->f5;
  v16 = v7->f2;
  v17 = v15 - v16;
  v18 = v7->f1;
  v19 = v18 - v17;
  v20 = v12->f4;
  v21 = v19 < v20;
  if (v21) {
    v47 = v3;
    goto label3;
  } else {
    goto label2;
  }
label2:
  v22 = v19 - v20;
  v47 = v22;
  goto label3;
label3:
  v23 = v17 + v47;
  v24 = v18 - v23;
  v25 = v24 > v3;
  if (v25) {
    goto label4;
  } else {
    v48 = v12;
    goto label5;
  }
label4:
  v26 = v12->f6;
  v27 = v20 - v24;
  v12->f4 = v27;
  v29 = v12->f5;
  v30 = v29 + v24;
  v12->f5 = v30;
  v32 = v12->f6;
  v33 = v32 + v24;
  v12->f6 = v33;
  v13->f0 = v26;
  v13->f1 = v24;
  v37 = &next_work_DMA_WRITE_REQ;
  v37->ctx = v6;
  v37->f0 = v8;
  v37->f1 = *v13;
  v38 = &next_work_ref_DMA_WRITE_REQ;
  *(v38) = *(v37);
  cls_workq_add_work(WORKQ_ID_DMA_WRITE_REQ, v38, sizeof(*v38));
  v48 = v12;
  goto label5;
label5:
  v10->table[me_cam_update(v11)] = *v48;
  v39 = v48->f0;
  v14->f0 = v39;
  v41 = v48->f5;
  v14->f1 = v41;
  v43 = v48->f4;
  v14->f2 = v43;
  v45 = &next_work_ACK_GEN;
  v45->ctx = v6;
  v45->f0 = *v14;
  v46 = &next_work_ref_ACK_GEN;
  *(v46) = *(v45);
  cls_workq_add_work(WORKQ_ID_ACK_GEN_1, v46, sizeof(*v46));
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
