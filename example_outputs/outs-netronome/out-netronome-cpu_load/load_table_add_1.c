#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

static struct coremap_t _loc_buf_1;
__xrw static struct coremap_t _loc_buf_1_xfer;
static int rr_ctr = 0;
__declspec(aligned(4)) struct event_param_LOAD_TABLE_ADD work;
__xrw struct event_param_LOAD_TABLE_ADD work_ref;

__forceinline
void __event___handler_LOAD_TABLE_ADD_load_table_add_1() {
  uint16_t v1;
  uint16_t v2;
  uint16_t v3;
  uint16_t v4;
  __declspec(aligned(4)) struct event_param_LOAD_TABLE_ADD* v5;
  __xrw struct event_param_LOAD_TABLE_ADD* v6;
  __shared __cls struct context_chain_1_t* v7;
  uint16_t v8;
  uint16_t v9;
  __export __shared __cls struct table_i32_coremap_t_16_t* v10;
  uint32_t v11;
  struct coremap_t* v12;
  char v13;
  uint16_t v14;
  uint16_t v15;
  struct coremap_t* v16;
  char v17;
  uint16_t v18;
  uint16_t v19;
  struct coremap_t* v20;
  char v21;
  uint16_t v22;
  uint16_t v23;
  struct coremap_t* v24;
  char v25;
  uint16_t v26;
  uint16_t v27;
  struct coremap_t* v28;
  struct coremap_t* v29;
  struct coremap_t* v30;
  struct coremap_t* v31;
  struct coremap_t* v32;
  v1 = 4;
  v2 = 3;
  v3 = 2;
  v4 = 1;
  v5 = &work;
  v6 = &work_ref;
  cls_workq_add_thread(WORKQ_ID_LOAD_TABLE_ADD_1, v6, sizeof(*v6));
  *(v5) = *(v6);
  v7 = v5->ctx;
  v8 = v5->f0;
  v9 = v5->f1;
  v10 = &service_load;
  v11 = (uint32_t) v8;
  v12 = &_loc_buf_1;
  *v12 = v10->table[me_cam_lookup(v11)];
  v13 = v9 == v4;
  if (v13) {
    goto label2;
  } else {
    v29 = v12;
    goto label3;
  }
label2:
  v14 = v12->f0;
  v15 = v14 + v4;
  v12->f0 = v15;
  v29 = v12;
  goto label3;
label3:
  v17 = v9 == v3;
  if (v17) {
    goto label4;
  } else {
    v30 = v29;
    goto label5;
  }
label4:
  v18 = v29->f1;
  v19 = v18 + v4;
  v29->f1 = v19;
  v30 = v29;
  goto label5;
label5:
  v21 = v9 == v2;
  if (v21) {
    goto label6;
  } else {
    v31 = v30;
    goto label7;
  }
label6:
  v22 = v30->f2;
  v23 = v22 + v4;
  v30->f2 = v23;
  v31 = v30;
  goto label7;
label7:
  v25 = v9 == v1;
  if (v25) {
    goto label8;
  } else {
    v32 = v31;
    goto label9;
  }
label8:
  v26 = v31->f3;
  v27 = v26 + v4;
  v31->f3 = v27;
  v32 = v31;
  goto label9;
label9:
  v10->table[me_cam_update(v11)] = *v32;
  return;
}


int main(void) {
	init_me_cam(16);
	init_recv_event_workq(WORKQ_ID_LOAD_TABLE_ADD_1, workq_LOAD_TABLE_ADD_1, WORKQ_TYPE_LOAD_TABLE_ADD, WORKQ_SIZE_LOAD_TABLE_ADD, 8);
	wait_global_start_();
	for (;;) {
		__event___handler_LOAD_TABLE_ADD_load_table_add_1();
	}
}
