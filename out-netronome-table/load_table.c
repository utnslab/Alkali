#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"

__declspec(aligned(4)) struct event_param_LOAD_TABLE work;
__xrw struct event_param_LOAD_TABLE work_ref;
struct __wrapper_arg_t wrap_in;
__forceinline
void __event___handler_LOAD_TABLE_load_table(struct __wrapper_arg_t* v1) {
  int32_t v2;
  int16_t v3;
  struct event_param_LOAD_TABLE* v4;
  struct context_chain_1_t* v5;
  struct table_t v6;
  int32_t v7;
  v2 = 11;
  v3 = 10;
  v4 = v1->f1;
  v5 = v4->ctx;
  v6 = create_table();
  __ep2_rt_table_update(v6, v3, v2);
  v7 = __ep2_rt_table_lookup(v6, v3);
  v5->f0 = v7;
  return;
}


int main(void) {
	init_context_chain_ring();
	init_recv_event_workq(WORKQ_ID_LOAD_TABLE, workq_LOAD_TABLE, WORKQ_TYPE_LOAD_TABLE, WORKQ_SIZE_LOAD_TABLE, 8);
	for (;;) {
		cls_workq_add_thread(WORKQ_ID_LOAD_TABLE, &work_ref, sizeof(work_ref));
		work = work_ref;
		wrap_in.f1 = &work;
		work.ctx = alloc_context_chain_ring_entry();
		__event___handler_LOAD_TABLE_load_table(&wrap_in);
	}
}
