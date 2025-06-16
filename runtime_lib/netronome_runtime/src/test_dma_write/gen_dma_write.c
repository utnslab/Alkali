#include "nfplib.h"
#include "prog_hdr.h"
#include "extern/extern_dma.h"
#include "extern/extern_net.h"


__declspec(aligned(4)) struct extern_event_param_DMA_WRITE_REQ next_work;
__xrw struct extern_event_param_DMA_WRITE_REQ next_work_ref;

int main(void) {

  wait_global_start_();
	for (;;) {
    next_work.addr = 128;
    next_work.size = 64;
    next_work_ref = next_work;
		cls_workq_add_work(WORKQ_ID_DMA_WRITE_REQ, &next_work_ref, sizeof(next_work_ref));


	}
}
