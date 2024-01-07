#ifndef _EP2_DEFS_
#define _EP2_DEFS_

#include "runtime.h"

// Should be sum(handler_replications)
#define NUM_QUEUES 2
#define NUM_HANDLERS 1
#define NUM_INSTANCES 4

int handler_replications[NUM_HANDLERS] = {4};

void __handler_NET_RECV_main_recv(void * event);
WORKER_FUNCTION(0,__handler_NET_RECV_main_recv);

handler_worker_t handler_workers[NUM_HANDLERS] = {
    __thread__handler_NET_RECV_main_recv,
};

// external workers
#define NUM_EXTERNS 2
extern_worker_t extern_workers[NUM_EXTERNS] = {
    {"source_NET_RECV", 0, 1, {}, {0}},
    {"sink_NET_SEND", 1, 0, {1}, {}},
};

#endif // _EP2_DEFS_