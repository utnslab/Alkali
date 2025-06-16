
#ifndef _EVENTQ_CONFIG_H_
#define _EVENTQ_CONFIG_H_

#include "nfplib.h"
#include "struct.h"
#include "extern/extern_dma.h"

struct event_param_USER_EVENT1
{
    unsigned int context_idx; // context id in the context ring
    uint32_t param;
};

// USER_EVENT1 controller
#define WORKQ_ID_USER_EVENT1 11
#define WORKQ_SIZE_USER_EVENT1 256
#define WORKQ_TYPE_USER_EVENT1 MEM_TYEP_CLS
CLS_WORKQ_DECLARE(workq_USER_EVENT1, WORKQ_SIZE_USER_EVENT1);
#endif