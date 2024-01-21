#ifndef _EP2_RUNTIME_EXTERN_H_
#define _EP2_RUNTIME_EXTERN_H_

#include "runtime.h"

struct event_NET_SEND {
  void *ctx;
  buf_t buf;
};

struct event_NET_RECV {
  void *ctx;
  buf_t buf;
};

typedef void * (*pthread_worker_t)(void *);
void * __extern_source_NET_RECV(extern_worker_t *worker);
void * __extern_sink_NET_SEND(extern_worker_t *worker);

void __handler_DMA_WRITE_REQ_dma_write(void *);
void __handler_NET_SEND_net_send(void *, void *);

#endif // _EP2_RUNTIME_EXTERN_H_