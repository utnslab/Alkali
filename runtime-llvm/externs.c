#include <stdio.h>

#include <pthread.h>
#include "runtime.h"
#include "glib.h"
#include "extern.h"

// built in source/sink/handlers
void * __extern_source_NET_RECV(extern_worker_t *worker) {
  int tag = worker->outs[0];
  while (1) {
    struct event_NET_RECV *event = malloc(sizeof(struct event_NET_RECV));
    __rt_buf_init(&(event->buf));
    // TODO: fill in some source
    dprintf("[sink %d]: event %p, with buf %p\n", tag, event, event->buf.data);
    g_async_queue_push(queues[tag], (void *)event);
  }
  pthread_exit(NULL);
}

void * __extern_sink_NET_SEND(extern_worker_t *worker) {
  int tag = worker->ins[0];
  static int counter = 0;
  while (1) {
    struct event_NET_SEND * event = (struct event_NET_SEND *)g_async_queue_pop(queues[tag]);
    __rt_buf_free(&(event->buf));
    free(event);
    dprintf("[sink %d]: packet %d, with buf %p\n", tag, counter ++, event->buf.data);
  }
  pthread_exit(NULL);
}