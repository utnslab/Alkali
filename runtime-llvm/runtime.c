#include <stdio.h>
#include <stdlib.h>

#include "runtime.h"
#include "glib.h"

GAsyncQueue **queues;

// queue related runtime
void __rt_generate(int tag, int size, void *data) {
    // TODO(zhiyuang): here we do a datacopy. but we could avoid the malloc, by keep a message pool
    void *msg = malloc(size);
    memcpy(msg, data, size);
    g_async_queue_push(queues[tag], msg);
}

// buffer related runtime
void __rt_buf_init(buf_t * buf) {
  buf->data = malloc(1024);
  buf->size = 1024;
  buf->offset = 0;
}

void __rt_buf_free(void *buf) {
  free(((buf_t *)buf)->data);
}

void * __rt_buf_extract(buf_t *buf, int size) {
  void * data = buf->data;
  buf->offset += size;
  return data;
}

void __rt_buf_emit(buf_t *buf, int size, void *data) {
  if (buf->offset + size > buf->size) {
    buf->size = buf->size * 2;
    buf->data = realloc(buf->data, buf->size);
  }
  memcpy(buf->data + buf->offset, data, size);
  buf->offset += size;
}

void __rt_buf_concat(buf_t *buf, buf_t *other) {
  int size = other->size - other->offset;
  if (buf->offset + size > buf->size) {
    buf->size = buf->size * 2;
    buf->data = realloc(buf->data, buf->size);
  }
  // copy the remaining data from other
  memcpy(buf->data + buf->offset, other->data + other->offset, size);
  buf->offset += other->offset;
  __rt_buf_free(other);
}
