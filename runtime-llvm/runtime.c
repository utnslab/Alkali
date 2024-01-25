#include <stdio.h>
#include <stdlib.h>

#include "runtime.h"
#include "glib.h"
#include <stdlib.h>

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
  printf("+ <buf> init: %p [%p,size:%d,offset:%d]\n", buf, buf->data, buf->size, buf->offset);
}

void __rt_buf_free(void *buf) {
  free(((buf_t *)buf)->data);
}

void * __rt_buf_extract(buf_t *buf, int size) {
  printf("+ <buf> extract: %p [%p,size:%d,offset:%d] size: %d", buf, buf->data, buf->size, buf->offset, size);
  printf("  next 4 bytes: %x %x %x %x\n", buf->data[buf->offset], buf->data[buf->offset+1], buf->data[buf->offset+2], buf->data[buf->offset+3]);
  char * data = buf->data + buf->offset;
  buf->offset += size;
  return data;
}

void __rt_buf_emit(buf_t *buf, int size, void *data) {
  if (buf->offset + size > buf->size) {
    buf->size = buf->size * 2;
    buf->data = realloc(buf->data, buf->size);
  }
  memcpy(buf->data + buf->offset, data, size);
  printf("+ <buf> emit: %p [%p,size:%d,offset:%d] size: %d", buf, buf->data, buf->size, buf->offset, size);
  printf("  next 4 bytes: %x %x %x %x\n", buf->data[buf->offset], buf->data[buf->offset+1], buf->data[buf->offset+2], buf->data[buf->offset+3]);
  buf->offset += size;
}

void __rt_buf_concat(buf_t *buf, buf_t *other) {
  int size = other->size - other->offset;
  printf("+ <buf> concat: %p [%p,size:%d,offset:%d] size: %d\n", buf, buf->data, buf->size, buf->offset, size);
  if (buf->offset + size > buf->size) {
    buf->size = buf->size * 2;
    buf->data = realloc(buf->data, buf->size);
  }
  // copy the remaining data from other
  memcpy(buf->data + buf->offset, other->data + other->offset, size);
  buf->offset += other->offset;
  // __rt_buf_free(other);
}


void * __rt_table_alloc(int vsize) {
  table_t *table = calloc(1, sizeof(table_t));
  __rt_table_init(table, vsize);
  return (void *)table;
}
void __rt_table_init(table_t * table, int vsize) {
  table->vsize = vsize;
  table->el = calloc(1, vsize);
}
void * __rt_table_lookup(table_t * table, int key) {
  printf("+ <table> read: %d:[%d,%d]\n", key, ((int *)table->el)[0], ((int *)table->el)[1]);
  return table->el;
}
void __rt_table_update(table_t * table, int key, void * value) {
  memcpy(table->el, value, table->vsize);
  printf("+ <table> update: %d:[%d,%d]\n", key, ((int *)table->el)[0], ((int *)table->el)[1]);
}
