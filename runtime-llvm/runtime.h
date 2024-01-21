#ifndef _EP2_LLVM_RUNTIME_H_
#define _EP2_LLVM_RUNTIME_H_

#include <pthread.h>
#include "glib.h"

#define NDEBUG
#ifndef NDEBUG
  #define dprintf(...) printf(__VA_ARGS__)
#else
  #define dprintf(...)
#endif

// Queues.
extern GAsyncQueue **queues;

// thread workers
typedef void * (*handler_worker_t)(void *);
#define WORKER_FUNCTION(tag,handler_func) \
    void * __thread ## handler_func (void * tid) { \
        while (1) { \
            void * event = g_async_queue_pop(queues[tag]); \
            printf("[thread %s][%lu][qid: %d] got event %p\n", #handler_func, (uintptr_t)tid, tag, event); \
            handler_func(event); \
        } \
        pthread_exit(NULL); \
    }

typedef struct {
  const char * name;
  int num_ins, num_outs;
  int ins[16];
  int outs[16];
} extern_worker_t;

typedef struct {
  char *data;
  int size;
  int offset;
} buf_t;

void __rt_generate(int tag, int size, void *data);
void __rt_buf_init(buf_t *buf);
void __rt_buf_free(void *buf);
void * __rt_buf_extract(buf_t *buf, int size);
void __rt_buf_emit(buf_t *buf, int size, void *data);
void __rt_buf_concat(buf_t *buf, buf_t *other);

// Standard libs
typedef struct {
  int vsize;
  void * el;
} table_t;

void __rt_table_init(table_t * table, int vsize);
void * __rt_table_lookup(table_t * table, int key);
void __rt_table_update(table_t * table, int key, void * value);


#endif // _EP2_LLVM_RUNTIME_H_