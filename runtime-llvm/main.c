#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#include <pthread.h>
#include <glib.h>

#include "runtime.h"
#include "extern.h"

#include "ep2_defs.inc.h"

int main() {
  // init queues
  queues = malloc(sizeof(GAsyncQueue *) * NUM_QUEUES);
  for (int i = 0; i < NUM_QUEUES; i++) {
      queues[i] = g_async_queue_new();
  }
  printf("Queues inited\n");

  // enable worker threads
  // for now, each hander have n hreads. merge later
  int num_threads = 0;
  pthread_t threads[NUM_INSTANCES];
  for (int i = 0; i < NUM_HANDLERS; i++) {
      for (int j = 0; j < handler_replications[i]; j++) {
        uintptr_t cur_thread = num_threads++;
        pthread_create(&threads[cur_thread], NULL, handler_workers[i], (void *)(cur_thread));
        printf("worker inited %d %d\n", i, j);
      }
  }

  // enable externs. for now, each extern have 1 thread. merge later
  // runtime could do what ever it wants here
  for (int i = 0; i < NUM_EXTERNS; i++) {
    extern_worker_t *worker = &extern_workers[i];
    if (strcmp(worker->name, "source_NET_RECV") == 0) {
      pthread_t thread;
      pthread_create(&thread, NULL, (pthread_worker_t)__extern_source_NET_RECV,
                     (void *)worker);
      printf("NET RECV Created, QID[%d]\n", worker->outs[0]);
    } else if (strcmp(worker->name, "sink_NET_SEND") == 0) {
      pthread_t thread;
      pthread_create(&thread, NULL, (pthread_worker_t)__extern_sink_NET_SEND,
                     (void *)worker);
      printf("NET SEND Created, QID[%d]\n", worker->ins[0]);
    } else {
      assert("unknown extern");
    }
  }

  // wait here for all threads to finish
  for (int i = 0; i < NUM_INSTANCES; i++) {
      pthread_join(threads[i], NULL);
  }
  assert("should not reach here");
}