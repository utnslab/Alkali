#include "runtime.h"
#include "extern.h"

const int MSG_SIZE = 256;

// table name
void __handler_NET_RECV_process_packet(buf_t *buf);
table_t flow_table;

int main() {
  // init table
  __rt_table_init(&flow_table, 32);

  // init message
  buf_t buf = {.data = malloc(MSG_SIZE), .size = 0, .offset = 0};
  __handler_NET_RECV_process_packet(&buf);
}