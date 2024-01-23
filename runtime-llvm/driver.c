#include "runtime.h"
#include "extern.h"
#include "ep2.inc.h"

const int MSG_SIZE = 256;

// table name
void __handler_NET_RECV_process_packet(buf_t *buf);

int main() {
  // init table
  __ep2_init();

  // init message
  buf_t buf = {.data = malloc(MSG_SIZE), .size = 0, .offset = 0};
  __handler_NET_RECV_process_packet(&buf);
}