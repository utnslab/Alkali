#include "runtime.h"
#include "extern.h"
#include "ep2.inc.h"
#include "stdint.h"
#include <stdio.h>

const int MSG_SIZE = 512;

// table name
void __handler_NET_RECV_process_packet(buf_t *buf);

void run_ids() {
  void * bufd = aligned_alloc(64, MSG_SIZE);
  uint8_t data[] = {
    0xB8,0x3F,0xD2,0x54,0xBE,0x7B,0xE8,0xEB,0xD3,0xF7,0x79,0x5F,0x08,0x00,0x45,0x00,0x00,0x4C,0xA1,0xE9,0x40,0x00,0x40,0x11,0x01,0x1D,0xC0,0xA8,0x0B,0x11,0xC0,0xA8,0x0B,0x64,0x23,0x28,0x03,0xEC,0x00,0x30,0xFD,0x27,0x01,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x20,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
  };

  for (int i = 0; i < 8; i++) {
    memcpy(bufd, data, sizeof(data));
    buf_t buf = {.data = bufd, .size = sizeof(data), .offset = 0};
    __handler_NET_RECV_process_packet(&buf);
    printf("finish processing packet %d, buf.offset = %d\n",i, buf.offset);
  }
}

int main() {
  // init table
  __ep2_init();
  uint8_t data[] = {0xb8,0x3f,0xd2,0x54,0xbe,0x7b,0xe8,0xeb,0xd3,0xf7,0x79,0x5f,0x8,0x0,0x45,0x0,0x0,0x21,0xa1,0xe9,0x40,0x0,0x40,0x11,0x1,0x1d,0xc0,0xa8,0xb,0x11,0xc0,0xa8,0xb,0x64,0x23,0x28,0x3,0xec,0x0,0xd,0xfd,0x27,0x68,0x65,0x6c,0x6c,0x6f,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0};
  

  // init message
  run_ids();
}
