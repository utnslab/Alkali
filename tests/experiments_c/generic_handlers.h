#ifndef ALKALI_GENERIC_HANDLERS_H
#define ALKALI_GENERIC_HANDLERS_H

#include "alkali.h"

void EXT__NET_SEND__net_send(buf_t packet);
void EXT__DMA_WRITE_REQ__dma_write(buf_t packet, void *dma_cmd);

#ifndef ALKALI_GENERIC_HANDLERS_NO_IMPL
void EXT__NET_SEND__net_send(buf_t packet) { }
void EXT__DMA_WRITE_REQ__dma_write(buf_t packet, void *dma_cmd) { }
#endif // ALKALI_GENERIC_HANDLERS_NO_IMPL

#endif // ALKALI_GENERIC_HANDLERS_H