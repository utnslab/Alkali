#ifndef ALKALI_GENERIC_HANDLERS_H
#define ALKALI_GENERIC_HANDLERS_H

#include "alkali.h"

void EXT__NET_SEND__net_send(buf_t packet);

#ifndef ALKALI_GENERIC_HANDLERS_NO_IMPL
void EXT__NET_SEND__net_send(buf_t packet) { }
#endif // ALKALI_GENERIC_HANDLERS_NO_IMPL

#endif // ALKALI_GENERIC_HANDLERS_H