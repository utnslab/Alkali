#include "alkali.h"
#include "generic_handlers.h"

struct eth_header_t {
    BITS_FIELD(48, dst_mac);
    BITS_FIELD(48, src_mac);
    BITS_FIELD(16, ether_type);
};

void NET_RECV__process_packet(buf_t packet) {
    struct eth_header_t old_eth_header;
    struct eth_header_t new_eth_header;

    // extract ethernet header from the RX packet
    bufextract(packet, (void *)&old_eth_header);

    // create a new ethernet header for the TX packet
    new_eth_header.dst_mac = old_eth_header.src_mac;
    new_eth_header.src_mac = old_eth_header.dst_mac;
    new_eth_header.ether_type = old_eth_header.ether_type;

    // create the TX packet
    buf_t packet_out = bufinit();
    bufemit(packet_out, &new_eth_header);
    bufemit(packet_out, packet);

    // send the TX packet
    EXT__NET_SEND__net_send(packet_out);
}

