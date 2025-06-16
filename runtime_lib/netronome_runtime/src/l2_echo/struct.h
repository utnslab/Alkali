#ifndef _STRUCT_H_
#define _STRUCT_H_
// TODO: alignment opt, reference 7.1.2.2 Overriding Natural Alignment in UG_nfp6000_nfcc.pdf
__packed struct bits48_t {
  uint8_t data[6];
};

#define ORI_eth_header_t_SIZE 14
__packed struct eth_header_t {
    struct bits48_t dst_mac;
    struct bits48_t src_mac;
    uint16_t ether_type;
    uint8_t padding[2]; // padding is necessary. Since Netronome's register needs to be declared as 4 bytes aligned for mem read operarion.
};

#define ORI_ip_header_t_SIZE 24
__packed struct ip_header_t {
    uint16_t misc;
    uint16_t length;
    uint16_t identification;
    uint16_t fragment_offset;
    uint16_t TTL_transport;
    uint16_t checksum;
    uint32_t source_ip;
    uint32_t dst_ip;
    uint32_t options;
};

#endif
