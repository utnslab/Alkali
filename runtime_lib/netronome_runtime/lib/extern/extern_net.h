#ifndef _EXTERN_NET_H_
#define _EXTERN_NET_H_

#include "nfplib.h"
#include "extern_net_meta.h"

#define OUT_PORT 0

__packed struct extern_event_param_NET_RECV
{
    uint64_t packet_ptr; // the ptr for packet buf
    unsigned packet_offs;
    unsigned packet_sz;
    struct recv_meta_t meta;
    char* context_p; // context id in the context ring
};

__packed struct extern_event_param_NET_SEND
{
    uint64_t packet_ptr; // the ptr for packet buf
    unsigned packet_offs;
    unsigned packet_sz;
    struct send_meta_t meta;
    char* context_p; // context id in the context ring
};


__forceinline void inlined_net_recv(struct extern_event_param_NET_RECV * event){
    __mem40 char *pbuf;
    __xread struct pkt_raw_t pkt;
    SIGNAL pkt_sig;
    pkt_nbi_recv_with_hdrs(&pkt, sizeof(struct pkt_raw_t), PKT_NBI_OFFSET, sig_done, &pkt_sig);

    pbuf = pkt_ctm_ptr40(pkt.meta.pkt_info.isl, pkt.meta.pkt_info.pnum, 0);

    event->meta.seq = pkt.meta.seq;
    event->meta.len = pkt.meta.pkt_info.len;
    event->packet_ptr = pbuf + PKT_NBI_OFFSET + MAC_PREPEND_BYTES + PKBUF_SHIF_BYTES;
    event->packet_offs = 0;
    event->packet_sz = event->meta.len;
}

/*
 * A 40-bit packet-address mode pointer in CTM is built as follows:
 *
 *  Bits[2;38] -- Must be 0b10 for "direct access"
 *  Bits[6;32] -- The island of the CTM. (can use '0' for the local island)
 *  Bits[1;31] -- Must be set to 1 to enable packet-addressing mode
 *  Bits[6;25] -- reserved
 *  Bits[9;16] -- The packet number of the CTM buffer
 *  Bits[2;14] -- reserved
 *  Bits[14;0] -- The offset within the CTM buffer
 *
 * Unfortunately, this is only partly documented in the NFP DB.
 */

__forceinline unsigned int get_pnum_from_addr(uint64_t addr){
    __gpr uint64_t pnum;
    pnum = addr & 0x7FFF0000;
    pnum = pnum >> 16;
    return pnum;
}

__forceinline void inlined_net_send(struct extern_event_param_NET_SEND * event){
    __gpr struct pkt_ms_info msi;

    event->packet_ptr -= (PKT_NBI_OFFSET + MAC_PREPEND_BYTES + PKBUF_SHIF_BYTES);
    // TODO: Double Check whether Seq matters.
    event->meta.seq = 0;

    pkt_mac_egress_cmd_write(event->packet_ptr, PKT_NBI_OFFSET + MAC_PREPEND_BYTES, 1, 1);
    msi = pkt_msd_write(event->packet_ptr, PKT_NBI_OFFSET + MAC_PREPEND_BYTES);
    pkt_nbi_send(__ISLAND,
                get_pnum_from_addr(event->packet_ptr),
                &msi,
                event->meta.len - MAC_PREPEND_BYTES,
                NBI,
                PORT_TO_TMQ(0),
                0, event->meta.seq, PKT_CTM_SIZE_256);
}

// ID start from 5, 0-4 is reserved for DMA
// Below is unused, if enable inlining when call NET_RECV/NET_SEND
// NET_RECV controller
#define WORKQ_ID_NET_RECV 5
#define WORKQ_SIZE_NET_RECV 256
#define WORKQ_TYPE_NET_RECV MEM_TYEP_CLS

CLS_WORKQ_DECLARE(workq_NET_RECV, WORKQ_SIZE_NET_RECV);

// NET_SEND controller
#define WORKQ_ID_NET_SEND 6
#define WORKQ_SIZE_NET_SEND 256
#define WORKQ_TYPE_NET_SEND MEM_TYEP_CLS
CLS_WORKQ_DECLARE(workq_NET_SEND, WORKQ_SIZE_NET_SEND);

#endif
