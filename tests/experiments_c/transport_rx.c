#include "alkali.h"
#include "generic_handlers.h"

struct pkt_info_t {
  BITS_FIELD(32, flow_id);
  BITS_FIELD(32, pk_len);
  BITS_FIELD(32, pk_seq);
};

struct eth_header_t {
  BITS_FIELD(32, dst_mac_1);
  BITS_FIELD(16, dst_mac_2);
  BITS_FIELD(32, src_mac_1);
  BITS_FIELD(16, src_mac_2);
  BITS_FIELD(16, ether_type);
};


struct ip_header_t {
  BITS_FIELD(16, misc);
  BITS_FIELD(16, length);
  BITS_FIELD(16, identification);
  BITS_FIELD(16, fragment_offset);
  BITS_FIELD(16, TTL_transport);
  BITS_FIELD(16, checksum);
  BITS_FIELD(32, source_ip);
  BITS_FIELD(32, dst_ip);
  BITS_FIELD(32, options);
};


struct tcp_header_t {
  BITS_FIELD(16, sport);                   // /** Source port */
  BITS_FIELD(16, dport);                   // /** Destination port */

  BITS_FIELD(32, seq);                     // /** Sequence number */

  BITS_FIELD(32, ack);                     // /** Acknowledgement number */

  BITS_FIELD(8,  off);                   // /** Data offset */
  BITS_FIELD(8,  flags);                   // /** Flags */
  BITS_FIELD(16, win);                     // /** Window */

  BITS_FIELD(16, sum);                     // /** Checksum */
  BITS_FIELD(16, urp);                     // /** Urgent pointer */
};


struct flow_state_t
{
    BITS_FIELD(32, tx_next_seq);             // /*, Sequence number of next byte to be sent */
    BITS_FIELD(16, flags);                   // /*, RX/TX Flags */
    BITS_FIELD(16, dupack_cnt);              // /*, Duplicate ACK count */
    BITS_FIELD(32, rx_len);                  // /*, Length of receive buffer */
    BITS_FIELD(32, rx_avail);                // /*, Available RX buffer space */
    BITS_FIELD(32, rx_next_seq);             // /*, Next sequence number expected */
    BITS_FIELD(32, rx_next_pos);             // /*, Offset of next byte in RX buffer */
    BITS_FIELD(32, rx_ooo_len);              // /*, Length of Out-of-Order bytes */
    BITS_FIELD(32, rx_ooo_start);            // /*, Start position of Out-of-Order bytes */
};

struct dma_write_cmd_t {
  BITS_FIELD(32, addr);
  BITS_FIELD(32, size);
};

ak_TABLE(64, BITS(32), struct flow_state_t,) flow_table;

void NET_RECV__process_packet(buf_t packet) {
  struct eth_header_t eth_header;
  struct ip_header_t ip_header;
  struct tcp_header_t tcp_header;
  struct pkt_info_t pkt_info;

  bufextract(packet, (void *)&eth_header);
  bufextract(packet, (void *)&ip_header);
  bufextract(packet, (void *)&tcp_header);

  pkt_info.pk_len = ip_header.length + 14;
  pkt_info.pk_seq = tcp_header.seq;
  pkt_info.flow_id = tcp_header.dport;

  struct flow_state_t flow_state;
  
  int flow_id = pkt_info.flow_id;
  table_lookup(&flow_table, &flow_id, &flow_state);

  int trim_start = flow_state.rx_next_seq - pkt_info.pk_seq;
  int trim_end;
  
  // TODO: aIR support "if else", but C frontend currently don't have support. TODO is to add support for "if else" in C frontend.

  // if ((pkt_info.pk_len - trim_start) < flow_state.rx_avail) {
  //   trim_end = 0;
  // } else {
    trim_end = pkt_info.pk_len - trim_start - flow_state.rx_avail;
  // }
  int payload_bytes = pkt_info.pk_len - (trim_start + trim_end);

  // if (trim_start <= pkt_info.pk_len) {
    // if (payload_bytes > 0) {
      int dma_pos = flow_state.rx_next_pos;
      flow_state.rx_avail =  payload_bytes;
      flow_state.rx_next_seq = flow_state.rx_next_seq + payload_bytes;
      flow_state.rx_next_pos = flow_state.rx_next_pos + payload_bytes;

      struct dma_write_cmd_t dma_cmd;
      dma_cmd.addr = dma_pos;
      dma_cmd.size = payload_bytes;
      // EXT__DMA_WRITE_REQ__dma_write(packet, &dma_cmd);
    // }
    table_update(&flow_table, &flow_id, &flow_state);

    // generate ack
    tcp_header.seq = flow_state.tx_next_seq;
    tcp_header.ack = flow_state.rx_next_seq;
    tcp_header.win = flow_state.rx_avail;

    struct buf_tag packet_out;
    bufemit(&packet_out, &eth_header);
    bufemit(&packet_out, &ip_header);
    bufemit(&packet_out, &tcp_header);
    bufemit(&packet_out, packet);

    EXT__NET_SEND__net_send(&packet_out);
  // }
}
