struct buf_t {
  char *data;
  short len;
};
typedef struct buf_t buf_t;

struct table_t {
  char *table;
  short size;
};
typedef struct table_t table_t;

struct pkt_info_t {
  int flow_id;
  int pk_len;
  int pk_seq;
};
typedef struct pkt_info_t pkt_info_t;

struct eth_header_t {
  int dst_mac_1;
  short dst_mac_2;
  int src_mac_1;
  short src_mac_2;
  short ether_type;
};
typedef struct eth_header_t eth_header_t;

struct ip_header_t {
  short misc;
  short length;
  short identification;
  short fragment_offset;
  short TTL_transport;
  short checksum;
  int source_ip;
  int dst_ip;
  int options;
};
typedef struct ip_header_t ip_header_t;

struct tcp_header_t {
  short sport;
  short dport;
  int seq;
  int ack;
  char off;
  char flags;
  short win;
  short sum;
  short urp;
};
typedef struct tcp_header_t tcp_header_t;

struct flow_state_t {
  int tx_next_seq;
  short flags;
  short dupack_cnt;
  int rx_len;
  int rx_avail;
  int rx_next_seq;
  int rx_next_pos;
  int rx_ooo_len;
  int rx_ooo_start;
};
typedef struct flow_state_t flow_state_t;

struct dma_write_cmd_t {
  int addr;
  int size;
};
typedef struct dma_write_cmd_t dma_write_cmd_t;

void bufextract(buf_t buf, char *extracted_data) {}

void bufemit(buf_t buf, char *emit_data) {}

void bufemitbuf(buf_t buf, buf_t emit_buf) {}

void table_lookup(table_t tab, int key, char *value) {}

void table_update(table_t tab, int key, char *value) {}

void _dma_write_event(buf_t packet, dma_write_cmd_t dma_cmd) {}

void _packet_send_event(buf_t packet) {}

table_t flow_table;

void _packet_event_handler(buf_t packet) {
  eth_header_t eth_header;
  ip_header_t ip_header;
  tcp_header_t tcp_header;
  pkt_info_t pkt_info;

  bufextract(packet, &eth_header);
  bufextract(packet, &ip_header);
  bufextract(packet, &tcp_header);

  pkt_info.pk_len = ip_header.length + 14;
  pkt_info.pk_seq = tcp_header.seq;
  pkt_info.flow_id = tcp_header.dport;

  flow_state_t flow_state;
  table_lookup(flow_table, pkt_info.flow_id, &flow_state);

  int trim_start = flow_state.rx_next_seq - pkt_info.pk_seq;
  int trim_end;
  if ((pkt_info.pk_len - trim_start) < flow_state.rx_avail) {
    trim_end = 0;
  } else {
    trim_end = pkt_info.pk_len - trim_start - flow_state.rx_avail;
  }
  int payload_bytes = pkt_info.pk_len - (trim_start + trim_end);

  if (trim_start <= pkt_info.pk_len) {
    if (payload_bytes > 0) {
      int dma_pos = flow_state.rx_next_pos;
      flow_state.rx_avail =  payload_bytes;
      flow_state.rx_next_seq = flow_state.rx_next_seq + payload_bytes;
      flow_state.rx_next_pos = flow_state.rx_next_pos + payload_bytes;

      dma_write_cmd_t dma_cmd;
      dma_cmd.addr = dma_pos;
      dma_cmd.size = payload_bytes;
      _dma_write_event(packet, dma_cmd);
    }
    table_update(flow_table, pkt_info.flow_id, &flow_state);

    // generate ack
    tcp_header.seq = flow_state.tx_next_seq;
    tcp_header.ack = flow_state.rx_next_seq;
    tcp_header.win = flow_state.rx_avail;

    buf_t packet_out;
    bufemit(packet_out, &eth_header);
    bufemit(packet_out, &ip_header);
    bufemit(packet_out, &tcp_header);
    bufemitbuf(packet_out, packet);

    _packet_send_event(packet_out);
  }
}

int main(){
  return 0;
}