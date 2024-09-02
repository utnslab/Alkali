struct buf_t {
  char *data;
  short len;
};
typedef struct buf_t buf_t;

struct pkt_info_t {
  int mac_src, mac_dst, ip_src, ip_dst, port_src, port_dst;
};

struct flow_tracker {
  int mac_src_cnt, mac_dst_cnt, ip_src_cnt, ip_dst_cnt, port_src_cnt,
      port_dst_cnt;
};

struct firwall_entries {
  int if_allow, new_ip_src, new_ip_dst, priority, timeout;
};

struct firwall_metaheader {
  int if_allow, new_ip_src, new_ip_dst, priority, timeout;
};

struct lb_DIP_entries {
  int if_alloc, mac_src, mac_dst, ip_src, ip_dst, port_src, port_dst;
};

struct _t1_type {
  int key;
  struct flow_tracker value;
};
struct _t1_type *flow_tracker_table;

struct _t2_type {
  int key;
  struct firwall_entries value;
};
struct _t2_type *firewall_table;

struct _t3_type {
  int key;
  struct lb_DIP_entries value;
};
struct _t3_type *lb_DIP_table;

extern void send_packet(buf_t packet);
extern void bufextract(buf_t packet, void *extracted_data);
extern void bufemit(buf_t packet, void *extracted_data);

extern void table_lookup(void *tab, void *key, void *value);
extern void table_update(void *tab, void *key, void *value);

void _packet_event_handler(buf_t packet) {
  struct pkt_info_t pkt_header;
  struct flow_tracker flow_tracker;
  struct firwall_entries firewall_entry;

  // packet tracker network functions
  bufextract(packet, (void *)&pkt_header);
  int flow_hash_id;
  flow_hash_id = pkt_header.mac_src + pkt_header.mac_dst +
                     pkt_header.ip_src + pkt_header.ip_dst +
                     pkt_header.port_src + pkt_header.port_dst;
  table_lookup((void*)flow_tracker_table, (void *)&flow_hash_id,
               (void *)&flow_tracker); // look up mac table, mac
  flow_tracker.mac_src_cnt++;
  flow_tracker.mac_dst_cnt++;
  flow_tracker.ip_src_cnt++;
  flow_tracker.ip_dst_cnt++;
  flow_tracker.port_src_cnt++;
  flow_tracker.port_dst_cnt++;
  table_update((void *)flow_tracker_table, (void *)&flow_hash_id, (void *)&flow_tracker);
  // TODO: can add error counters

  // firewall
  int firewall_hash_id = pkt_header.ip_src + pkt_header.ip_dst;
  table_lookup((void *)firewall_table, (void *)&firewall_hash_id, (void *)&firewall_entry);
  struct firwall_metaheader firewall_metaheader;
  firewall_metaheader.if_allow = firewall_entry.if_allow;
  firewall_metaheader.new_ip_src = firewall_entry.new_ip_src;
  firewall_metaheader.new_ip_dst = firewall_entry.new_ip_dst;
  firewall_metaheader.priority = firewall_entry.priority;
  firewall_metaheader.timeout = firewall_entry.timeout;
  bufemit(packet, (void *)&firewall_metaheader);

  // L4 load balancer
  struct lb_DIP_entries lb_DIP_entry;
  table_lookup((void *)lb_DIP_table, (void *)&flow_hash_id, (void *)&lb_DIP_entry);
  int base_mac_src = 10;
  int base_mac_dst = 20;
  int base_ip_src = 30;
  int base_ip_dst = 40;
  int base_port_src = 50;
  int base_port_dst = 60;

  if (lb_DIP_entry.if_alloc == 1) {
    pkt_header.mac_src = lb_DIP_entry.mac_src;
    pkt_header.mac_dst = lb_DIP_entry.mac_dst;
    pkt_header.ip_src = lb_DIP_entry.ip_src;
    pkt_header.ip_dst = lb_DIP_entry.ip_dst;
    pkt_header.port_src = lb_DIP_entry.port_src;
    pkt_header.port_dst = lb_DIP_entry.port_dst;
  } else {
    // allocate entry for DIP
    lb_DIP_entry.if_alloc = 1;
    lb_DIP_entry.mac_src = base_mac_src + flow_hash_id;
    lb_DIP_entry.mac_dst = base_mac_dst + flow_hash_id;
    lb_DIP_entry.ip_src = base_ip_src + flow_hash_id;
    lb_DIP_entry.ip_dst = base_ip_dst + flow_hash_id;
    lb_DIP_entry.port_src = base_port_src + flow_hash_id;
    lb_DIP_entry.port_dst = base_port_dst + flow_hash_id;
    table_update((void *)lb_DIP_table, (void *)&flow_hash_id, (void *)&lb_DIP_entry);
  }
  bufemit(packet, (void *)&pkt_header);
  send_packet(packet);
}

int main() { return 0; }