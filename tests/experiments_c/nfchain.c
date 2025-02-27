#include "alkali.h"
#include "generic_handlers.h"

struct eth_header_t {
    BITS_FIELD(48, dst_mac);
    BITS_FIELD(48, src_mac);
    BITS_FIELD(16, ether_type);
};

struct ip_header_t {
    // Version (4),  IHL (4), DSCP (6), ECN(2)
    BITS_FIELD(16, misc); 
    BITS_FIELD(16, length);
    BITS_FIELD(16, identification);
    // Flags (3), Fragment Offset (13)
    BITS_FIELD(16, fragment_offset);
    
    // TTL (8), Transport Protocol (8)
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

    BITS_FIELD(16, sport);                   // /** Source port */
    BITS_FIELD(16, dport);                   // /** Destination port */

    BITS_FIELD(32, seq);                     // /** Sequence number */

    BITS_FIELD(32, ack);                     // /** Acknowledgement number */

    BITS_FIELD(8,  off);                     // /** Data offset */
    BITS_FIELD(8,  flags2);                  // /** Flags */
    BITS_FIELD(16, win);                     // /** Window */

    BITS_FIELD(16, sum);                     // /** Checksum */
    BITS_FIELD(16, urp);                     // /** Urgent pointer */
    BITS_FIELD(16, TTL_transport);
    BITS_FIELD(16, checksum);
    BITS_FIELD(32, source_ip);
    BITS_FIELD(32, dst_ip);
    BITS_FIELD(32, options);
};

struct pkt_info_t
{
    BITS_FIELD(64, mac_src);
    BITS_FIELD(64, mac_dst);                  // /*, Length of receive buffer */
    BITS_FIELD(32, ip_src);
    BITS_FIELD(32, ip_dst);
    BITS_FIELD(16, port_src);
    BITS_FIELD(16, port_dst);
};

struct flow_tracker_t
{
    BITS_FIELD(32, total_packet_cnt);
    BITS_FIELD(32, total_byte_cnt);
};

struct err_tracker_t
{
    BITS_FIELD(32, total_drop_cnt);
    BITS_FIELD(32, total_err_cnt);
    BITS_FIELD(32, total_byte_cnt);
};

struct ip_tracker_t
{
    BITS_FIELD(32, ip_src_cnt);
    BITS_FIELD(32, ip_dst_cnt);
    BITS_FIELD(32, ip_outbound_cnt);
};

struct tcp_tracker_t
{
    BITS_FIELD(32, tcp_src_port_cnt);
    BITS_FIELD(32, tcp_dst_port_cnt);
    BITS_FIELD(32, tcp_sync_cnt);
};

struct firewall_ip_entries_t
{
    BITS_FIELD(32, new_ip_src);
    BITS_FIELD(32, new_ip_dst);
    BITS_FIELD(32, priority);
    BITS_FIELD(32, timeout);
    BITS_FIELD(32, if_allow);
};

struct firewall_tcpport_entries_t
{
    BITS_FIELD(32, new_tcp_src);
    BITS_FIELD(32, new_tcp_dst);
    BITS_FIELD(32, priority);
    BITS_FIELD(32, timeout);
    BITS_FIELD(32, if_allow);
};
struct priority_entries_t
{
    BITS_FIELD(32, cookies);
    BITS_FIELD(32, cookiesx);
};


struct firewall_meta_header_t
{
    BITS_FIELD(32, fid);
    BITS_FIELD(32, allow);
    BITS_FIELD(32, acc_priority);
    BITS_FIELD(32, reserved);
    BITS_FIELD(32, reserved1);
    BITS_FIELD(32, reserved2);
    BITS_FIELD(32, reserved3);
};

struct connect_tracker_meta_header_t
{
    BITS_FIELD(32, tcp_tracker);
    BITS_FIELD(32, ip_tracker);
    BITS_FIELD(32, mac_tracker);
    BITS_FIELD(32, reserved);
    BITS_FIELD(32, reserved1);
    BITS_FIELD(32, reserved2);
    BITS_FIELD(32, reserved3);
};

struct lb_DIP_entries_t
{
    BITS_FIELD(64, mac_src);
    BITS_FIELD(64, mac_dst);                  // /*, Length of receive buffer */
    BITS_FIELD(32, ip_src);
    BITS_FIELD(32, ip_dst);
    BITS_FIELD(16, port_src);
    BITS_FIELD(16, port_dst);
    BITS_FIELD(32, if_alloc);
    BITS_FIELD(64, hash);
};

struct lb_fwd_tcp_hdr_t
{
    BITS_FIELD(64, raw1);                  
    BITS_FIELD(64, raw2);                  
    BITS_FIELD(32, raw3);                  
};


ak_TABLE(64, BITS(32), struct firewall_ip_entries_t) firewall_ip_table;
ak_TABLE(64, BITS(32), struct firewall_tcpport_entries_t) firewall_tcpport_table;
ak_TABLE(64, BITS(32), struct priority_entries_t,) priority_table;
ak_TABLE(64, BITS(32), struct flow_tracker_t,) flow_tracker_table;
ak_TABLE(64, BITS(32), struct err_tracker_t,) err_tracker_table;
ak_TABLE(64, BITS(32), struct ip_tracker_t,) ip_tracker_table;
ak_TABLE(64, BITS(32), struct tcp_tracker_t,) tcp_tracker_table;
ak_TABLE(64, BITS(16), struct lb_DIP_entries_t,) lb_table;
ak_TABLE(64, BITS(32), struct lb_fwd_tcp_hdr_t,) lb_fwd_table;

void NET_RECV__process_packet(buf_t packet) {
    struct eth_header_t eth_header;
    struct ip_header_t ip_header;
    struct tcp_header_t tcp_header;
    struct pkt_info_t pkt_info;
    struct firewall_ip_entries_t firewall_ip_entries;
    struct firewall_tcpport_entries_t firewall_tcpport_entries;
    struct priority_entries_t priority_entry1;
    struct priority_entries_t priority_entry2;

    
    bufextract(packet, (void *)&eth_header);
    bufextract(packet, (void *)&ip_header);
    bufextract(packet, (void *)&tcp_header);

    BITS(32) fid;
    fid = ip_header.source_ip + ip_header.dst_ip + tcp_header.sport + tcp_header.dport;

    // firewall logic
    BITS(32) src_ip;
    src_ip = ip_header.source_ip;
    table_lookup(&firewall_ip_table, &fid, &firewall_ip_entries);

    table_lookup(&firewall_tcpport_table, &fid, &firewall_tcpport_entries);

    table_lookup(&priority_table, &fid, &priority_entry1);
    table_lookup(&priority_table, &fid, &priority_entry2);
    struct firewall_meta_header_t meta_header;

    meta_header.allow = firewall_tcpport_entries.if_allow + firewall_ip_entries.if_allow;
    meta_header.acc_priority = priority_entry1.cookies + priority_entry2.cookies;
    meta_header.fid = fid;
    meta_header.reserved = firewall_ip_entries.timeout + firewall_tcpport_entries.timeout;
    BITS(32) firewall_drop_bit = firewall_tcpport_entries.if_allow + firewall_ip_entries.if_allow;
    bufemit(packet, &meta_header);

    // flow tracker logic
    struct err_tracker_t err_tracker_entry;
    table_lookup(&err_tracker_table, &fid, &err_tracker_entry);
    err_tracker_entry.total_drop_cnt = err_tracker_entry.total_drop_cnt + 1 - firewall_drop_bit;
    err_tracker_entry.total_byte_cnt = err_tracker_entry.total_byte_cnt + 256 + tcp_header.seq + tcp_header.sum;
    BITS(32) signature =  err_tracker_entry.total_drop_cnt + err_tracker_entry.total_byte_cnt + tcp_header.sum;
    table_update(&err_tracker_table, &fid, &err_tracker_entry);

    struct tcp_tracker_t tcp_tracker_entry;
    table_lookup(&tcp_tracker_table, &src_ip, &tcp_tracker_entry);
    tcp_tracker_entry.tcp_src_port_cnt = tcp_tracker_entry.tcp_src_port_cnt -  firewall_drop_bit;
    BITS(32) signature2 =  tcp_tracker_entry.tcp_sync_cnt;
    table_update(&tcp_tracker_table, &src_ip, &tcp_tracker_entry);
    tcp_tracker_entry.tcp_sync_cnt = tcp_tracker_entry.tcp_sync_cnt + signature;
    struct connect_tracker_meta_header_t meta_header2;
    meta_header2.tcp_tracker = tcp_tracker_entry.tcp_src_port_cnt;
    meta_header2.ip_tracker = tcp_tracker_entry.tcp_sync_cnt;
    bufemit(packet, &meta_header2);

    // L4 load balancer
    BITS(16) src_port;
    src_port = tcp_header.sport;
    BITS(32) base_ip_src;
    base_ip_src = 134744072;
    BITS(32) base_ip_dst;
    base_ip_dst = 134744071;
    BITS(16) base_port_src;
    base_port_src = 50;
    BITS(16) base_port_dst;
    base_port_dst = 60;
    struct lb_DIP_entries_t lb_DIP_entry;
    table_lookup(&lb_table, &src_port, &lb_DIP_entry);
    lb_DIP_entry.if_alloc = 1;
    lb_DIP_entry.mac_src =  lb_DIP_entry.mac_src + fid;
    lb_DIP_entry.mac_dst =  lb_DIP_entry.mac_dst + fid;
    lb_DIP_entry.ip_src = base_ip_src + fid;
    lb_DIP_entry.ip_dst = base_ip_dst + fid;
    lb_DIP_entry.port_src = base_port_src + fid;
    lb_DIP_entry.port_dst = base_port_dst + fid;
    lb_DIP_entry.hash = base_ip_src + base_ip_dst + base_port_src + base_port_dst + fid;
    table_update(&lb_table, &src_port, &lb_DIP_entry);

    BITS(32) new_fid;
    new_fid = lb_DIP_entry.ip_src + lb_DIP_entry.ip_dst + lb_DIP_entry.port_src + lb_DIP_entry.port_dst;
    struct lb_fwd_tcp_hdr_t new_tcp_hdr;
    new_tcp_hdr.raw1 = new_fid;
    bufemit(packet, &new_tcp_hdr);

    EXT__NET_SEND__net_send(packet);
}


