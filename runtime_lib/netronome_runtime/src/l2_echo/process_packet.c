
#include "nfplib.h"
#include "context.h"
#include "eventq_config.h"
#include "extern/extern_net.h"
#include "struct.h"

// handler NET_RECV:process_packet (context ctx, buf packet) {
//     eth_header_t eth_header;
//     bits<48> tmp_mac;
//     ip_header_t ip_header;
//     buf packet_out;

//     packet.extract(eth_header);

//     # swap src and dst mac
//     tmp_mac = eth_header.src_mac;
//     eth_header.src_mac = eth_header.dst_mac;
//     eth_header.dst_mac = tmp_mac;

//     packet_out.emit(eth_header);
//     packet_out.emit(packet);

//     generate NET_SEND{ctx, packet_out};
// }

int main(void)
{
    // The type should be __xrw if net_recv_event is get from init_recv_event_workq
    // However, since here we are receiving from inlined function call, it is decalred as __gpr
    struct event_param_NET_RECV net_recv_event;
    struct event_param_NET_SEND net_send_event;
    __xwrite struct event_param_NET_SEND net_send_event_ref;

    struct eth_header_t eth_header;
    __xread struct eth_header_t eth_header_read;
    __xwrite struct eth_header_t eth_header_write;

    __mem40 uint8_t* packet_out_buf;

    struct bits48_t tmp_mac;
    int extract_offset = 0;
    int emit_length = 0;

    // // If this is the first stage of the pipeline chain, initialize the context chain ring.
    // // Unused in this example, since context is unused
    // init_context_chain1_ring();

    // // Initial event queue for this pipeline stage. the queue can be instanted at different memory hierarchy (mem/ctm/cls)
    // // Unused in this example, since NET_RECV is called from inlined extern function call, not from the event work queue.
    // init_recv_event_workq(WORKQ_ID_NET_RECV, workq_NET_RECV, WORKQ_TYPE_NET_RECV, WORKQ_SIZE_NET_RECV, 8);

    for (;;)
    {
        // // Receive a event from event queue, Change based on memory type:
        // // mem_workq_add_thread / ctm_ring_get /  cls_workq_add_thread
        // // Unused in this example, we receive event from inlined extern function call.
        // cls_workq_add_thread(WORKQ_ID_NET_RECV, &net_recv_event, sizeof(net_recv_event));

        inlined_net_recv(&net_recv_event);

        // Read from register to mem buf, the extract is splited into to mem read
        // mem_read32 requires the read size to be 4 bytes aligned
        // mem_read8 requires the read size to be 1 byte aligned
        mem_read32(&eth_header_read, (net_recv_event.packet_ptr + extract_offset), 12);
        extract_offset +=  12;
        mem_read8(&(eth_header_read.ether_type), (net_recv_event.packet_ptr + extract_offset), 2);
        extract_offset +=  2;

        // copy from xfer read register to register
        eth_header = eth_header_read;

        tmp_mac = eth_header.src_mac;
        eth_header.src_mac = eth_header.dst_mac;
        eth_header.dst_mac = tmp_mac;

        packet_out_buf = alloc_packet_buffer();

        // first copy to xfer write register
        eth_header_write = eth_header;

        // Then copy from xfer write register to mem buf
        mem_write32(&eth_header_write, packet_out_buf, 12);
        emit_length += 12;
        mem_write8(&(eth_header_write.ether_type), packet_out_buf, 2);
        emit_length += 2;

        // Copy from mem to mem
        bulk_memcpy(packet_out_buf, (net_recv_event.packet_ptr + extract_offset), net_recv_event.meta.len - extract_offset);
        emit_length += (net_recv_event.meta.len - extract_offset);
        // ua_memcpy((__mem40 uint8_t *)(net_recv_event.packet_ptr + extract_offset), 0, packet_out_buf, 0, net_recv_event.meta.len - extract_offset);

        // Generate next event
        net_send_event.meta.len = emit_length;
        net_send_event.packet_ptr = packet_out_buf;
        inlined_net_send(&net_send_event);

        
    }

    return 0;
}