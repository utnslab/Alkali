
#ifndef _NFPLIB_H_
#define _NFPLIB_H_

#include <nfp.h>
#include <nfp_intrinsic.h>
#include <nfp/cls.h>
// #include <nfp/ctm.h>
#include <nfp/mem_bulk.h>
#include <stdint.h>

#include <nfp/me.h>
#include <nfp6000/nfp_me.h>
#include <pkt/pkt.h>
#include <std/reg_utils.h>
#include <blm.h>
#include "lmem_lookup.h"
#include <lu/cam_hash.h>

#define ALIGNED_U64(x) ((x&0x7)==0)
#define ALIGNED_U32(x) ((x&0x3)==0)
#define ALIGNED_U16(x) ((x&0x1)==0)

struct pkt_raw_t {
  struct nbi_meta_catamaran meta; /*> NBI characterization metadata */
  uint32_t mac_timestamp;         /*> Ingress timestamp added by MAC */
  uint32_t mac_prepend;           /*> Ingress MAC prepend data */
};

#ifndef IF_SIMULATION
    #define IF_SIMULATION 1
#endif

#ifndef IF_GENERATE_FAKE_DMA_RECV_REQ
    #define IF_GENERATE_FAKE_DMA_RECV_REQ 0
#endif

#ifndef NBI
#define NBI 0
#endif

#define MAC_CHAN_PER_PORT   4
#define TMQ_PER_PORT        (MAC_CHAN_PER_PORT * 8)

#define MAC_TO_PORT(x)      (x / MAC_CHAN_PER_PORT)
#define PORT_TO_TMQ(x)      (x * TMQ_PER_PORT)

#define MEM_TYEP_CLS 0
#define MEM_TYEP_CTM 1
#define MEM_TYEP_EMEM 3

#define EMEM_WORKQ_DECLARE(_name, _entries) \
    __export __emem __align(_entries * sizeof(uint8_t)) uint8_t _name[_entries]

#define CTM_WORKQ_DECLARE(_name, _entries) \
    __export __ctm __align(_entries * sizeof(uint8_t)) uint8_t _name[_entries]

#define CLS_WORKQ_DECLARE(_name, _entries) \
    __export __cls __align(_entries * sizeof(uint8_t)) uint8_t _name[_entries]

#define CLS_CONTEXTQ_DECLARE(_struct, _name, _entries) \
    __export __shared __cls struct _struct _name[_entries]

#define EMEM_CONTEXTQ_DECLARE(_struct, _name, _entries) \
    __export __shared __emem struct _struct _name[_entries]

#define CTM_CONTEXQ_DECLARE(_struct, _name, _entries) \
    __export __shared __ctm struct _struct _name[_entries]

#define CLS_CONTEXQ_DECLARE(_struct, _name, _entries) \
    __export __shared __cls struct _struct _name[_entries]


#define SPLIT_LENGTH            3   // Split offset = 2048
#define PKT_NBI_OFFSET          64
#define PKTBUF_CTM_SIZE         SPLIT_LENGTH
#define PKTBUF_MU_OFFSET        (256 << SPLIT_LENGTH)
#define MAC_PREPEND_BYTES       0
#define PKBUF_SHIF_BYTES        0

__export __shared __emem int global_start;

void wait_global_start_(){
    __xrw t;
    /* Wait for start signal */
    for (;;) {
      mem_read32(&t, &global_start, sizeof(int));
      if (t == 1)
        break;
    }
}

__forceinline void init_recv_event_workq(int ring_id, void *ring_ptr, int mem_type, int ring_size, int thread_count)
{
    int i;
    SIGNAL start_sig;

    if (ctx() == 0)
    {
        if (mem_type == MEM_TYEP_CLS)
        {
            cls_workq_setup(ring_id, ring_ptr, ring_size);
        }
        else if (mem_type == MEM_TYEP_CTM)
        {
            ctm_ring_setup(ring_id, ring_ptr, ring_size);
        }
        else if (mem_type == MEM_TYEP_EMEM)
        {
            mem_ring_setup(ring_id, ring_ptr, ring_size);
        }

        for (i = 0; i < thread_count; i++)
        {
            signal_ctx(i, __signal_number(&start_sig));
        }
    }

    __implicit_write(&start_sig);
    __wait_for_all(&start_sig);
}

__forceinline void bulk_memcpy(__mem40 void * dest, __mem40 void * src, int n){

    uint32_t  p= (uint32_t)dest;
    uint32_t  q= (uint32_t)src;
    if (ALIGNED_U64(p) && ALIGNED_U64(q))
    {
        while (n >= 8)
        {
            *(__mem uint64_t *) p = *(__mem uint64_t *) q;
        p += 8;
        q += 8;
            n -= 8;
        }
    }

    if (ALIGNED_U32(p) && ALIGNED_U32(q))

    {
        while (n >= 4)
        {
            *(__mem uint32_t *) p= *(__mem uint32_t *) q;
        p += 4;
        q += 4;
            n -= 4;
        }
    }

    while (n > 0)
    {
        if(n > 8 && ALIGNED_U64(p) && ALIGNED_U64(q)){
            *(__mem uint64_t *) p = *(__mem uint64_t *) q;
            p += 8;
            q += 8;
            n -= 8;
        }
        else if(n > 4 && ALIGNED_U32(p) && ALIGNED_U32(q)){

            *(__mem uint32_t *) p= *(__mem uint32_t *) q;
            p += 4;
            q += 4;
            n -= 4;
        }
        else{
            *(__mem int8_t *) p= *(__mem int8_t *) q;
            p += 1;
            q += 1;
            n -= 1;
        }
    }
}


/**
 * The following rules must be followed to when using the CAM.
 * 1. CAM is not reset by a FPC reset. Software must either do a CAM_clear prior to using
 * the CAM to initialize the LRU and clear the tags to 0, or explicitly write all entries with
 * CAM_write.
 * 2. No two tags can be written to have same value. If this rule is violated, the result of a
 * lookup that matches that value will be unpredictable, and the LRU state is unpredictable.
 *
 * The value 0x00000000 can be used as a valid lookup value. However, note that CAM_clear instruction puts
 * 0x00000000 into all tags. To avoid violating rule 2 after doing CAM_clear, it is necessary to write all
 * entries to unique values prior to doing a lookup of 0x00000000.
 *
 * NOTE: Disallow flow_id=0 to avoid complications!
 */
__intrinsic void init_me_cam(int size)
{
    int i = 0;
    if (ctx() == 0) {
        cam_clear();

        for (i = 0 ; i < size; i++)
        {
            cam_write(i, 0xFFFFFFFF - i, 0);
        }
    }
}


int me_cam_lookup(int key){
    cam_lookup_t cam_lookup_result;
    unsigned int lookup_index;
    cam_lookup_result = cam_lookup(key);
    lookup_index = cam_lookup_result.entry_num;
    if(cam_lookup_result.hit)
        return lookup_index;
    else
        return -1;
}

int lmem_cam_lookup(__lmem struct flowht_entry_t* tb, int key, int size){
    __gpr struct flow_key_t fkey;
    int flow_id;
    fkey.key = key;
    flow_id = lmem_flow_lookup(tb, &fkey, size);
    return flow_id;
}

int lmem_cam_update(__lmem struct flowht_entry_t* tb, int key, int size){
    __gpr struct flow_key_t fkey;
    __gpr struct flowht_entry_t fentry;
    int flow_id;
    fkey.key = key;
    flow_id = lmem_flow_lookup(tb, &fkey, size);
    if(flow_id == -1)
    {
        flow_id = 0;
        fentry.key = fkey;
        fentry.flow_id = 0; // TODO: flow_id should be atomic int
        flow_id = lmem_flow_insert(tb, &fentry, size);
    }
    return flow_id;
}

// If key is already in cam, return its index
// If ket is not in cam, write key into cam, and return its index
int me_cam_update(int key){
    cam_lookup_t cam_lookup_result;
    unsigned int lookup_index;
    cam_lookup_result = cam_lookup(key);
    lookup_index = cam_lookup_result.entry_num;
    if(cam_lookup_result.hit)
        return lookup_index;
    else
    {    
        cam_write(lookup_index, key, 0);
        return lookup_index; // if miss, lookup_index stores the ideal empty slot to store the key
    }
}

__intrinsic uint64_t alloc_packet_buffer()
{
  __gpr struct nbi_meta_pkt_info pkt_info;
  __xwrite struct nbi_meta_pkt_info pkt_info_write;
  unsigned int pnum;
  __xread blm_buf_handle_t buf;
  __mem40 uint8_t* pbuf;
  __cls struct ctm_pkt_credits ctm_credits;

  ctm_credits.pkts = 128;
  ctm_credits.bufs = 64;

  /* Allocate CTM buffer */
  while (1) {
    /* Allocate and Replenish credits */
    pnum = pkt_ctm_alloc(&ctm_credits, __ISLAND, PKTBUF_CTM_SIZE, 1, 1);
    if (pnum != ((unsigned int)-1))
      break;
  }

  /* Allocate MU buffer */
  while (1) {
    if (blm_buf_alloc(&buf, 0) == 0)
      break;
  }
  pbuf = pkt_ctm_ptr40(__ISLAND, pnum, 0);

  pkt_info.isl   = __ISLAND;
  pkt_info.pnum  = pnum;
  pkt_info.bls  = 0;
  pkt_info.muptr = buf;

  pkt_info_write = pkt_info;
  mem_write32(&pkt_info_write, pbuf, sizeof(pkt_info_write));
  
  pbuf += PKT_NBI_OFFSET + MAC_PREPEND_BYTES + PKBUF_SHIF_BYTES;

  return (uint64_t)pbuf;
}

#endif