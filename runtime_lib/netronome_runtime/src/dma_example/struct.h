#ifndef _STRUCT_H_
#define _STRUCT_H_


// 32B received descriptor
__packed struct recv_desc_t
{
    uint32_t flow_id;
    uint32_t bump_seq;
    uint32_t flags;
    uint32_t flow_grp;
};


#endif
