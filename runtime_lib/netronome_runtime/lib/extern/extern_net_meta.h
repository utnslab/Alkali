
#ifndef _EXTERN_NET_META_H_
#define _EXTERN_NET_META_H_

struct recv_meta_t {
    union {
        struct {
            unsigned int seq:16;  /**< Packet number of the packet */
            unsigned int len:16;   /**< Length of the packet in bytes
                                        (includes possible MAC prepend bytes) */
        };
    };
};

struct send_meta_t {
    union {
        struct {
            unsigned int seq:16;  /**< Packet number of the packet */
            unsigned int len:16;   /**< Length of the packet in bytes
                                        (includes possible MAC prepend bytes) */
        };
    };
};

#endif

