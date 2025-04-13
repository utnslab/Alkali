#ifndef ALKALI_H
#define ALKALI_H

struct buf_tag {
    int buf_tag;
};
typedef struct buf_tag * buf_t;

/* optional types */
#define BITS_FIELD(N,name) \
    int name

/* BITS for defining non-field data */
#define BITS(N) BITS ## _ ## N

#define BITS_32 int
#define BITS_16 short

/* options used by ak_TABLE */
#define anno_MAX_REPLICA(num) float anno_rep[num];

/* Use type level information to tag a special structure - doubles wont be using anywhere */
#define ak_TABLE(size,kt,vt,...) \
    struct { \
        kt key; \
        vt value; \
        char cap[size]; \
        __VA_ARGS__ \
    }

/* function prototypes */
extern void send_packet(buf_t packet);
extern buf_t bufinit();
extern void bufextract(buf_t packet, void *extracted_data);
extern void bufemit(buf_t packet, void *extracted_data);

extern void table_lookup(void *tab, void *key, void *value);
extern void table_update(void *tab, void *key, void *value);

extern void generate(const char * target, buf_t packet);
        
/* dummy main function to make the compiler happy */
int main() { return 0; }

#endif // ALKALI_H