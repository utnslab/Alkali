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
  int a,b,c,d,e,f;
};

extern void send_packet(buf_t packet);
extern void bufextract(buf_t packet, void *extracted_data);
extern void bufemit(buf_t packet, void *extracted_data);

void _packet_event_handler(buf_t packet) {
  struct pkt_info_t x,y,z;
  struct pkt_info_t s;

  bufextract(packet, (void *)&x);
  bufextract(packet, (void *)&y);
  bufextract(packet, (void *)&z);

  s.a = x.a + y.a + z.a;
  s.b = x.b + y.b + z.b;
  s.c = x.c + y.c + z.c;
  s.d = x.d + y.d + z.d;

  bufemit(packet, (void *)&s);
  send_packet(packet);
}

int main(){
  return 0;
}