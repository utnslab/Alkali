#include "nfplib.h"
#include "prog_hdr.h"


// "__shared __lmem" means the table is locallly defined within one ME but shared between this ME's 8 threads. This maps to the local table defined within a handler.
__shared __lmem struct example_struct lookuptable[16];

// In the future if we have global table that shared between handler (across ME), need to use this following declaration
//__export __cls struct example_struct table[16];

int main(void) {
  struct example_struct tmp_struct;
  int tmp_index;

  // A table consist of 1. A cam that maps key to index. and 2. A arrary (which is lookuptable[16]) that maps index to value
  // Call init_me_cam(size); to init a cam in a ME, this cam is shared between 8 threads within a ME, but not shared between me.
  // TODO: In the future if we want to have global table, that shared between handler, need another way to init cam that can shared between ME. Need to add that library function.
  init_me_cam(16);


  // Table Update:
  tmp_struct.a1 = 14;
  tmp_struct.a2 = 15;
  // update the key in the camm, returns the index of the key
  tmp_index = me_cam_update(4);
  // store the value based on index
  lookuptable[tmp_index] = tmp_struct;
  
  // Table Update:
  tmp_struct.a1 = 6;
  tmp_struct.a2 = 7;
  // update the key in the camm, returns the index of the key
  tmp_index = me_cam_update(5);
  // store the value based on index
  lookuptable[tmp_index] = tmp_struct;
	

  // Table Lookup
  tmp_index = me_cam_lookup(5);
  tmp_struct = lookuptable[tmp_index];

  // For debug
  // local_csr_write(local_csr_mailbox_1, tmp_index);
  // local_csr_write(local_csr_mailbox_2, tmp_struct.a1);
  // local_csr_write(local_csr_mailbox_3, tmp_struct.a2 );
}
