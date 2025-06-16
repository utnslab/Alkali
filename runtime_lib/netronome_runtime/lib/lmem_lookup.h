/* SPDX-License-Identifier: BSD 3-Clause License */
/* Copyright (c) 2022, University of Washington, Max Planck Institute for Software Systems, and The University of Texas at Austin */

#ifndef LMEM_FLOW_LOOKUP_H_
#define LMEM_FLOW_LOOKUP_H_

#include <stdint.h>
#include <nfp.h>
#include <lu/cam_hash.h>
#include <lu/cam_cls_hash.h>
#include <lu/cam_lmem_hash.h>
#include <nfp/mem_bulk.h>
#include <nfp/mem_atomic.h>
#include <std/reg_utils.h>
#include "fp_mem.h"
#include "nfplib.h"

#define INVALID_FLOWID  ((uint32_t) -1)

struct flow_key_t {
  uint32_t key;
};

struct flowht_entry_t {
  struct flow_key_t   key;
  uint32_t          flow_id;
};

__intrinsic int lmem_flow_insert(__lmem struct flowht_entry_t* tb, __gpr struct flowht_entry_t* entry, int size)
{
  __gpr uint64_t b_idx;
  b_idx = lmem_flow_hash(&(entry->key));
  b_idx = CAMHT_BUCKET_IDX(b_idx, size);
  tb[b_idx] = *entry;
}


__intrinsic uint64_t lmem_flow_hash(__gpr struct flow_key_t* key)
{
  struct camht_hash_t hash;
  camht_hash((void*) key, sizeof(struct flow_key_t), &hash);

  return hash.__raw;
}

__intrinsic int lmem_flow_lookup(__lmem struct flowht_entry_t* tb, __gpr struct flow_key_t* key, int size)
{
  __gpr struct flowht_entry_t entry;
  __gpr uint64_t b_idx;
  __gpr int result;
  b_idx = lmem_flow_hash(key);
  b_idx = CAMHT_BUCKET_IDX(b_idx, size);
  result = tb[b_idx].flow_id;
  if (!reg_eq(&entry, (void*) key, sizeof(struct flow_key_t)))
    goto out;

  return result;

out:
  return -1;
}

#endif /* LMEM_FLOW_LOOKUP_H_ */