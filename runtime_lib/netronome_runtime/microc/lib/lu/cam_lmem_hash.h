/*
 * Copyright 2012-2018 Netronome, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file          lib/lu/cam_hash.h
 * @brief         Interface for a CAM assisted hash table
 */

#ifndef _LU__CAM_HASH_lmem_H_
#define _LU__CAM_HASH_lmem_H_

#include <nfp.h>
#include <stdint.h>
#include <types.h>
#include <nfp/cls.h>
#include <lu/cam_hash.h>

/**
 * This module provides an implementation for a CAM assisted hash
 * table.  It provides a very efficient lookup mechanism suitable for
 * large data structures such as a flow tables.  A single memory
 * operation can determine either a potential match or no match.  A
 * subsequent memory access is required to determine if a potential
 * match is a real match.
 *
 * The hash table is fixed in size with @_nb_entries and consists of
 * two tables/arrays: One table containing hash values (organised in
 * fixed buckets of size @CAMHT_BUCKET_ENTRIES) and a table containing
 * the keys.  Two hash functions are used for a key lookup: The first
 * hash provides a (suitably masked) index for a bucket and the second
 * hash is used in a CAM lookup within the bucket.  This structure
 * supports up to @CAMHT_BUCKET_ENTRIES collisions on the first hash
 * but does not allow for collisions on the second hash with in the
 * bucket (though support for this could be added).
 *
 * The combination of bucket index and the index of the matching entry
 * in the bucket (returned by the CAM lookup) provides the index into
 * the key table (and possibly other associated tables).
 *
 * The key table is organised in entries of type @_entry_type.  The
 * entry type must be at least the size of the key but may be larger,
 * for example, to align the keys to a cache line or to store
 * additional information along with the key.
 */


/* The current implementation only represents the minimum set of
 * features which are currently needed.  It can easily be extended in
 * a number of ways:
 *
 * - Add more API functions, e.g. for adding entries or for providing
 *   asynchronous access functions.
 *
 * - Provide more configuration options.  One can use a single 32bit
 *   compile time constant for flags indicating things like: 24/32bit
 *   CAM lookup, size of the bucket/CAM lookup, type of hash functions
 *   to use etc. Being a compile time constant the C compiler will be
 *   able to generate efficient code.  The config word should also be
 *   stored in EMEM for the host to be able to determine the
 *   configuration of a particular hash table.
 *
 * - Add support for accessing the top 8bits when 24bit CAM lookups
 *   are used. These can be used as flags associated with an entry and
 *   may provide some optimisation opportunities.
 *
 * - Add support for CLS based hash tables.
 */

/**
 * Lookup a key in the hash table.
 * @param hash_tbl     Address of the hash table
 * @param key_tbl      Address of the key table
 * @param entries      Total number of entries in the hash table
 * @param key          Pointer to the key to lookup
 * @param key_sz       Size of the key
 * @param entry_sz     Size of an entry in the key table
 * @return             Hash table entry index or -1 if not found.
 *
 * This function performs the full hash table lookup synchronously.
 * It will compute the hash values, perform a lookup in the hash table
 * and compare any matching keys.
 */
__intrinsic int32_t camht_lmem_lookup(__lmem void *hash_tbl, __lmem void *key_tbl,
                                 int32_t entries, size_t entry_sz,
                                 void *key, size_t key_sz);

/**
 * Lookup a key in the hash table.
 * @param hash_tbl     Address of the hash table
 * @param entries      Total number of entries in the hash table
 * @param hash         Pointer to hash computed from the key
 * @return             A index if a match was found. -1 on error.
 *
 * This function performs the partial hash table lookup.  It computes
 * hash values and perform a lookup in the hash table.  It will *not*
 * compare the key.  Use this function if a hash table entry contains
 * additional information apart from the hash key itself.
 */
__intrinsic int32_t camht_lmem_lookup_idx(__lmem void *hash_tbl, int32_t entries,
                                    struct camht_hash_t* hash);


/**
 * Lookup and add a key in the hash table.
 * @param hash_tbl     Address of the hash table
 * @param entries      Total number of entries in the hash table
 * @param hash         Pointer to hash computed from the key
 * @param added        A return flag that indicates that an add (1) was done
 *                     Only valid if the return value is not -1
 * @return             The index of the found or added hash value.
 *                     -1 if the entry was not found and the CAM is full.
 *
 * This function performs a cam lookup add operation.  It computes
 * hash values and performs a lookup add in the hash table.  It will *not*
 * compare the key in case of a hit.
 * If the entry was found (hit) the return value is the index of the found
 * entry and the @add flag is set to 0.
 * If an add was performed the return value is the index of the CAM entry
 * where the hash value was inserted to and the @add flag will be set to 1.
 * If the entry was not found but the add has failed (CAM was full) -1 will
 * be returned.
 */
__intrinsic int32_t camht_lmem_lookup_idx_add(__lmem void *hash_tbl,
                                         int32_t entries,
                                         struct camht_hash_t* hash,
                                         int32_t* added);

/**
 * Lookup and add a key in the hash table.
 * @param hash_tbl     Address of the hash table
 * @param entries      Total number of entries in the hash table
 * @param key          Pointer to the key to lookup
 * @param key_sz       Size of the key
 * @param entry_sz     Size of the entry
 * @param added        A return flag that indicates that an add (1) was done
 *                     Only valid if the return value is not -1
 * @return             The index of the found or added hash value.
 *                     -1 if the entry was not found and the CAM is full.
 *
 * This function performs a cam lookup add operation.  It computes
 * hash values and performs a lookup add in the hash table.  It will *not*
 * compare the key in case of a hit.
 * If the entry was found (hit) the return value is the index of the found
 * entry and the @add flag is set to 0.
 * If an add was performed the return value is the index of the CAM entry
 * where the hash value was inserted to and the @add flag will be set to 1.
 * If the entry was not found but the add has failed (CAM was full) -1 will
 * be returned. If @add is 1, it will also copy the entry to key table.
 */
__intrinsic int32_t camht_lmem_lookup_add(__lmem void *hash_tbl,
                                    __lmem void *key_tbl,
                                    int32_t entries,
                                    void *key, size_t key_sz,
                                    size_t entry_sz,
                                    int32_t* added);
#endif /* _LU__CAM_HASH_lmem_H_ */

/* -*-  Mode:C; c-basic-offset:4; tab-width:4 -*- */
