#pragma once

#include "itty-bit-string.h"
#include "itty-manager.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct itty_exec_buffer_t itty_exec_buffer_t;

typedef size_t itty_exec_buffer_id_t;

#define ITTY_EXEC_BUFFER_INVALID_ID ((itty_exec_buffer_id_t) -1)

typedef enum {
        ITTY_EXEC_BUFFER_STORAGE_EXTERNAL,
        ITTY_EXEC_BUFFER_STORAGE_OWNED
} itty_exec_buffer_storage_t;

typedef enum {
        ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
        ITTY_EXEC_BUFFER_ACCESS_READ_WRITE
} itty_exec_buffer_access_t;

typedef struct {
        itty_exec_buffer_id_t buffer_id;
        size_t                word_offset;
        size_t                number_of_words;
        size_t                bit_length;
} itty_exec_buffer_slice_t;
typedef itty_exec_buffer_slice_t itty_exec_buffer_value_t;
typedef itty_exec_buffer_slice_t itty_exec_buffer_array_t;
typedef itty_exec_buffer_slice_t itty_exec_buffer_bits_t;

itty_exec_buffer_t *itty_exec_buffer_new (void);
void itty_exec_buffer_free (itty_exec_buffer_t *exec_buffer);

itty_exec_buffer_id_t itty_exec_buffer_register_words (itty_exec_buffer_t          *exec_buffer,
                                                       size_t                      *words,
                                                       size_t                       number_of_words,
                                                       itty_exec_buffer_access_t    access,
                                                       const char                  *debug_name);
itty_exec_buffer_id_t itty_exec_buffer_register_bit_string (itty_exec_buffer_t          *exec_buffer,
                                                            itty_bit_string_t           *bit_string,
                                                            itty_exec_buffer_access_t    access,
                                                            const char                  *debug_name);
bool itty_exec_buffer_rebind_words (itty_exec_buffer_t       *exec_buffer,
                                    itty_exec_buffer_id_t     buffer_id,
                                    size_t                   *words,
                                    size_t                    number_of_words);
bool itty_exec_buffer_rebind_bit_string (itty_exec_buffer_t       *exec_buffer,
                                         itty_exec_buffer_id_t     buffer_id,
                                         itty_bit_string_t        *bit_string);
itty_exec_buffer_id_t itty_exec_buffer_allocate_words (itty_exec_buffer_t *exec_buffer,
                                                       size_t              number_of_words,
                                                       const char         *debug_name);
bool itty_exec_buffer_find_descriptor (itty_exec_buffer_t    *exec_buffer,
                                       const char            *debug_name,
                                       itty_exec_buffer_id_t *buffer_id);

itty_exec_buffer_slice_t itty_exec_buffer_get_word_slice (itty_exec_buffer_id_t buffer_id,
                                                          size_t                word_offset,
                                                          size_t                number_of_words);
itty_exec_buffer_slice_t itty_exec_buffer_get_bit_slice (itty_exec_buffer_id_t buffer_id,
                                                         size_t                word_offset,
                                                         size_t                number_of_words,
                                                         size_t                bit_length);
itty_exec_buffer_value_t itty_exec_buffer_get_value (itty_exec_buffer_id_t buffer_id,
                                                     size_t                word_offset);
itty_exec_buffer_array_t itty_exec_buffer_get_array (itty_exec_buffer_id_t buffer_id,
                                                     size_t                word_offset,
                                                     size_t                number_of_words);
itty_exec_buffer_bits_t itty_exec_buffer_get_bits (itty_exec_buffer_id_t buffer_id,
                                                   size_t                word_offset,
                                                   size_t                number_of_words,
                                                   size_t                bit_length);

void itty_exec_buffer_begin_stage (itty_exec_buffer_t *exec_buffer);
void itty_exec_buffer_begin_named_stage (itty_exec_buffer_t *exec_buffer,
                                         const char         *debug_name);
bool itty_exec_buffer_add_xor (itty_exec_buffer_t       *exec_buffer,
                               itty_exec_buffer_slice_t  destination,
                               itty_exec_buffer_slice_t  a,
                               itty_exec_buffer_slice_t  b);
bool itty_exec_buffer_add_xnor (itty_exec_buffer_t       *exec_buffer,
                                itty_exec_buffer_slice_t  destination,
                                itty_exec_buffer_slice_t  a,
                                itty_exec_buffer_slice_t  b);
bool itty_exec_buffer_add_popcount (itty_exec_buffer_t       *exec_buffer,
                                    itty_exec_buffer_slice_t  destination,
                                    itty_exec_buffer_slice_t  source);
bool itty_exec_buffer_add_condense (itty_exec_buffer_t             *exec_buffer,
                                    itty_exec_buffer_slice_t        destination,
                                    itty_exec_buffer_slice_t const *inputs,
                                    size_t                          input_count);
bool itty_exec_buffer_add_weighted_condense (itty_exec_buffer_t             *exec_buffer,
                                             itty_exec_buffer_slice_t        destination,
                                             itty_exec_buffer_slice_t const *inputs,
                                             itty_exec_buffer_slice_t        votes,
                                             size_t                          input_count);
bool itty_exec_buffer_add_double (itty_exec_buffer_t       *exec_buffer,
                                  itty_exec_buffer_slice_t  destination,
                                  itty_exec_buffer_slice_t  source);
bool itty_exec_buffer_add_clear_array_range (itty_exec_buffer_t        *exec_buffer,
                                             itty_exec_buffer_array_t   destination,
                                             itty_exec_buffer_value_t   start,
                                             itty_exec_buffer_value_t   count);

bool itty_exec_buffer_run (itty_exec_buffer_t *exec_buffer);
bool itty_exec_buffer_run_with_manager (itty_exec_buffer_t *exec_buffer,
                                        itty_manager_t     *manager);
size_t itty_exec_buffer_get_stage_count (itty_exec_buffer_t *exec_buffer);
bool itty_exec_buffer_find_stage (itty_exec_buffer_t *exec_buffer,
                                  const char         *debug_name,
                                  size_t             *stage_index);
bool itty_exec_buffer_run_stage (itty_exec_buffer_t *exec_buffer,
                                 size_t              stage_index);
bool itty_exec_buffer_run_stage_with_manager (itty_exec_buffer_t *exec_buffer,
                                              size_t              stage_index,
                                              itty_manager_t     *manager);
bool itty_exec_buffer_run_named_stage (itty_exec_buffer_t *exec_buffer,
                                       const char         *debug_name);
bool itty_exec_buffer_run_named_stage_with_manager (itty_exec_buffer_t *exec_buffer,
                                                    const char         *debug_name,
                                                    itty_manager_t     *manager);
char *itty_exec_buffer_present (itty_exec_buffer_t *exec_buffer);
