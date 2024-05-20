#pragma once

#include "itty-bit-string-list.h"
#include <stddef.h>
#include <stdbool.h>

typedef struct itty_bit_string_map_file_t itty_bit_string_map_file_t;

itty_bit_string_map_file_t *itty_bit_string_map_file_new (const char *file_name);

void itty_bit_string_map_file_free (itty_bit_string_map_file_t *mapped_file);

itty_bit_string_t *itty_bit_string_map_file_next (itty_bit_string_map_file_t *mapped_file,
                                                  size_t                      number_of_words);
char *itty_bit_string_map_file_get_mapped_data (itty_bit_string_map_file_t *mapped_file);

bool itty_bit_string_map_file_resize (itty_bit_string_map_file_t *mapped_file,
                                      size_t                      new_size);

