#pragma once

#include "itty-bit-string.h"
#include <stddef.h>

size_t itty_position_locality_bonus (size_t query_index,
                                     size_t key_index,
                                     size_t window);
size_t itty_position_gray_code (size_t position);
itty_bit_string_t *itty_position_gray_encode (size_t position,
                                              size_t number_of_words);
size_t itty_position_gray_similarity (size_t query_index,
                                      size_t key_index,
                                      size_t number_of_bits);
