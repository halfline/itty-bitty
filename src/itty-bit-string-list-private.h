#pragma once
#include <stddef.h>
#include <stdbool.h>

#include "itty-bit-string.h"

struct itty_bit_string_list_t {
        itty_bit_string_t **bit_strings;
        size_t              count;
        size_t              max_number_of_words;
};
