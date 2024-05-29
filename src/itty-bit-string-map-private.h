#pragma once

#include "itty-bit-string-list.h"
#include <stddef.h>
#include <stdbool.h>

struct itty_bit_string_map_file_t {
        int         fd;
        size_t      file_size;
        void       *mapped_data;
        size_t      word_count_per_bit_string;
        size_t      current_index;
        itty_bit_string_list_t *bit_string_list;
};
