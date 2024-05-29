#pragma once

#include <stddef.h>
#include <stdbool.h>

typedef enum itty_bit_string_mutability_t itty_bit_string_mutability_t;

struct itty_bit_string_t {
        size_t *words;
        size_t number_of_words;
        size_t pop_count;
        size_t bit_length;
        itty_bit_string_mutability_t mutability;
        unsigned long pop_count_computed : 1;
        unsigned long bit_length_computed : 1;
};

static inline void
itty_bit_string_set_bit (itty_bit_string_t *bit_string,
                         size_t             bit_index,
                         bool               value)
{
        size_t word_index = bit_index / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t bit_position = bit_index % ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        if (value) {
                bit_string->words[word_index] |= (1UL << bit_position);
        } else {
                bit_string->words[word_index] &= ~(1UL << bit_position);
        }
}
