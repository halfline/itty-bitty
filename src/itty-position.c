#include "itty-position.h"
#include "itty-bit-string-private.h"

size_t
itty_position_locality_bonus (size_t query_index,
                              size_t key_index,
                              size_t window)
{
        size_t distance = query_index > key_index ?
                query_index - key_index :
                key_index - query_index;

        if (distance >= window)
                return 0;

        return window - distance;
}

size_t
itty_position_gray_code (size_t position)
{
        return position ^ (position >> 1);
}

itty_bit_string_t *
itty_position_gray_encode (size_t position,
                           size_t number_of_words)
{
        if (number_of_words == 0)
                return NULL;

        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        if (!bit_string)
                return NULL;

        itty_bit_string_append_word (bit_string, itty_position_gray_code (position));
        for (size_t i = 1; i < number_of_words; i++)
                itty_bit_string_append_word (bit_string, 0);

        return bit_string;
}

size_t
itty_position_gray_similarity (size_t query_index,
                               size_t key_index,
                               size_t number_of_bits)
{
        size_t query_gray = itty_position_gray_code (query_index);
        size_t key_gray = itty_position_gray_code (key_index);
        size_t difference = query_gray ^ key_gray;
        size_t similarity = 0;

        size_t max_bits = ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        if (number_of_bits > max_bits)
                number_of_bits = max_bits;

        for (size_t i = 0; i < number_of_bits; i++) {
                if ((difference & (1UL << i)) == 0)
                        similarity++;
        }

        return similarity;
}
