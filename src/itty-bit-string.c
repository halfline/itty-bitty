#include "itty-bit-string.h"
#include "itty-bit-string-private.h"
#include "itty-bit-string-list.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

itty_bit_string_t *
itty_bit_string_new (void)
{
        itty_bit_string_t *bit_string = malloc (sizeof (itty_bit_string_t));
        bit_string->words = NULL;
        bit_string->number_of_words = 0;
        bit_string->pop_count = 0;
        bit_string->pop_count_computed = false;
        bit_string->bit_length = 0;
        bit_string->bit_length_computed = false;
        return bit_string;
}

itty_bit_string_t *
itty_bit_string_duplicate (itty_bit_string_t *input_bit_string)
{
        itty_bit_string_t *bit_string = itty_bit_string_new ();
        *bit_string = *input_bit_string;
        return bit_string;
}

void
itty_bit_string_free (itty_bit_string_t *bit_string)
{
        if (!bit_string) {
                return;
        }
        free (bit_string);
}

void
itty_bit_string_append_word (itty_bit_string_t *bit_string,
                             size_t             word)
{
        bit_string->words = realloc (bit_string->words,
                                    (bit_string->number_of_words + 1) * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
        bit_string->words[bit_string->number_of_words] = word;
        bit_string->number_of_words++;
        bit_string->pop_count_computed = false;
        bit_string->bit_length_computed = false;
}

void
itty_bit_string_append_zeros (itty_bit_string_t *bit_string,
                              size_t             count)
{
        for (size_t i = 0; i < count; i++) {
                itty_bit_string_append_word (bit_string, 0);
        }
}

itty_bit_string_t *
itty_bit_string_exclusive_nor (itty_bit_string_t *a,
                               itty_bit_string_t *b)
{
        size_t max_number_of_words = a->number_of_words > b->number_of_words ? a->number_of_words : b->number_of_words;
        size_t a_padding = max_number_of_words - a->number_of_words;
        size_t b_padding = max_number_of_words - b->number_of_words;

        if (a_padding > 0) {
                itty_bit_string_append_zeros (a, a_padding);
        }

        if (b_padding > 0) {
                itty_bit_string_append_zeros (b, b_padding);
        }

        itty_bit_string_t *result = itty_bit_string_new ();
        result->number_of_words = max_number_of_words;
        result->words = malloc (result->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);

        for (size_t i = 0; i < max_number_of_words; i++) {
                result->words[i] = ~(a->words[i] ^ b->words[i]);
        }

        return result;
}

itty_bit_string_t *
itty_bit_string_exclusive_or (itty_bit_string_t *a,
                              itty_bit_string_t *b)
{
        size_t max_number_of_words = a->number_of_words > b->number_of_words ? a->number_of_words : b->number_of_words;
        size_t a_padding = max_number_of_words - a->number_of_words;
        size_t b_padding = max_number_of_words - b->number_of_words;

        if (a_padding > 0) {
                itty_bit_string_append_zeros (a, a_padding);
        }

        if (b_padding > 0) {
                itty_bit_string_append_zeros (b, b_padding);
        }

        itty_bit_string_t *result = itty_bit_string_new ();
        result->number_of_words = max_number_of_words;
        result->words = malloc (result->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);

        for (size_t i = 0; i < max_number_of_words; i++) {
                result->words[i] = a->words[i] ^ b->words[i];
        }

        return result;
}

itty_bit_string_t *
itty_bit_string_combine (itty_bit_string_t *a,
                         itty_bit_string_t *b)
{
        size_t max_number_of_words = a->number_of_words > b->number_of_words ? a->number_of_words : b->number_of_words;
        size_t a_padding = max_number_of_words - a->number_of_words;
        size_t b_padding = max_number_of_words - b->number_of_words;

        if (a_padding > 0) {
                itty_bit_string_append_zeros (a, a_padding);
        }

        if (b_padding > 0) {
                itty_bit_string_append_zeros (b, b_padding);
        }

        itty_bit_string_t *result = itty_bit_string_new ();
        result->number_of_words = max_number_of_words;
        result->words = malloc (result->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);

        for (size_t i = 0; i < max_number_of_words; i++) {
                result->words[i] = a->words[i] | b->words[i];
        }

        return result;
}

itty_bit_string_t *
itty_bit_string_mask (itty_bit_string_t *a,
                      itty_bit_string_t *b)
{
        size_t max_number_of_words = a->number_of_words > b->number_of_words ? a->number_of_words : b->number_of_words;
        size_t a_padding = max_number_of_words - a->number_of_words;
        size_t b_padding = max_number_of_words - b->number_of_words;

        if (a_padding > 0) {
                itty_bit_string_append_zeros (a, a_padding);
        }

        if (b_padding > 0) {
                itty_bit_string_append_zeros (b, b_padding);
        }

        itty_bit_string_t *result = itty_bit_string_new ();
        result->number_of_words = max_number_of_words;
        result->words = malloc (result->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);

        for (size_t i = 0; i < max_number_of_words; i++) {
                result->words[i] = a->words[i] & b->words[i];
        }

        return result;
}

size_t
itty_bit_string_get_pop_count (itty_bit_string_t *bit_string)
{
        if (!bit_string->pop_count_computed) {
                bit_string->pop_count = 0;
                for (size_t i = 0; i < bit_string->number_of_words; i++) {
                        bit_string->pop_count += __builtin_popcountl (bit_string->words[i]);
                }
                bit_string->pop_count_computed = true;
        }
        return bit_string->pop_count;
}

size_t
itty_bit_string_get_length (itty_bit_string_t *bit_string)
{
        if (!bit_string->bit_length_computed) {
                bit_string->bit_length = 0;
                for (size_t i = 0; i < bit_string->number_of_words; i++) {
                        if (bit_string->words[i] == 0)
                                continue;

                        size_t leading_zeros = __builtin_clzll (bit_string->words[i]);
                        size_t word_bit_length = ITTY_BIT_STRING_WORD_SIZE_IN_BITS - leading_zeros;
                        bit_string->bit_length = ((bit_string->number_of_words - i - 1) * ITTY_BIT_STRING_WORD_SIZE_IN_BITS) + word_bit_length;
                        break;
                }
                bit_string->bit_length_computed = true;
        }
        return bit_string->bit_length;
}

size_t
itty_bit_string_evaluate_similarity (itty_bit_string_t *a,
                                     itty_bit_string_t *b)
{
        itty_bit_string_t *exclusive_nor_result = itty_bit_string_exclusive_nor (a, b);
        size_t similarity = itty_bit_string_get_pop_count (exclusive_nor_result);
        itty_bit_string_free (exclusive_nor_result);
        return similarity;
}

int
itty_bit_string_compare (itty_bit_string_t *a,
                         itty_bit_string_t *b)
{
        if (a == b)
                return 0;
        if (itty_bit_string_get_number_of_words (a) < itty_bit_string_get_number_of_words (b))
                return -1;
        if (itty_bit_string_get_number_of_words (a) > itty_bit_string_get_number_of_words (b))
                return 1;
        return memcmp (a->words, b->words, a->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
}

int
itty_bit_string_compare_by_pop_count (itty_bit_string_t *a,
                                      itty_bit_string_t *b)
{
        return (int) (itty_bit_string_get_pop_count (a) - itty_bit_string_get_pop_count (b));
}

void *
itty_bit_string_get_words (itty_bit_string_t *bit_string)
{
        return bit_string->words;
}

size_t
itty_bit_string_get_number_of_words (itty_bit_string_t *bit_string)
{
        return bit_string->number_of_words;
}

void
itty_bit_string_iterator_init_at_word_offset (itty_bit_string_t          *bit_string,
                                              itty_bit_string_iterator_t *iterator,
                                              size_t                      word_offset)
{
        if (word_offset < bit_string->number_of_words) {
                iterator->bit_string = bit_string;
                iterator->current_index = word_offset;
        } else {
                iterator->bit_string = bit_string;
                iterator->current_index = 0;
        }
}

void
itty_bit_string_iterator_init (itty_bit_string_t          *bit_string,
                               itty_bit_string_iterator_t *iterator)
{
        iterator->bit_string = bit_string;
        iterator->current_index = 0;
}

bool
itty_bit_string_iterator_next (itty_bit_string_iterator_t *iterator,
                               size_t                     *word)
{
        if (iterator->current_index < iterator->bit_string->number_of_words) {
                *word = iterator->bit_string->words[iterator->current_index];
                iterator->current_index++;
                return true;
        } else {
                *word = 0;
                return false;
        }
}

itty_bit_string_list_t *
itty_bit_string_split (itty_bit_string_t *bit_string,
                       size_t             number_of_bit_strings)
{
        if (number_of_bit_strings == 0) {
                return NULL;
        }

        size_t total_bits = bit_string->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t bits_per_split = (total_bits + number_of_bit_strings - 1) / number_of_bit_strings;
        size_t words_per_split = (bits_per_split + ITTY_BIT_STRING_WORD_SIZE_IN_BITS - 1) / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        itty_bit_string_list_t *split_list = itty_bit_string_list_new ();
        for (size_t i = 0; i < number_of_bit_strings; i++) {
                itty_bit_string_t *split = itty_bit_string_new ();
                for (size_t j = 0; j < words_per_split; j++) {
                        size_t word_index = i * words_per_split + j;
                        size_t word = (word_index < bit_string->number_of_words) ? bit_string->words[word_index] : 0;
                        itty_bit_string_append_word (split, word);
                }
                itty_bit_string_list_append (split_list, split);
        }

        return split_list;
}

itty_bit_string_t *
itty_bit_string_concatenate (itty_bit_string_t *a,
                             itty_bit_string_t *b)
{
        size_t total_words = a->number_of_words + b->number_of_words;
        itty_bit_string_t *result = itty_bit_string_new ();
        result->words = malloc (total_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
        if (!result->words) {
                free (result);
                return NULL;
        }

        memcpy (result->words, a->words, a->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
        memcpy (result->words + a->number_of_words, b->words, b->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);

        result->number_of_words = total_words;
        result->pop_count_computed = false;

        return result;
}

itty_bit_string_t *
itty_bit_string_double (itty_bit_string_t *bit_string)
{
        return itty_bit_string_concatenate (bit_string, bit_string);
}

itty_bit_string_t *
itty_bit_string_reduce_by_half (itty_bit_string_t *bit_string)
{
        assert (bit_string->number_of_words % 2 == 0);

        size_t half_number_of_words = bit_string->number_of_words / 2;

        itty_bit_string_t *first_half = itty_bit_string_new ();
        first_half->words = bit_string->words;
        first_half->number_of_words = half_number_of_words;
        first_half->pop_count_computed = false;

        itty_bit_string_t *second_half = itty_bit_string_new ();
        second_half->words = bit_string->words + half_number_of_words;
        second_half->number_of_words = half_number_of_words;
        second_half->pop_count_computed = false;

        itty_bit_string_t *reduced_bit_string = itty_bit_string_mask (first_half, second_half);

        first_half->words = NULL;
        second_half->words = NULL;

        itty_bit_string_free (first_half);
        itty_bit_string_free (second_half);

        return reduced_bit_string;
}

char *
itty_bit_string_present (itty_bit_string_t                     *bit_string,
                         itty_bit_string_presentation_format_t  format)
{
        size_t total_bits = bit_string->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t buffer_size = 0;

        switch (format) {
                case ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY:
                case ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY:
                        buffer_size = total_bits;

                        if (format == ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY)
                                buffer_size += strlen ("0b");
                        break;
                case ITTY_BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL:
                case ITTY_BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL_FOR_DISPLAY:
                        buffer_size = bit_string->number_of_words * (ITTY_BIT_STRING_WORD_SIZE_IN_BYTES * 2 + 1);

                        if (format == ITTY_BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL_FOR_DISPLAY)
                                buffer_size += strlen ("0x");
                        break;
                default:
                        return NULL;
        }
        buffer_size++;

        char *bit_string_representation = malloc (buffer_size);

        size_t output_index = 0;
        bool at_leading_zero = true;
        switch (format) {
        case ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY:
                for (size_t i = 0; i < bit_string->number_of_words; i++) {
                        size_t word = bit_string->words[i];
                        for (size_t j = ITTY_BIT_STRING_WORD_SIZE_IN_BITS; j > 0; j--) {
                                bit_string_representation[output_index++] = (word & (1UL << (j - 1))) ? '1' : '0';
                        }
                }
                bit_string_representation[output_index] = '\0';
                break;

        case ITTY_BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL:
                bit_string_representation[output_index] = '\0';
                for (size_t i = 0; i < bit_string->number_of_words; i++) {
                        output_index += snprintf (&bit_string_representation[output_index], buffer_size - output_index, "%016lx", bit_string->words[i]);
                }
                break;

        case ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY:
                strcpy (bit_string_representation, "0b");
                output_index = 2;

                for (size_t i = 0; i < bit_string->number_of_words; i++) {
                        size_t word = bit_string->words[i];
                        for (size_t j = ITTY_BIT_STRING_WORD_SIZE_IN_BITS; j > 0; j--) {
                                char bit = (word & (1UL << (j - 1))) ? '1' : '0';
                                if (bit == '1' || !at_leading_zero) {
                                        bit_string_representation[output_index++] = bit;
                                        at_leading_zero = false;
                                }
                        }
                }
                bit_string_representation[output_index] = '\0';
                break;

        case ITTY_BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL_FOR_DISPLAY:
                strcpy (bit_string_representation, "0x");
                output_index = 2;

                for (size_t i = 0; i < bit_string->number_of_words; i++) {
                        char word_buffer[ITTY_BIT_STRING_WORD_SIZE_IN_BYTES * 2 + 1];
                        snprintf (word_buffer, sizeof (word_buffer), "%016lx", bit_string->words[i]);
                        for (size_t j = 0; j < strlen (word_buffer); j++) {
                                char hex_digit = word_buffer[j];
                                if (hex_digit != '0' || !at_leading_zero) {
                                        bit_string_representation[output_index++] = hex_digit;
                                        at_leading_zero = false;
                                }
                        }
                        if (i < bit_string->number_of_words - 1)
                                bit_string_representation[output_index++] = ' ';
                }
                bit_string_representation[output_index] = '\0';
                break;
        }

        return bit_string_representation;
}
