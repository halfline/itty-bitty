#define _GNU_SOURCE

#include "itty-bit-string.h"

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

#define WORD_SIZE_IN_BYTES (sizeof (size_t))
#define WORD_SIZE_IN_BITS (WORD_SIZE_IN_BYTES * CHAR_BIT)

struct itty_bit_string_t {
        size_t *words;
        size_t number_of_words;
        size_t pop_count;
        bool pop_count_computed;
};

struct itty_bit_string_list_t {
        itty_bit_string_t **bit_strings;
        size_t         count;
        size_t         max_number_of_words;
};

struct itty_bit_string_map_file_t {
        int         fd;
        size_t      file_size;
        void       *mapped_data;
        size_t      word_count_per_bit_string;
        size_t      current_index;
        itty_bit_string_list_t *bit_string_list;
};

itty_bit_string_t *
itty_bit_string_new (void)
{
        itty_bit_string_t *itty_bit_string = malloc (sizeof (itty_bit_string_t));
        itty_bit_string->words = NULL;
        itty_bit_string->number_of_words = 0;
        itty_bit_string->pop_count = 0;
        itty_bit_string->pop_count_computed = false;
        return itty_bit_string;
}

itty_bit_string_t *
itty_bit_string_duplicate (itty_bit_string_t *input_bit_string)
{
        itty_bit_string_t *bit_string = itty_bit_string_new ();
        *bit_string = *input_bit_string;
        return bit_string;
}

void
itty_bit_string_free (itty_bit_string_t *itty_bit_string)
{
        if (!itty_bit_string) {
                return;
        }
        free (itty_bit_string);
}

void
itty_bit_string_append_word (itty_bit_string_t *itty_bit_string,
                             size_t             word)
{
        itty_bit_string->words = realloc (itty_bit_string->words,
                                    (itty_bit_string->number_of_words + 1) * WORD_SIZE_IN_BYTES);
        itty_bit_string->words[itty_bit_string->number_of_words] = word;
        itty_bit_string->number_of_words++;
        itty_bit_string->pop_count_computed = false;  // Invalidate cached pop count
}

void
itty_bit_string_append_zeros (itty_bit_string_t *itty_bit_string,
                              size_t             count)
{
        for (size_t i = 0; i < count; i++) {
                itty_bit_string_append_word (itty_bit_string, 0);
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
        result->words = malloc (result->number_of_words * WORD_SIZE_IN_BYTES);

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
        result->words = malloc (result->number_of_words * WORD_SIZE_IN_BYTES);

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
        result->words = malloc (result->number_of_words * WORD_SIZE_IN_BYTES);

        for (size_t i = 0; i < max_number_of_words; i++) {
                result->words[i] = a->words[i] | b->words[i];
        }

        return result;
}

size_t
itty_bit_string_get_pop_count (itty_bit_string_t *itty_bit_string)
{
        if (!itty_bit_string->pop_count_computed) {
                itty_bit_string->pop_count = 0;
                for (size_t i = 0; i < itty_bit_string->number_of_words; i++) {
                        itty_bit_string->pop_count += __builtin_popcountl (itty_bit_string->words[i]);
                }
                itty_bit_string->pop_count_computed = true;
        }
        return itty_bit_string->pop_count;
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
        return (int) (itty_bit_string_get_pop_count (a) - itty_bit_string_get_pop_count (b));
}

itty_bit_string_list_t *
itty_bit_string_list_new (void)
{
        itty_bit_string_list_t *list = malloc (sizeof (itty_bit_string_list_t));
        list->bit_strings = NULL;
        list->count = 0;
        list->max_number_of_words = 0;
        return list;
}

void
itty_bit_string_list_free (itty_bit_string_list_t *list)
{
        if (!list) {
                return;
        }
        for (size_t i = 0; i < list->count; i++) {
                itty_bit_string_free (list->bit_strings[i]);
        }
        free (list->bit_strings);
        free (list);
}

void
itty_bit_string_list_append (itty_bit_string_list_t *list,
                             itty_bit_string_t      *bit_string)
{
        list->bit_strings = realloc (list->bit_strings,
                                    (list->count + 1) * sizeof (itty_bit_string_t *));
        list->bit_strings[list->count] = bit_string;
        list->count++;

        if (bit_string->number_of_words > list->max_number_of_words)
                list->max_number_of_words = bit_string->number_of_words;
}

size_t
itty_bit_string_list_get_length (itty_bit_string_list_t *list)
{
        return list->count;
}

itty_bit_string_list_t *
itty_bit_string_list_exclusive_or (itty_bit_string_list_t *list_a,
                                   itty_bit_string_list_t *list_b)
{
        if (!list_a || !list_b) {
                return NULL;
        }

        size_t min_count = (list_a->count < list_b->count) ? list_a->count : list_b->count;

        itty_bit_string_list_t *result_list = itty_bit_string_list_new ();

        for (size_t i = 0; i < min_count; i++) {
                itty_bit_string_t *result = itty_bit_string_exclusive_or (list_a->bit_strings[i],
                                                                          list_b->bit_strings[i]);
                itty_bit_string_list_append (result_list, result);
        }

        return result_list;
}

itty_bit_string_t *
itty_bit_string_list_fetch (itty_bit_string_list_t *list,
                            size_t                  index)
{
    if (list == NULL || index >= list->count)
            return NULL;

    return list->bit_strings[index];
}

itty_bit_string_t *
itty_bit_string_list_condense (itty_bit_string_list_t *list)
{
        if (list->count == 0) {
                return NULL;
        }

        size_t max_number_of_words = list->max_number_of_words;
        itty_bit_string_t *condensed_bit_string = itty_bit_string_new ();
        condensed_bit_string->words = calloc (max_number_of_words, sizeof (size_t));
        condensed_bit_string->number_of_words = max_number_of_words;

        size_t majority_threshold = list->count / 2 + 1;

        for (size_t word_index = 0; word_index < max_number_of_words; word_index++) {
                for (size_t bit_index = 0; bit_index < WORD_SIZE_IN_BITS; bit_index++) {
                        size_t bit_count = 0;
                        for (size_t i = 0; i < list->count; i++) {
                                size_t word = list->bit_strings[i]->words[word_index];
                                if (word & ((size_t) 1 << (WORD_SIZE_IN_BITS - 1 - bit_index))) {
                                        bit_count++;
                                }
                        }
                        if (bit_count >= majority_threshold) {
                                condensed_bit_string->words[word_index] |= ((size_t) 1 << (WORD_SIZE_IN_BITS - 1 - bit_index));
                        }
                }
        }

        return condensed_bit_string;
}

itty_bit_string_list_t *
itty_bit_string_list_transpose (itty_bit_string_list_t *list)
{
        if (list == NULL || list->count == 0) {
                return NULL;
        }

        itty_bit_string_list_t *transposed_list = itty_bit_string_list_new ();

        size_t number_of_words = list->max_number_of_words;
        for (size_t i = 0; i < WORD_SIZE_IN_BITS; i++) {
                itty_bit_string_t *new_bit_string = itty_bit_string_new();
                new_bit_string->words = calloc (number_of_words, sizeof (size_t));
                new_bit_string->number_of_words = number_of_words;
                itty_bit_string_list_append (transposed_list, new_bit_string);
        }

        itty_bit_string_list_iterator_t list_iterator;
        itty_bit_string_list_iterator_init (list, &list_iterator);
        itty_bit_string_t *original_bit_string;

        size_t current_string_index = 0;
        while (itty_bit_string_list_iterator_next (&list_iterator, &original_bit_string)) {
                itty_bit_string_iterator_t word_iterator;
                itty_bit_string_iterator_init (original_bit_string, &word_iterator);
                size_t word_index = 0;
                size_t word;

                while (itty_bit_string_iterator_next (&word_iterator, &word)) {
                        for (size_t bit = 0; bit < WORD_SIZE_IN_BITS; bit++) {
                                size_t bit_value = (word >> (WORD_SIZE_IN_BITS - 1 - bit)) & 1;
                                itty_bit_string_t *transposed_bit_string = transposed_list->bit_strings[bit];
                                transposed_bit_string->words[word_index] |= bit_value << (list->count - 1 - current_string_index);
                        }
                        word_index++;
                }
                current_string_index++;
        }

        return transposed_list;
}

size_t
itty_bit_string_list_get_max_number_of_words (itty_bit_string_list_t *list)
{
        return list->max_number_of_words;
}

void
itty_bit_string_list_iterator_init (itty_bit_string_list_t          *list,
                                    itty_bit_string_list_iterator_t *iterator)
{
        iterator->list = list;
        iterator->current_index = 0;
}

bool
itty_bit_string_list_iterator_next (itty_bit_string_list_iterator_t  *iterator,
                                    itty_bit_string_t               **bit_string)
{
        if (iterator->current_index < iterator->list->count) {
                *bit_string = iterator->list->bit_strings[iterator->current_index];
                iterator->current_index++;
                return true;
        } else {
                *bit_string = NULL;
                return false;
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
itty_bit_string_split (itty_bit_string_t *itty_bit_string,
                       size_t             number_of_bit_strings)
{
        if (number_of_bit_strings == 0) {
                return NULL;
        }

        size_t total_bits = itty_bit_string->number_of_words * WORD_SIZE_IN_BITS;
        size_t bits_per_split = (total_bits + number_of_bit_strings - 1) / number_of_bit_strings;
        size_t words_per_split = (bits_per_split + WORD_SIZE_IN_BITS - 1) / WORD_SIZE_IN_BITS;

        itty_bit_string_list_t *split_list = itty_bit_string_list_new ();
        for (size_t i = 0; i < number_of_bit_strings; i++) {
                itty_bit_string_t *split = itty_bit_string_new ();
                for (size_t j = 0; j < words_per_split; j++) {
                        size_t word_index = i * words_per_split + j;
                        size_t word = (word_index < itty_bit_string->number_of_words) ? itty_bit_string->words[word_index] : 0;
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
        result->words = malloc (total_words * WORD_SIZE_IN_BYTES);
        if (!result->words) {
                free (result);
                return NULL;
        }

        memcpy (result->words, a->words, a->number_of_words * WORD_SIZE_IN_BYTES);
        memcpy (result->words + a->number_of_words, b->words, b->number_of_words * WORD_SIZE_IN_BYTES);

        result->number_of_words = total_words;
        result->pop_count_computed = false;

        return result;
}

itty_bit_string_t *
itty_bit_string_double (itty_bit_string_t *a)
{
        return itty_bit_string_concatenate (a, a);
}

itty_bit_string_t *
itty_bit_string_reduce_by_half (itty_bit_string_t *itty_bit_string)
{
        assert (itty_bit_string->number_of_words % 2 == 0);

        size_t half_number_of_words = itty_bit_string->number_of_words / 2;

        itty_bit_string_t *first_half = itty_bit_string_new ();
        first_half->words = itty_bit_string->words;
        first_half->number_of_words = half_number_of_words;
        first_half->pop_count_computed = false;

        itty_bit_string_t *second_half = itty_bit_string_new ();
        second_half->words = itty_bit_string->words + half_number_of_words;
        second_half->number_of_words = half_number_of_words;
        second_half->pop_count_computed = false;

        itty_bit_string_t *reduced_bit_string = itty_bit_string_exclusive_nor (first_half, second_half);

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
        size_t total_bits = bit_string->number_of_words * WORD_SIZE_IN_BITS;
        size_t buffer_size = 0;

        switch (format) {
                case BIT_STRING_PRESENTATION_FORMAT_BINARY:
                case BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY:
                        buffer_size = total_bits;

                        if (format == BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY)
                                buffer_size += strlen ("0b");
                        break;
                case BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL:
                case BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL_FOR_DISPLAY:
                        buffer_size = bit_string->number_of_words * (WORD_SIZE_IN_BYTES * 2 + 1);

                        if (format == BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL_FOR_DISPLAY)
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
        case BIT_STRING_PRESENTATION_FORMAT_BINARY:
                for (size_t i = 0; i < bit_string->number_of_words; i++) {
                        size_t word = bit_string->words[i];
                        for (size_t j = WORD_SIZE_IN_BITS; j > 0; j--) {
                                bit_string_representation[output_index++] = (word & (1UL << (j - 1))) ? '1' : '0';
                        }
                }
                bit_string_representation[output_index] = '\0';
                break;

        case BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL:
                bit_string_representation[output_index] = '\0';
                for (size_t i = 0; i < bit_string->number_of_words; i++) {
                        output_index += snprintf (&bit_string_representation[output_index], buffer_size - output_index, "%016lx", bit_string->words[i]);
                }
                break;

        case BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY:
                strcpy (bit_string_representation, "0b");
                output_index = 2;

                for (size_t i = 0; i < bit_string->number_of_words; i++) {
                        size_t word = bit_string->words[i];
                        for (size_t j = WORD_SIZE_IN_BITS; j > 0; j--) {
                                char bit = (word & (1UL << (j - 1))) ? '1' : '0';
                                if (bit == '1' || !at_leading_zero) {
                                        bit_string_representation[output_index++] = bit;
                                        at_leading_zero = false;
                                }
                        }
                }
                bit_string_representation[output_index] = '\0';
                break;

        case BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL_FOR_DISPLAY:
                strcpy (bit_string_representation, "0x");
                output_index = 2;

                for (size_t i = 0; i < bit_string->number_of_words; i++) {
                        char word_buffer[WORD_SIZE_IN_BYTES * 2 + 1];
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

itty_bit_string_list_t *
itty_bit_string_list_popcount_softmax (itty_bit_string_list_t *list,
                                       size_t                  num_words)
{
        itty_bit_string_list_t *softmax_list = itty_bit_string_list_new ();

        size_t total_popcount = 0;
        itty_bit_string_list_iterator_t iterator;
        itty_bit_string_list_iterator_init (list, &iterator);
        itty_bit_string_t *itty_bit_string;

        while (itty_bit_string_list_iterator_next (&iterator, &itty_bit_string)) {
                total_popcount += itty_bit_string_get_pop_count (itty_bit_string);
        }

        size_t cumulative_ones = 0;
        itty_bit_string_list_iterator_init (list, &iterator);
        while (itty_bit_string_list_iterator_next (&iterator, &itty_bit_string)) {
                itty_bit_string_t *new_itty_bit_string = itty_bit_string_new ();
                size_t num_ones = (itty_bit_string_get_pop_count (itty_bit_string) * WORD_SIZE_IN_BITS + total_popcount - 1) / total_popcount;
                cumulative_ones += num_ones;

                size_t remaining_ones = num_ones;
                for (size_t i = 0; i < num_words; i++) {
                        size_t word = 0;
                        if (remaining_ones > 0) {
                                size_t ones_in_word = remaining_ones > WORD_SIZE_IN_BITS ? WORD_SIZE_IN_BITS : remaining_ones;
                                word = (1UL << ones_in_word) - 1;
                                remaining_ones -= ones_in_word;
                        }
                        itty_bit_string_append_word (new_itty_bit_string, word);
                }

                itty_bit_string_list_append (softmax_list, new_itty_bit_string);
        }

        size_t total_bits = WORD_SIZE_IN_BITS * num_words;
        if (cumulative_ones != total_bits && softmax_list->count > 0) {
                size_t excess_ones = cumulative_ones - total_bits;
                size_t last_word_index = num_words - 1;
                softmax_list->bit_strings[softmax_list->count - 1]->words[last_word_index] >>= excess_ones;
        }

        return softmax_list;
}

bool
itty_bit_string_list_popcount_argmax (itty_bit_string_list_t *list,
                                      size_t                  num_words,
                                      size_t                 *index)
{
        itty_bit_string_list_iterator_t iterator;
        itty_bit_string_t *current_bit_string;
        size_t highest_popcount = 0;
        size_t i = 0;
        bool found_one = false;

        itty_bit_string_list_t *softmax = itty_bit_string_list_popcount_softmax (list, num_words);
        itty_bit_string_list_iterator_init (softmax, &iterator);
        while (itty_bit_string_list_iterator_next (&iterator, &current_bit_string)) {
                size_t current_popcount = itty_bit_string_get_pop_count (current_bit_string);

                if (!found_one) {
                        found_one = true;
                        if (index)
                                *index = i;
                }

                if (current_popcount > highest_popcount) {
                        highest_popcount = current_popcount;

                        if (index)
                                *index = i;
                }
                i++;
        }
        itty_bit_string_list_free (softmax);

        return found_one;
}

int
itty_bit_string_compare_qsort (const void *a,
                               const void *b,
                               void       *order)
{
        itty_bit_string_t *itty_bit_string_a = *(itty_bit_string_t **) a;
        itty_bit_string_t *itty_bit_string_b = *(itty_bit_string_t **) b;
        itty_bit_string_sort_order_t sort_order = *(itty_bit_string_sort_order_t *) order;
        return (sort_order == BIT_STRING_SORT_ORDER_ASCENDING) ? itty_bit_string_compare (itty_bit_string_a, itty_bit_string_b)
                                                               : itty_bit_string_compare (itty_bit_string_b, itty_bit_string_a);
}

void
itty_bit_string_list_sort (itty_bit_string_list_t      *list,
                           itty_bit_string_sort_order_t order)
{
        qsort_r (list->bit_strings, list->count, sizeof (itty_bit_string_t *), itty_bit_string_compare_qsort, &order);
}

itty_bit_string_map_file_t *
itty_bit_string_map_file_new (const char *file_name)
{
        itty_bit_string_map_file_t *mapped_file = malloc (sizeof (itty_bit_string_map_file_t));

        mapped_file->fd = open (file_name, O_RDONLY);
        if (mapped_file->fd == -1) {
                free (mapped_file);
                return NULL;
        }

        struct stat sb;
        if (fstat (mapped_file->fd, &sb) == -1) {
                close (mapped_file->fd);
                free (mapped_file);
                return NULL;
        }

        mapped_file->file_size = sb.st_size;
        mapped_file->mapped_data = mmap (NULL, mapped_file->file_size, PROT_READ, MAP_PRIVATE, mapped_file->fd, 0);
        if (mapped_file->mapped_data == MAP_FAILED) {
                close (mapped_file->fd);
                free (mapped_file);
                return NULL;
        }

        mapped_file->current_index = 0;
        mapped_file->bit_string_list = itty_bit_string_list_new ();

        return mapped_file;
}

void
itty_bit_string_map_file_free (itty_bit_string_map_file_t *mapped_file)
{
        if (!mapped_file) {
                return;
        }

        itty_bit_string_list_iterator_t iterator;
        itty_bit_string_list_iterator_init (mapped_file->bit_string_list, &iterator);
        itty_bit_string_t *itty_bit_string;
        while (itty_bit_string_list_iterator_next (&iterator, &itty_bit_string)) {
                itty_bit_string->words = NULL;  // Clear the words pointer
        }

        itty_bit_string_list_free (mapped_file->bit_string_list);
        munmap (mapped_file->mapped_data, mapped_file->file_size);
        close (mapped_file->fd);
        free (mapped_file);
}

itty_bit_string_t *
itty_bit_string_map_file_next (itty_bit_string_map_file_t  *mapped_file,
                               size_t                       number_of_words)
{
        size_t total_words = mapped_file->file_size / WORD_SIZE_IN_BYTES;
        if (mapped_file->current_index >= total_words) {
                return NULL;
        }

        itty_bit_string_t *bit_string = itty_bit_string_new ();
        bit_string->words = (size_t *) (mapped_file->mapped_data) + mapped_file->current_index;
        bit_string->number_of_words = number_of_words;
        bit_string->pop_count_computed = false;

        mapped_file->current_index += bit_string->number_of_words;

        return bit_string;
}
