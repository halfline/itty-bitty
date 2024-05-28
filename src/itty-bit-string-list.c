#include "itty-bit-string-list.h"
#include "itty-bit-string-list-private.h"
#include "itty-bit-string.h"
#include "itty-bit-string-private.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

        size_t number_of_words = itty_bit_string_get_number_of_words (bit_string);
        if (number_of_words > list->max_number_of_words)
                list->max_number_of_words = number_of_words;
}

size_t
itty_bit_string_list_get_length (itty_bit_string_list_t *list)
{
        return list->count;
}

size_t
itty_bit_string_list_get_bit_length (itty_bit_string_list_t *list)
{
        size_t max_bit_length = 0;
        for (size_t i = 0; i < list->count; i++) {
                size_t bit_length = itty_bit_string_get_length (list->bit_strings[i]);
                if (bit_length > max_bit_length) {
                        max_bit_length = bit_length;
                }
        }
        return max_bit_length;
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

        itty_bit_string_list_t *transposed_list = itty_bit_string_list_transpose (list);
        if (!transposed_list) {
                return NULL;
        }

        size_t max_number_of_words = transposed_list->max_number_of_words;
        itty_bit_string_t *condensed_bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        condensed_bit_string->words = calloc (max_number_of_words, sizeof (size_t));
        condensed_bit_string->number_of_words = max_number_of_words;

        size_t majority_threshold = list->count / 2 + 1;

        for (size_t i = 0; i < transposed_list->count; i++) {
                itty_bit_string_t *bit_string = transposed_list->bit_strings[i];
                size_t pop_count = itty_bit_string_get_pop_count(bit_string);

                if (pop_count >= majority_threshold) {
                        for (size_t word_index = 0; word_index < bit_string->number_of_words; word_index++) {
                                condensed_bit_string->words[word_index] |= ((size_t) 1 << (transposed_list->count - 1 - i));
                        }
                }
        }

        itty_bit_string_list_free (transposed_list);

        return condensed_bit_string;
}

itty_bit_string_list_t *
itty_bit_string_list_transpose (itty_bit_string_list_t *list)
{
        if (!list || list->count == 0) {
                return NULL;
        }

        size_t bit_length = itty_bit_string_list_get_bit_length (list);
        size_t number_of_words = (bit_length + ITTY_BIT_STRING_WORD_SIZE_IN_BITS - 1) / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        itty_bit_string_list_t *transposed_list = itty_bit_string_list_new ();

        for (size_t bit_position = bit_length; bit_position > 0; bit_position--) {
                itty_bit_string_t *transposed_bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
                itty_bit_string_append_zeros (transposed_bit_string, number_of_words);

                itty_bit_string_list_iterator_t list_iterator;
                itty_bit_string_list_iterator_init (list, &list_iterator);
                itty_bit_string_t *bit_string;
                size_t string_index = 0;

                while (itty_bit_string_list_iterator_next (&list_iterator, &bit_string)) {
                        itty_bit_string_iterator_t bit_string_iterator;
                        itty_bit_string_iterator_init_at_word_offset (bit_string, &bit_string_iterator, (bit_position - 1) / ITTY_BIT_STRING_WORD_SIZE_IN_BITS);

                        size_t word;
                        if (itty_bit_string_iterator_next (&bit_string_iterator, &word)) {
                                size_t bit_in_word = (bit_position - 1) % ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
                                bool bit_value = (word & (1UL << bit_in_word)) != 0;
                                itty_bit_string_set_bit (transposed_bit_string, string_index, bit_value);
                        }
                        string_index++;
                }
                itty_bit_string_list_append (transposed_list, transposed_bit_string);
        }
        return transposed_list;
}

size_t
itty_bit_string_list_get_max_number_of_words (itty_bit_string_list_t *list)
{
        return list->max_number_of_words;
}

char *
itty_bit_string_list_present (itty_bit_string_list_t                *bit_string_list,
                              itty_bit_string_presentation_format_t  format)
{
        size_t buffer_size = 0;
        size_t buffer_used = 0;
        char *list_representation = NULL;
        size_t max_length = 0;

        bool is_display_format = (format == ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY ||
                                  format == ITTY_BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL_FOR_DISPLAY);

        for (size_t i = 0; i < bit_string_list->count; i++) {
                char *bit_string_representation = itty_bit_string_present (bit_string_list->bit_strings[i], format);
                if (bit_string_representation) {
                        size_t length = strlen (bit_string_representation);
                        if (length > max_length) {
                                max_length = length;
                        }
                        free (bit_string_representation);
                }
        }

        if (is_display_format) {
                buffer_size = strlen ("\t[\n\t]\n") + 1;
        } else {
                buffer_size = 1;
        }

        for (size_t i = 0; i < bit_string_list->count; i++) {
                char *bit_string_representation = itty_bit_string_present (bit_string_list->bit_strings[i], format);
                if (bit_string_representation) {
                        size_t representation_length = max_length + (is_display_format ? strlen ("\t\t\n") : 0);
                        buffer_size += representation_length;
                        free (bit_string_representation);
                }
        }

        list_representation = malloc (buffer_size);
        if (!list_representation) {
                return NULL;
        }

        if (is_display_format) {
                buffer_used += sprintf (&list_representation[buffer_used], "\t[\n");
        }

        for (size_t i = 0; i < bit_string_list->count; i++) {
                char *bit_string_representation = itty_bit_string_present (bit_string_list->bit_strings[i], format);
                if (!bit_string_representation)
                        continue;

                if (is_display_format) {
                        buffer_used += sprintf (&list_representation[buffer_used], "\t\t%*s\n", (int) max_length, bit_string_representation);
                } else {
                        buffer_used += sprintf (&list_representation[buffer_used], "%s", bit_string_representation);
                }

                free (bit_string_representation);
        }

        if (is_display_format) {
                buffer_used += sprintf (&list_representation[buffer_used], "\t]\n");
        }

        return list_representation;
}

itty_bit_string_list_t *
itty_bit_string_list_popcount_softmax (itty_bit_string_list_t *list,
                                       size_t                  num_words)
{
        itty_bit_string_list_t *softmax_list = itty_bit_string_list_new ();

        size_t total_popcount = 0;
        itty_bit_string_list_iterator_t iterator;
        itty_bit_string_list_iterator_init (list, &iterator);
        itty_bit_string_t *bit_string;

        while (itty_bit_string_list_iterator_next (&iterator, &bit_string)) {
                total_popcount += itty_bit_string_get_pop_count (bit_string);
        }

        size_t cumulative_ones = 0;
        itty_bit_string_list_iterator_init (list, &iterator);
        while (itty_bit_string_list_iterator_next (&iterator, &bit_string)) {
                itty_bit_string_t *new_bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
                size_t num_ones = (itty_bit_string_get_pop_count (bit_string) * ITTY_BIT_STRING_WORD_SIZE_IN_BITS + total_popcount - 1) / total_popcount;
                cumulative_ones += num_ones;

                size_t remaining_ones = num_ones;
                for (size_t i = 0; i < num_words; i++) {
                        size_t word = 0;
                        if (remaining_ones > 0) {
                                size_t ones_in_word = remaining_ones > ITTY_BIT_STRING_WORD_SIZE_IN_BITS ? ITTY_BIT_STRING_WORD_SIZE_IN_BITS : remaining_ones;
                                word = (1UL << ones_in_word) - 1;
                                remaining_ones -= ones_in_word;
                        }
                        itty_bit_string_append_word (new_bit_string, word);
                }

                itty_bit_string_list_append (softmax_list, new_bit_string);
        }

        size_t total_bits = ITTY_BIT_STRING_WORD_SIZE_IN_BITS * num_words;
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

static int
itty_bit_string_compare_by_pop_count_qsort (const void *a,
                                            const void *b,
                                            void       *order)
{
        itty_bit_string_t *itty_bit_string_a = *(itty_bit_string_t **) a;
        itty_bit_string_t *itty_bit_string_b = *(itty_bit_string_t **) b;
        itty_bit_string_sort_order_t sort_order = *(itty_bit_string_sort_order_t *) order;
        return (sort_order == ITTY_BIT_STRING_SORT_ORDER_ASCENDING) ?
                itty_bit_string_compare_by_pop_count (itty_bit_string_a, itty_bit_string_b) :
                itty_bit_string_compare_by_pop_count (itty_bit_string_b, itty_bit_string_a);
}

void
itty_bit_string_list_sort (itty_bit_string_list_t      *list,
                           itty_bit_string_sort_order_t order)
{
        qsort_r (list->bit_strings, list->count, sizeof (itty_bit_string_t *), itty_bit_string_compare_by_pop_count_qsort, &order);
}

void
itty_bit_string_list_iterator_init_at_index (itty_bit_string_list_t          *list,
                                             itty_bit_string_list_iterator_t *iterator,
                                             size_t                          index)
{
        if (index < list->count) {
                iterator->list = list;
                iterator->current_index = index;
        } else {
                iterator->list = list;
                iterator->current_index = 0;
        }
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
