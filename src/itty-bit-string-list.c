#include "itty-bit-string-list.h"
#include "itty-bit-string-list-private.h"
#include "itty-bit-string.h"
#include "itty-bit-string-private.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
        size_t     index;
        __uint128_t remainder;
} itty_popcount_vote_remainder_t;

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

        size_t bit_length = itty_bit_string_list_get_bit_length (list);
        size_t number_of_words = (bit_length + ITTY_BIT_STRING_WORD_SIZE_IN_BITS - 1) / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        itty_bit_string_t *condensed_bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        condensed_bit_string->words = calloc (number_of_words, sizeof (size_t));
        condensed_bit_string->number_of_words = number_of_words;

        size_t majority_threshold = list->count / 2 + 1;
        for (size_t bit_index = 0; bit_index < bit_length; bit_index++) {
                size_t one_votes = 0;

                for (size_t string_index = 0; string_index < list->count; string_index++) {
                        if (itty_bit_string_get_bit (list->bit_strings[string_index], bit_index))
                                one_votes++;
                }

                if (one_votes >= majority_threshold)
                        itty_bit_string_set_bit (condensed_bit_string, bit_index, true);
        }

        return condensed_bit_string;
}

itty_bit_string_t *
itty_bit_string_list_weighted_condense (itty_bit_string_list_t *list,
                                        size_t const           *votes,
                                        size_t                  vote_count)
{
        if (!list || list->count == 0 || !votes || vote_count != list->count)
                return NULL;

        size_t bit_length = itty_bit_string_list_get_bit_length (list);
        size_t number_of_words = (bit_length + ITTY_BIT_STRING_WORD_SIZE_IN_BITS - 1) / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        itty_bit_string_t *condensed_bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        if (!condensed_bit_string)
                return NULL;

        condensed_bit_string->words = calloc (number_of_words, sizeof (size_t));
        if (number_of_words > 0 && !condensed_bit_string->words) {
                itty_bit_string_free (condensed_bit_string);
                return NULL;
        }
        condensed_bit_string->number_of_words = number_of_words;

        __uint128_t total_votes = 0;
        for (size_t i = 0; i < vote_count; i++)
                total_votes += votes[i];

        if (total_votes == 0)
                return condensed_bit_string;

        __uint128_t majority_threshold = total_votes / 2 + 1;
        for (size_t bit_index = 0; bit_index < bit_length; bit_index++) {
                __uint128_t one_votes = 0;

                for (size_t string_index = 0; string_index < list->count; string_index++) {
                        if (itty_bit_string_get_bit (list->bit_strings[string_index], bit_index))
                                one_votes += votes[string_index];
                }

                if (one_votes >= majority_threshold)
                        itty_bit_string_set_bit (condensed_bit_string, bit_index, true);
        }

        return condensed_bit_string;
}

itty_bit_string_list_t *
itty_bit_string_list_transpose (itty_bit_string_list_t *list)
{
        if (!list || list->count == 0) {
                return NULL;
        }

        size_t bit_length = itty_bit_string_list_get_bit_length (list);
        size_t number_of_words = (list->count + ITTY_BIT_STRING_WORD_SIZE_IN_BITS - 1) / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

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

static int
compare_vote_remainders_descending (const void *a,
                                    const void *b)
{
        const itty_popcount_vote_remainder_t *remainder_a = a;
        const itty_popcount_vote_remainder_t *remainder_b = b;

        if (remainder_a->remainder > remainder_b->remainder)
                return -1;
        if (remainder_a->remainder < remainder_b->remainder)
                return 1;
        if (remainder_a->index < remainder_b->index)
                return -1;
        if (remainder_a->index > remainder_b->index)
                return 1;
        return 0;
}

bool
itty_popcount_allocate_votes (size_t const *scores,
                              size_t        score_count,
                              size_t        total_votes,
                              size_t       *votes)
{
        if (score_count == 0)
                return true;

        if (!scores || !votes)
                return false;

        __uint128_t total_score = 0;
        for (size_t i = 0; i < score_count; i++) {
                votes[i] = 0;
                total_score += scores[i];
        }

        if (total_score == 0 || total_votes == 0)
                return true;

        itty_popcount_vote_remainder_t *remainders = calloc (score_count, sizeof (itty_popcount_vote_remainder_t));
        if (!remainders)
                return false;

        size_t allocated_votes = 0;
        for (size_t i = 0; i < score_count; i++) {
                __uint128_t scaled_score = (__uint128_t) scores[i] * total_votes;
                votes[i] = (size_t) (scaled_score / total_score);
                remainders[i] = (itty_popcount_vote_remainder_t) {
                        i,
                        scaled_score % total_score
                };
                allocated_votes += votes[i];
        }

        qsort (remainders, score_count, sizeof (itty_popcount_vote_remainder_t), compare_vote_remainders_descending);

        size_t remaining_votes = total_votes - allocated_votes;
        for (size_t i = 0; i < remaining_votes; i++)
                votes[remainders[i].index]++;

        free (remainders);

        return true;
}

bool
itty_bit_string_list_allocate_popcount_votes (itty_bit_string_list_t *list,
                                              size_t                  total_votes,
                                              size_t                 *votes)
{
        if (!list || !votes)
                return false;

        if (list->count == 0)
                return true;

        size_t *scores = calloc (list->count, sizeof (size_t));
        if (!scores)
                return false;

        for (size_t i = 0; i < list->count; i++)
                scores[i] = itty_bit_string_get_pop_count (list->bit_strings[i]);

        bool allocated = itty_popcount_allocate_votes (scores, list->count, total_votes, votes);
        free (scores);

        return allocated;
}

static itty_bit_string_t *
itty_bit_string_new_vote_mask (size_t number_of_words,
                               size_t number_of_votes)
{
        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        if (!bit_string)
                return NULL;

        for (size_t i = 0; i < number_of_words; i++) {
                size_t ones_in_word = number_of_votes > ITTY_BIT_STRING_WORD_SIZE_IN_BITS ?
                        ITTY_BIT_STRING_WORD_SIZE_IN_BITS : number_of_votes;
                size_t word = 0;

                if (ones_in_word == ITTY_BIT_STRING_WORD_SIZE_IN_BITS)
                        word = ~0UL;
                else if (ones_in_word > 0)
                        word = (1UL << ones_in_word) - 1;

                itty_bit_string_append_word (bit_string, word);
                number_of_votes -= ones_in_word;
        }

        return bit_string;
}

itty_bit_string_list_t *
itty_bit_string_list_make_popcount_vote_masks (itty_bit_string_list_t *list,
                                               size_t                  num_words)
{
        if (!list)
                return NULL;

        itty_bit_string_list_t *vote_masks = itty_bit_string_list_new ();
        if (!vote_masks)
                return NULL;

        size_t *votes = calloc (list->count, sizeof (size_t));
        if (!votes) {
                itty_bit_string_list_free (vote_masks);
                return NULL;
        }

        size_t total_votes = ITTY_BIT_STRING_WORD_SIZE_IN_BITS * num_words;
        if (!itty_bit_string_list_allocate_popcount_votes (list, total_votes, votes)) {
                free (votes);
                itty_bit_string_list_free (vote_masks);
                return NULL;
        }

        for (size_t i = 0; i < list->count; i++) {
                itty_bit_string_t *vote_mask = itty_bit_string_new_vote_mask (num_words, votes[i]);
                if (!vote_mask) {
                        free (votes);
                        itty_bit_string_list_free (vote_masks);
                        return NULL;
                }
                itty_bit_string_list_append (vote_masks, vote_mask);
        }

        free (votes);

        return vote_masks;
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

        (void) num_words;

        itty_bit_string_list_iterator_init (list, &iterator);
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
