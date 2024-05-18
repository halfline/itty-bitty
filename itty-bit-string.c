#define _GNU_SOURCE
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

typedef struct {
        size_t *words;
        size_t number_of_words;
        size_t pop_count;
        bool pop_count_computed;
} itty_bit_string_t;

typedef struct {
        itty_bit_string_t **itty_bit_strings;
        size_t         count;
} itty_bit_string_list_t;

typedef struct {
        itty_bit_string_list_t *list;
        size_t             current_index;
} itty_bit_string_list_iterator_t;

typedef struct {
        itty_bit_string_t *itty_bit_string;
        size_t        current_index;
} itty_bit_string_iterator_t;

typedef enum {
        BIT_STRING_PRESENTATION_FORMAT_BINARY,
        BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL
} itty_bit_string_presentation_format_t;

typedef enum {
        BIT_STRING_SORT_ORDER_ASCENDING,
        BIT_STRING_SORT_ORDER_DESCENDING
} itty_bit_string_sort_order_t;

typedef struct {
        int         fd;
        size_t      file_size;
        void       *mapped_data;
        size_t      word_count_per_itty_bit_string;
        size_t      current_index;
        itty_bit_string_list_t *itty_bit_string_list;
} itty_bit_string_map_file_t;

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

void
itty_bit_string_free (itty_bit_string_t *itty_bit_string)
{
        if (!itty_bit_string) {
                return;
        }
        free (itty_bit_string->words);
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
        size_t max_length = a->number_of_words > b->number_of_words ? a->number_of_words : b->number_of_words;
        size_t a_padding = max_length - a->number_of_words;
        size_t b_padding = max_length - b->number_of_words;

        if (a_padding > 0) {
                itty_bit_string_append_zeros (a, a_padding);
        }

        if (b_padding > 0) {
                itty_bit_string_append_zeros (b, b_padding);
        }

        itty_bit_string_t *result = itty_bit_string_new ();
        result->number_of_words = max_length;
        result->words = malloc (result->number_of_words * WORD_SIZE_IN_BYTES);

        for (size_t i = 0; i < max_length; i++) {
                result->words[i] = ~(a->words[i] ^ b->words[i]);
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
        list->itty_bit_strings = NULL;
        list->count = 0;
        return list;
}

void
itty_bit_string_list_free (itty_bit_string_list_t *list)
{
        if (!list) {
                return;
        }
        for (size_t i = 0; i < list->count; i++) {
                itty_bit_string_free (list->itty_bit_strings[i]);
        }
        free (list->itty_bit_strings);
        free (list);
}

void
itty_bit_string_list_append (itty_bit_string_list_t *list,
                             itty_bit_string_t      *itty_bit_string)
{
        list->itty_bit_strings = realloc (list->itty_bit_strings,
                                    (list->count + 1) * sizeof (itty_bit_string_t *));
        list->itty_bit_strings[list->count] = itty_bit_string;
        list->count++;
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
                                    itty_bit_string_t               **itty_bit_string)
{
        if (iterator->current_index < iterator->list->count) {
                *itty_bit_string = iterator->list->itty_bit_strings[iterator->current_index];
                iterator->current_index++;
                return true;
        } else {
                *itty_bit_string = NULL;
                return false;
        }
}

void
itty_bit_string_iterator_init (itty_bit_string_t          *itty_bit_string,
                               itty_bit_string_iterator_t *iterator)
{
        iterator->itty_bit_string = itty_bit_string;
        iterator->current_index = 0;
}

bool
itty_bit_string_iterator_next (itty_bit_string_iterator_t *iterator,
                               size_t                     *word)
{
        if (iterator->current_index < iterator->itty_bit_string->number_of_words) {
                *word = iterator->itty_bit_string->words[iterator->current_index];
                iterator->current_index++;
                return true;
        } else {
                *word = 0;
                return false;
        }
}

itty_bit_string_list_t *
itty_bit_string_split (itty_bit_string_t *itty_bit_string,
                       size_t             number_of_itty_bit_strings)
{
        if (number_of_itty_bit_strings == 0) {
                return NULL;
        }

        size_t total_bits = itty_bit_string->number_of_words * WORD_SIZE_IN_BITS;
        size_t bits_per_split = (total_bits + number_of_itty_bit_strings - 1) / number_of_itty_bit_strings;
        size_t words_per_split = (bits_per_split + WORD_SIZE_IN_BITS - 1) / WORD_SIZE_IN_BITS;

        itty_bit_string_list_t *split_list = itty_bit_string_list_new ();
        for (size_t i = 0; i < number_of_itty_bit_strings; i++) {
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
itty_bit_string_present (itty_bit_string_t                    *itty_bit_string,
                         itty_bit_string_presentation_format_t format)
{
        size_t total_bits = itty_bit_string->number_of_words * WORD_SIZE_IN_BITS;
        size_t buffer_size;

        if (format == BIT_STRING_PRESENTATION_FORMAT_BINARY) {
                buffer_size = total_bits + 1;
        } else {
                buffer_size = itty_bit_string->number_of_words * (WORD_SIZE_IN_BYTES * 2 + 1);
        }

        char *itty_bit_string_representation = malloc (buffer_size);
        if (!itty_bit_string_representation) {
                return NULL;
        }

        if (format == BIT_STRING_PRESENTATION_FORMAT_BINARY) {
                itty_bit_string_representation[total_bits] = '\0';
                size_t itty_bitindex = 0;
                for (size_t i = 0; i < itty_bit_string->number_of_words; i++) {
                        size_t word = itty_bit_string->words[i];
                        for (size_t j = WORD_SIZE_IN_BITS; j > 0; j--) {
                                itty_bit_string_representation[itty_bitindex++] = (word & (1UL << (j - 1))) ? '1' : '0';
                        }
                }
        } else {
                size_t buffer_index = 0;
                for (size_t i = 0; i < itty_bit_string->number_of_words; i++) {
                        int written = snprintf (&itty_bit_string_representation[buffer_index], buffer_size - buffer_index, "%016lx ", itty_bit_string->words[i]);
                        buffer_index += written;
                }
                itty_bit_string_representation[buffer_index - 1] = '\0';
        }

        return itty_bit_string_representation;
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
                softmax_list->itty_bit_strings[softmax_list->count - 1]->words[last_word_index] >>= excess_ones;
        }

        return softmax_list;
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
        qsort_r (list->itty_bit_strings, list->count, sizeof (itty_bit_string_t *), itty_bit_string_compare_qsort, &order);
}

itty_bit_string_map_file_t *
itty_bit_string_map_file_new (const char *file_name,
                              size_t      word_count_per_itty_bit_string)
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

        mapped_file->word_count_per_itty_bit_string = word_count_per_itty_bit_string;
        mapped_file->current_index = 0;
        mapped_file->itty_bit_string_list = itty_bit_string_list_new ();

        return mapped_file;
}

void
itty_bit_string_map_file_free (itty_bit_string_map_file_t *mapped_file)
{
        if (!mapped_file) {
                return;
        }

        itty_bit_string_list_iterator_t iterator;
        itty_bit_string_list_iterator_init (mapped_file->itty_bit_string_list, &iterator);
        itty_bit_string_t *itty_bit_string;
        while (itty_bit_string_list_iterator_next (&iterator, &itty_bit_string)) {
                itty_bit_string->words = NULL;  // Clear the words pointer
        }

        itty_bit_string_list_free (mapped_file->itty_bit_string_list);
        munmap (mapped_file->mapped_data, mapped_file->file_size);
        close (mapped_file->fd);
        free (mapped_file);
}

itty_bit_string_t *
itty_bit_string_map_file_get_next (itty_bit_string_map_file_t *mapped_file)
{
        size_t total_words = mapped_file->file_size / WORD_SIZE_IN_BYTES;
        if (mapped_file->current_index >= total_words) {
                return NULL;
        }

        itty_bit_string_t *itty_bit_string = itty_bit_string_new ();
        itty_bit_string->words = (size_t *) (mapped_file->mapped_data) + mapped_file->current_index;
        itty_bit_string->number_of_words = mapped_file->word_count_per_itty_bit_string;
        itty_bit_string->pop_count_computed = false;

        mapped_file->current_index += itty_bit_string->number_of_words;
        return itty_bit_string;
}

bool
itty_bit_string_map_file_next (itty_bit_string_map_file_t *mapped_file)
{
        itty_bit_string_t *itty_bit_string = itty_bit_string_map_file_get_next (mapped_file);
        if (!itty_bit_string) {
                return false;
        }

        itty_bit_string_list_append (mapped_file->itty_bit_string_list, itty_bit_string);
        return true;
}

itty_bit_string_list_t *
itty_bit_string_map_file_peek_at_string_list (itty_bit_string_map_file_t *mapped_file)
{
        return mapped_file->itty_bit_string_list;
}

int
main (void)
{
        itty_bit_string_t *a = itty_bit_string_new ();
        itty_bit_string_t *b = itty_bit_string_new ();

        itty_bit_string_append_word (a, 0b10101010);
        itty_bit_string_append_word (a, 0b11110000);
        itty_bit_string_append_word (a, 0b11111111);
        itty_bit_string_append_word (a, 0b00000000);

        itty_bit_string_append_word (b, 0b11001100);

        printf ("Bit string a:\n");
        char *a_binary_representation = itty_bit_string_present (a, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        if (a_binary_representation) {
                printf ("  Binary: %s\n", a_binary_representation);
                free (a_binary_representation);
        }
        char *a_hexadecimal_representation = itty_bit_string_present (a, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
        if (a_hexadecimal_representation) {
                printf ("  Hexadecimal: %s\n", a_hexadecimal_representation);
                free (a_hexadecimal_representation);
        }

        printf ("Bit string b:\n");
        char *b_binary_representation = itty_bit_string_present (b, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        if (b_binary_representation) {
                printf ("  Binary: %s\n", b_binary_representation);
                free (b_binary_representation);
        }
        char *b_hexadecimal_representation = itty_bit_string_present (b, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
        if (b_hexadecimal_representation) {
                printf ("  Hexadecimal: %s\n", b_hexadecimal_representation);
                free (b_hexadecimal_representation);
        }

        size_t similarity = itty_bit_string_evaluate_similarity (a, b);
        printf ("Rate of similarity: %zu\n", similarity);

        int comparison = itty_bit_string_compare (a, b);
        printf ("Pop count comparison (a - b): %d\n", comparison);

        size_t number_of_itty_bit_strings = 3;
        itty_bit_string_list_t *string_list = itty_bit_string_split (a, number_of_itty_bit_strings);
        printf ("Splits:\n");

        itty_bit_string_list_iterator_t list_iterator;
        itty_bit_string_list_iterator_init (string_list, &list_iterator);

        itty_bit_string_t *itty_bit_string;
        size_t i = 0;
        while (itty_bit_string_list_iterator_next (&list_iterator, &itty_bit_string)) {
                printf ("Bit string %zu:\n", i + 1);
                itty_bit_string_iterator_t itty_bit_string_iterator;
                itty_bit_string_iterator_init (itty_bit_string, &itty_bit_string_iterator);
                size_t word;
                size_t j = 0;
                while (itty_bit_string_iterator_next (&itty_bit_string_iterator, &word)) {
                        printf ("  Word %zu: %zu\n", j + 1, word);
                        j++;
                }

                char *binary_representation = itty_bit_string_present (itty_bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
                if (binary_representation) {
                        printf ("  Binary: %s\n", binary_representation);
                        free (binary_representation);
                }

                char *hexadecimal_representation = itty_bit_string_present (itty_bit_string, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
                if (hexadecimal_representation) {
                        printf ("  Hexadecimal: %s\n", hexadecimal_representation);
                        free (hexadecimal_representation);
                }

                i++;
        }

        size_t num_words = 1;
        itty_bit_string_list_t *softmax_list = itty_bit_string_list_popcount_softmax (string_list, num_words);
        printf ("Softmax list:\n");

        itty_bit_string_list_iterator_init (softmax_list, &list_iterator);
        i = 0;
        while (itty_bit_string_list_iterator_next (&list_iterator, &itty_bit_string)) {
                printf ("Softmax bit string %zu:\n", i + 1);
                itty_bit_string_iterator_t itty_bit_string_iterator;
                itty_bit_string_iterator_init (itty_bit_string, &itty_bit_string_iterator);
                size_t word;
                size_t j = 0;
                while (itty_bit_string_iterator_next (&itty_bit_string_iterator, &word)) {
                        printf ("  Word %zu: %zu\n", j + 1, word);
                        j++;
                }

                char *binary_representation = itty_bit_string_present (itty_bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
                if (binary_representation) {
                        printf ("  Binary: %s\n", binary_representation);
                        free (binary_representation);
                }

                char *hexadecimal_representation = itty_bit_string_present (itty_bit_string, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
                if (hexadecimal_representation) {
                        printf ("  Hexadecimal: %s\n", hexadecimal_representation);
                        free (hexadecimal_representation);
                }

                i++;
        }

        printf ("Sorting softmax list by pop count...\n");
        itty_bit_string_list_sort (softmax_list, BIT_STRING_SORT_ORDER_DESCENDING);

        printf ("Sorted Softmax list:\n");
        itty_bit_string_list_iterator_init (softmax_list, &list_iterator);
        i = 0;
        while (itty_bit_string_list_iterator_next (&list_iterator, &itty_bit_string)) {
                printf ("Sorted Softmax bit string %zu:\n", i + 1);
                itty_bit_string_iterator_t itty_bit_string_iterator;
                itty_bit_string_iterator_init (itty_bit_string, &itty_bit_string_iterator);
                size_t word;
                size_t j = 0;
                while (itty_bit_string_iterator_next (&itty_bit_string_iterator, &word)) {
                        printf ("  Word %zu: %zu\n", j + 1, word);
                        j++;
                }

                char *binary_representation = itty_bit_string_present (itty_bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
                if (binary_representation) {
                        printf ("  Binary: %s\n", binary_representation);
                        free (binary_representation);
                }

                char *hexadecimal_representation = itty_bit_string_present (itty_bit_string, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
                if (hexadecimal_representation) {
                        printf ("  Hexadecimal: %s\n", hexadecimal_representation);
                        free (hexadecimal_representation);
                }

                i++;
        }

        itty_bit_string_list_free (softmax_list);
        itty_bit_string_list_free (string_list);

        itty_bit_string_free (a);
        itty_bit_string_free (b);

        itty_bit_string_map_file_t *mapped_file = itty_bit_string_map_file_new ("itty_bit_strings.bin", 2);
        if (mapped_file) {
                printf ("Reading bit strings from file:\n");
                while (itty_bit_string_map_file_next (mapped_file)) {
                        itty_bit_string_list_t *list = itty_bit_string_map_file_peek_at_string_list (mapped_file);
                        itty_bit_string_list_iterator_init (list, &list_iterator);
                        while (itty_bit_string_list_iterator_next (&list_iterator, &itty_bit_string)) {
                                printf ("Bit string from file:\n");
                                itty_bit_string_iterator_t itty_bit_string_iterator;
                                itty_bit_string_iterator_init (itty_bit_string, &itty_bit_string_iterator);
                                size_t word;
                                size_t j = 0;
                                while (itty_bit_string_iterator_next (&itty_bit_string_iterator, &word)) {
                                        printf ("  Word %zu: %zu\n", j + 1, word);
                                        j++;
                                }

                                char *binary_representation = itty_bit_string_present (itty_bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
                                if (binary_representation) {
                                        printf ("  Binary: %s\n", binary_representation);
                                        free (binary_representation);
                                }

                                char *hexadecimal_representation = itty_bit_string_present (itty_bit_string, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
                                if (hexadecimal_representation) {
                                        printf ("  Hexadecimal: %s\n", hexadecimal_representation);
                                        free (hexadecimal_representation);
                                }
                        }
                }
                itty_bit_string_map_file_free (mapped_file);
        }

        return 0;
}
