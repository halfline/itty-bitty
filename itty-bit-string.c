#define _GNU_SOURCE
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
} bit_string_t;

typedef struct {
        bit_string_t **bit_strings;
        size_t         count;
} bit_string_list_t;

typedef struct {
        bit_string_list_t *list;
        size_t             current_index;
} bit_string_list_iterator_t;

typedef struct {
        bit_string_t *bit_string;
        size_t        current_index;
} bit_string_iterator_t;

typedef enum {
        BIT_STRING_PRESENTATION_FORMAT_BINARY,
        BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL
} bit_string_presentation_format_t;

typedef enum {
        BIT_STRING_SORT_ORDER_ASCENDING,
        BIT_STRING_SORT_ORDER_DESCENDING
} bit_string_sort_order_t;

typedef struct {
        int         fd;
        size_t      file_size;
        void       *mapped_data;
        size_t      word_count_per_bit_string;
        size_t      current_index;
        bit_string_list_t *bit_string_list;
} bit_string_map_file_t;

bit_string_t *
bit_string_new (void)
{
        bit_string_t *bit_string = malloc (sizeof (bit_string_t));
        bit_string->words = NULL;
        bit_string->number_of_words = 0;
        bit_string->pop_count = 0;
        bit_string->pop_count_computed = false;
        return bit_string;
}

void
bit_string_free (bit_string_t *bit_string)
{
        if (!bit_string) {
                return;
        }
        free (bit_string->words);
        free (bit_string);
}

void
bit_string_append_word (bit_string_t *bit_string,
                        size_t        word)
{
        bit_string->words = realloc (bit_string->words,
                                    (bit_string->number_of_words + 1) * WORD_SIZE_IN_BYTES);
        bit_string->words[bit_string->number_of_words] = word;
        bit_string->number_of_words++;
        bit_string->pop_count_computed = false;  // Invalidate cached pop count
}

void
bit_string_append_zeros (bit_string_t *bit_string,
                         size_t        count)
{
        for (size_t i = 0; i < count; i++) {
                bit_string_append_word (bit_string, 0);
        }
}

bit_string_t *
bit_string_exclusive_nor (bit_string_t *a,
                          bit_string_t *b)
{
        size_t max_length = a->number_of_words > b->number_of_words ? a->number_of_words : b->number_of_words;
        size_t a_padding = max_length - a->number_of_words;
        size_t b_padding = max_length - b->number_of_words;

        if (a_padding > 0) {
                bit_string_append_zeros (a, a_padding);
        }

        if (b_padding > 0) {
                bit_string_append_zeros (b, b_padding);
        }

        bit_string_t *result = bit_string_new ();
        result->number_of_words = max_length;
        result->words = malloc (result->number_of_words * WORD_SIZE_IN_BYTES);

        for (size_t i = 0; i < max_length; i++) {
                result->words[i] = ~(a->words[i] ^ b->words[i]);
        }

        return result;
}

size_t
bit_string_get_pop_count (bit_string_t *bit_string)
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
bit_string_evaluate_similarity (bit_string_t *a,
                                bit_string_t *b)
{
        bit_string_t *exclusive_nor_result = bit_string_exclusive_nor (a, b);
        size_t similarity = bit_string_get_pop_count (exclusive_nor_result);
        bit_string_free (exclusive_nor_result);
        return similarity;
}

int
bit_string_compare (bit_string_t *a,
                    bit_string_t *b)
{
        return (int) (bit_string_get_pop_count (a) - bit_string_get_pop_count (b));
}

bit_string_list_t *
bit_string_list_new (void)
{
        bit_string_list_t *list = malloc (sizeof (bit_string_list_t));
        list->bit_strings = NULL;
        list->count = 0;
        return list;
}

void
bit_string_list_free (bit_string_list_t *list)
{
        if (!list) {
                return;
        }
        for (size_t i = 0; i < list->count; i++) {
                bit_string_free (list->bit_strings[i]);
        }
        free (list->bit_strings);
        free (list);
}

void
bit_string_list_append (bit_string_list_t *list,
                        bit_string_t      *bit_string)
{
        list->bit_strings = realloc (list->bit_strings,
                                    (list->count + 1) * sizeof (bit_string_t *));
        list->bit_strings[list->count] = bit_string;
        list->count++;
}

void
bit_string_list_iterator_init (bit_string_list_t           *list,
                               bit_string_list_iterator_t *iterator)
{
        iterator->list = list;
        iterator->current_index = 0;
}

bool
bit_string_list_iterator_next (bit_string_list_iterator_t *iterator,
                               bit_string_t              **bit_string)
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
bit_string_iterator_init (bit_string_t         *bit_string,
                          bit_string_iterator_t *iterator)
{
        iterator->bit_string = bit_string;
        iterator->current_index = 0;
}

bool
bit_string_iterator_next (bit_string_iterator_t *iterator,
                          size_t               *word)
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

bit_string_list_t *
bit_string_split (bit_string_t *bit_string,
                  size_t       number_of_bit_strings)
{
        if (number_of_bit_strings == 0) {
                return NULL;
        }

        size_t total_bits = bit_string->number_of_words * WORD_SIZE_IN_BITS;
        size_t bits_per_split = (total_bits + number_of_bit_strings - 1) / number_of_bit_strings;
        size_t words_per_split = (bits_per_split + WORD_SIZE_IN_BITS - 1) / WORD_SIZE_IN_BITS;

        bit_string_list_t *split_list = bit_string_list_new ();
        for (size_t i = 0; i < number_of_bit_strings; i++) {
                bit_string_t *split = bit_string_new ();
                for (size_t j = 0; j < words_per_split; j++) {
                        size_t word_index = i * words_per_split + j;
                        size_t word = (word_index < bit_string->number_of_words) ? bit_string->words[word_index] : 0;
                        bit_string_append_word (split, word);
                }
                bit_string_list_append (split_list, split);
        }

        return split_list;
}

char *
bit_string_present (bit_string_t                    *bit_string,
                    bit_string_presentation_format_t format)
{
        size_t total_bits = bit_string->number_of_words * WORD_SIZE_IN_BITS;
        size_t buffer_size;

        if (format == BIT_STRING_PRESENTATION_FORMAT_BINARY) {
                buffer_size = total_bits + 1;
        } else {
                buffer_size = bit_string->number_of_words * (WORD_SIZE_IN_BYTES * 2 + 1);
        }

        char *bit_string_representation = malloc (buffer_size);
        if (!bit_string_representation) {
                return NULL;
        }

        if (format == BIT_STRING_PRESENTATION_FORMAT_BINARY) {
                bit_string_representation[total_bits] = '\0';
                size_t bit_index = 0;
                for (size_t i = 0; i < bit_string->number_of_words; i++) {
                        size_t word = bit_string->words[i];
                        for (size_t j = WORD_SIZE_IN_BITS; j > 0; j--) {
                                bit_string_representation[bit_index++] = (word & (1UL << (j - 1))) ? '1' : '0';
                        }
                }
        } else {
                size_t buffer_index = 0;
                for (size_t i = 0; i < bit_string->number_of_words; i++) {
                        int written = snprintf (&bit_string_representation[buffer_index], buffer_size - buffer_index, "%016lx ", bit_string->words[i]);
                        buffer_index += written;
                }
                bit_string_representation[buffer_index - 1] = '\0';
        }

        return bit_string_representation;
}

bit_string_list_t *
bit_string_list_popcount_softmax (bit_string_list_t *list,
                                  size_t             num_words)
{
        bit_string_list_t *softmax_list = bit_string_list_new ();

        size_t total_popcount = 0;
        bit_string_list_iterator_t iterator;
        bit_string_list_iterator_init (list, &iterator);
        bit_string_t *bit_string;

        while (bit_string_list_iterator_next (&iterator, &bit_string)) {
                total_popcount += bit_string_get_pop_count (bit_string);
        }

        size_t cumulative_ones = 0;
        bit_string_list_iterator_init (list, &iterator);
        while (bit_string_list_iterator_next (&iterator, &bit_string)) {
                bit_string_t *new_bit_string = bit_string_new ();
                size_t num_ones = (bit_string_get_pop_count (bit_string) * WORD_SIZE_IN_BITS + total_popcount - 1) / total_popcount;
                cumulative_ones += num_ones;

                size_t remaining_ones = num_ones;
                for (size_t i = 0; i < num_words; i++) {
                        size_t word = 0;
                        if (remaining_ones > 0) {
                                size_t ones_in_word = remaining_ones > WORD_SIZE_IN_BITS ? WORD_SIZE_IN_BITS : remaining_ones;
                                word = (1UL << ones_in_word) - 1;
                                remaining_ones -= ones_in_word;
                        }
                        bit_string_append_word (new_bit_string, word);
                }

                bit_string_list_append (softmax_list, new_bit_string);
        }

        size_t total_bits = WORD_SIZE_IN_BITS * num_words;
        if (cumulative_ones != total_bits && softmax_list->count > 0) {
                size_t excess_ones = cumulative_ones - total_bits;
                size_t last_word_index = num_words - 1;
                softmax_list->bit_strings[softmax_list->count - 1]->words[last_word_index] >>= excess_ones;
        }

        return softmax_list;
}

int
bit_string_compare_qsort (const void *a,
                          const void *b,
                          void       *order)
{
        bit_string_t *bit_string_a = *(bit_string_t **) a;
        bit_string_t *bit_string_b = *(bit_string_t **) b;
        bit_string_sort_order_t sort_order = *(bit_string_sort_order_t *) order;
        return (sort_order == BIT_STRING_SORT_ORDER_ASCENDING) ? bit_string_compare (bit_string_a, bit_string_b)
                                                               : bit_string_compare (bit_string_b, bit_string_a);
}

void
bit_string_list_sort (bit_string_list_t      *list,
                      bit_string_sort_order_t order)
{
        qsort_r (list->bit_strings, list->count, sizeof (bit_string_t *), bit_string_compare_qsort, &order);
}

bit_string_map_file_t *
bit_string_map_file_new (const char *file_name,
                         size_t     word_count_per_bit_string)
{
        bit_string_map_file_t *mapped_file = malloc (sizeof (bit_string_map_file_t));

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

        mapped_file->word_count_per_bit_string = word_count_per_bit_string;
        mapped_file->current_index = 0;
        mapped_file->bit_string_list = bit_string_list_new ();

        return mapped_file;
}

void
bit_string_map_file_free (bit_string_map_file_t *mapped_file)
{
        if (!mapped_file) {
                return;
        }

        bit_string_list_iterator_t iterator;
        bit_string_list_iterator_init (mapped_file->bit_string_list, &iterator);
        bit_string_t *bit_string;
        while (bit_string_list_iterator_next (&iterator, &bit_string)) {
                bit_string->words = NULL;  // Clear the words pointer
        }

        bit_string_list_free (mapped_file->bit_string_list);
        munmap (mapped_file->mapped_data, mapped_file->file_size);
        close (mapped_file->fd);
        free (mapped_file);
}

bit_string_t *
bit_string_map_file_get_next (bit_string_map_file_t *mapped_file)
{
        size_t total_words = mapped_file->file_size / WORD_SIZE_IN_BYTES;
        if (mapped_file->current_index >= total_words) {
                return NULL;
        }

        bit_string_t *bit_string = bit_string_new ();
        bit_string->words = (size_t *) (mapped_file->mapped_data) + mapped_file->current_index;
        bit_string->number_of_words = mapped_file->word_count_per_bit_string;
        bit_string->pop_count_computed = false;

        mapped_file->current_index += bit_string->number_of_words;
        return bit_string;
}

bool
bit_string_map_file_next (bit_string_map_file_t *mapped_file)
{
        bit_string_t *bit_string = bit_string_map_file_get_next (mapped_file);
        if (!bit_string) {
                return false;
        }

        bit_string_list_append (mapped_file->bit_string_list, bit_string);
        return true;
}

bit_string_list_t *
bit_string_map_file_peek_at_string_list (bit_string_map_file_t *mapped_file)
{
        return mapped_file->bit_string_list;
}

int
main (void)
{
        bit_string_t *a = bit_string_new ();
        bit_string_t *b = bit_string_new ();

        bit_string_append_word (a, 0b10101010);
        bit_string_append_word (a, 0b11110000);
        bit_string_append_word (a, 0b11111111);
        bit_string_append_word (a, 0b00000000);

        bit_string_append_word (b, 0b11001100);

        printf ("Bit string a:\n");
        char *a_binary_representation = bit_string_present (a, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        if (a_binary_representation) {
                printf ("  Binary: %s\n", a_binary_representation);
                free (a_binary_representation);
        }
        char *a_hexadecimal_representation = bit_string_present (a, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
        if (a_hexadecimal_representation) {
                printf ("  Hexadecimal: %s\n", a_hexadecimal_representation);
                free (a_hexadecimal_representation);
        }

        printf ("Bit string b:\n");
        char *b_binary_representation = bit_string_present (b, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        if (b_binary_representation) {
                printf ("  Binary: %s\n", b_binary_representation);
                free (b_binary_representation);
        }
        char *b_hexadecimal_representation = bit_string_present (b, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
        if (b_hexadecimal_representation) {
                printf ("  Hexadecimal: %s\n", b_hexadecimal_representation);
                free (b_hexadecimal_representation);
        }

        size_t similarity = bit_string_evaluate_similarity (a, b);
        printf ("Rate of similarity: %zu\n", similarity);

        int comparison = bit_string_compare (a, b);
        printf ("Pop count comparison (a - b): %d\n", comparison);

        size_t number_of_bit_strings = 3;
        bit_string_list_t *string_list = bit_string_split (a, number_of_bit_strings);
        printf ("Splits:\n");

        bit_string_list_iterator_t list_iterator;
        bit_string_list_iterator_init (string_list, &list_iterator);

        bit_string_t *bit_string;
        size_t i = 0;
        while (bit_string_list_iterator_next (&list_iterator, &bit_string)) {
                printf ("Bit string %zu:\n", i + 1);
                bit_string_iterator_t bit_string_iterator;
                bit_string_iterator_init (bit_string, &bit_string_iterator);
                size_t word;
                size_t j = 0;
                while (bit_string_iterator_next (&bit_string_iterator, &word)) {
                        printf ("  Word %zu: %zu\n", j + 1, word);
                        j++;
                }

                char *binary_representation = bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
                if (binary_representation) {
                        printf ("  Binary: %s\n", binary_representation);
                        free (binary_representation);
                }

                char *hexadecimal_representation = bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
                if (hexadecimal_representation) {
                        printf ("  Hexadecimal: %s\n", hexadecimal_representation);
                        free (hexadecimal_representation);
                }

                i++;
        }

        size_t num_words = 1;
        bit_string_list_t *softmax_list = bit_string_list_popcount_softmax (string_list, num_words);
        printf ("Softmax list:\n");

        bit_string_list_iterator_init (softmax_list, &list_iterator);
        i = 0;
        while (bit_string_list_iterator_next (&list_iterator, &bit_string)) {
                printf ("Softmax bit string %zu:\n", i + 1);
                bit_string_iterator_t bit_string_iterator;
                bit_string_iterator_init (bit_string, &bit_string_iterator);
                size_t word;
                size_t j = 0;
                while (bit_string_iterator_next (&bit_string_iterator, &word)) {
                        printf ("  Word %zu: %zu\n", j + 1, word);
                        j++;
                }

                char *binary_representation = bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
                if (binary_representation) {
                        printf ("  Binary: %s\n", binary_representation);
                        free (binary_representation);
                }

                char *hexadecimal_representation = bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
                if (hexadecimal_representation) {
                        printf ("  Hexadecimal: %s\n", hexadecimal_representation);
                        free (hexadecimal_representation);
                }

                i++;
        }

        printf ("Sorting softmax list by pop count...\n");
        bit_string_list_sort (softmax_list, BIT_STRING_SORT_ORDER_DESCENDING);

        printf ("Sorted Softmax list:\n");
        bit_string_list_iterator_init (softmax_list, &list_iterator);
        i = 0;
        while (bit_string_list_iterator_next (&list_iterator, &bit_string)) {
                printf ("Sorted Softmax bit string %zu:\n", i + 1);
                bit_string_iterator_t bit_string_iterator;
                bit_string_iterator_init (bit_string, &bit_string_iterator);
                size_t word;
                size_t j = 0;
                while (bit_string_iterator_next (&bit_string_iterator, &word)) {
                        printf ("  Word %zu: %zu\n", j + 1, word);
                        j++;
                }

                char *binary_representation = bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
                if (binary_representation) {
                        printf ("  Binary: %s\n", binary_representation);
                        free (binary_representation);
                }

                char *hexadecimal_representation = bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
                if (hexadecimal_representation) {
                        printf ("  Hexadecimal: %s\n", hexadecimal_representation);
                        free (hexadecimal_representation);
                }

                i++;
        }

        bit_string_list_free (softmax_list);
        bit_string_list_free (string_list);

        bit_string_free (a);
        bit_string_free (b);

        bit_string_map_file_t *mapped_file = bit_string_map_file_new ("bit_strings.bin", 2);
        if (mapped_file) {
                printf ("Reading bit strings from file:\n");
                while (bit_string_map_file_next (mapped_file)) {
                        bit_string_list_t *list = bit_string_map_file_peek_at_string_list (mapped_file);
                        bit_string_list_iterator_init (list, &list_iterator);
                        while (bit_string_list_iterator_next (&list_iterator, &bit_string)) {
                                printf ("Bit string from file:\n");
                                bit_string_iterator_t bit_string_iterator;
                                bit_string_iterator_init (bit_string, &bit_string_iterator);
                                size_t word;
                                size_t j = 0;
                                while (bit_string_iterator_next (&bit_string_iterator, &word)) {
                                        printf ("  Word %zu: %zu\n", j + 1, word);
                                        j++;
                                }

                                char *binary_representation = bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
                                if (binary_representation) {
                                        printf ("  Binary: %s\n", binary_representation);
                                        free (binary_representation);
                                }

                                char *hexadecimal_representation = bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
                                if (hexadecimal_representation) {
                                        printf ("  Hexadecimal: %s\n", hexadecimal_representation);
                                        free (hexadecimal_representation);
                                }
                        }
                }
                bit_string_map_file_free (mapped_file);
        }

        return 0;
}

