#pragma once

#include <stddef.h>
#include <stdbool.h>

typedef struct itty_bit_string_t itty_bit_string_t;
typedef struct itty_bit_string_list_t itty_bit_string_list_t;
typedef struct itty_bit_string_list_iterator_t itty_bit_string_list_iterator_t;
typedef struct itty_bit_string_iterator_t itty_bit_string_iterator_t;
typedef struct itty_bit_string_map_file_t itty_bit_string_map_file_t;

typedef enum {
        BIT_STRING_PRESENTATION_FORMAT_BINARY,
        BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY,
        BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL,
        BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL_FOR_DISPLAY
} itty_bit_string_presentation_format_t;

typedef enum {
        BIT_STRING_SORT_ORDER_ASCENDING,
        BIT_STRING_SORT_ORDER_DESCENDING
} itty_bit_string_sort_order_t;

struct itty_bit_string_list_iterator_t {
        itty_bit_string_list_t *list;
        size_t                  current_index;
};

struct itty_bit_string_iterator_t {
        itty_bit_string_t *bit_string;
        size_t             current_index;
};

itty_bit_string_t *itty_bit_string_new (void);
itty_bit_string_t *itty_bit_string_duplicate (itty_bit_string_t *bit_string);

void itty_bit_string_free (itty_bit_string_t *itty_bit_string);

void itty_bit_string_append_word (itty_bit_string_t *itty_bit_string,
                                  size_t             word);

void itty_bit_string_append_zeros (itty_bit_string_t *itty_bit_string,
                                   size_t             count);

itty_bit_string_t *itty_bit_string_exclusive_nor (itty_bit_string_t *a,
                                                  itty_bit_string_t *b);
itty_bit_string_t *itty_bit_string_exclusive_or (itty_bit_string_t *a,
                                                 itty_bit_string_t *b);

itty_bit_string_t *itty_bit_string_combine (itty_bit_string_t *a,
                                            itty_bit_string_t *b);

size_t itty_bit_string_get_pop_count (itty_bit_string_t *itty_bit_string);

size_t itty_bit_string_evaluate_similarity (itty_bit_string_t *a,
                                            itty_bit_string_t *b);

int itty_bit_string_compare (itty_bit_string_t *a,
                             itty_bit_string_t *b);

itty_bit_string_list_t *itty_bit_string_list_new (void);

void itty_bit_string_list_free (itty_bit_string_list_t *list);

void itty_bit_string_list_append (itty_bit_string_list_t *list,
                                  itty_bit_string_t      *itty_bit_string);

size_t itty_bit_string_list_get_length (itty_bit_string_list_t *list);
itty_bit_string_list_t *itty_bit_string_list_exclusive_or (itty_bit_string_list_t *list_a,
                                                           itty_bit_string_list_t *list_b);

itty_bit_string_t *itty_bit_string_list_condense (itty_bit_string_list_t *list);
itty_bit_string_list_t *itty_bit_string_list_transpose (itty_bit_string_list_t *list);

itty_bit_string_t *itty_bit_string_list_fetch (itty_bit_string_list_t *list,
                                               size_t                  index);
size_t itty_bit_string_list_get_max_number_of_words (itty_bit_string_list_t *list);

void itty_bit_string_list_iterator_init (itty_bit_string_list_t          *list,
                                         itty_bit_string_list_iterator_t *iterator);

bool itty_bit_string_list_iterator_next (itty_bit_string_list_iterator_t  *iterator,
                                         itty_bit_string_t               **itty_bit_string);

void itty_bit_string_iterator_init (itty_bit_string_t          *itty_bit_string,
                                    itty_bit_string_iterator_t *iterator);

bool itty_bit_string_iterator_next (itty_bit_string_iterator_t *iterator,
                                    size_t                     *word);

itty_bit_string_list_t *itty_bit_string_split (itty_bit_string_t *itty_bit_string,
                                               size_t             number_of_itty_bit_strings);

itty_bit_string_t *itty_bit_string_concatenate (itty_bit_string_t *a,
                                                itty_bit_string_t *b);

itty_bit_string_t *itty_bit_string_double (itty_bit_string_t *a);

itty_bit_string_t *itty_bit_string_reduce_by_half (itty_bit_string_t *itty_bit_string);

char *itty_bit_string_present (itty_bit_string_t                    *itty_bit_string,
                               itty_bit_string_presentation_format_t format);

itty_bit_string_list_t *itty_bit_string_list_popcount_softmax (itty_bit_string_list_t *list,
                                                               size_t                  num_words);

bool itty_bit_string_list_popcount_argmax (itty_bit_string_list_t *list,
                                           size_t                  num_words,
                                           size_t                 *index);

void itty_bit_string_list_sort (itty_bit_string_list_t      *list,
                                itty_bit_string_sort_order_t order);

itty_bit_string_map_file_t *itty_bit_string_map_file_new (const char *file_name);

void itty_bit_string_map_file_free (itty_bit_string_map_file_t *mapped_file);

itty_bit_string_t *itty_bit_string_map_file_next (itty_bit_string_map_file_t *mapped_file,
                                                  size_t                      number_of_words);
