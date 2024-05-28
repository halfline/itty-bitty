#pragma once

#include <limits.h>
#include <stddef.h>
#include <stdbool.h>

#define ITTY_BIT_STRING_WORD_SIZE_IN_BYTES (sizeof (size_t))
#define ITTY_BIT_STRING_WORD_SIZE_IN_BITS (sizeof (size_t) * CHAR_BIT)

typedef struct itty_bit_string_t itty_bit_string_t;
typedef struct itty_bit_string_list_t itty_bit_string_list_t;
typedef struct itty_bit_string_iterator_t itty_bit_string_iterator_t;

typedef enum {
        ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY,
        ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY,
        ITTY_BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL,
        ITTY_BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL_FOR_DISPLAY
} itty_bit_string_presentation_format_t;

typedef enum {
        ITTY_BIT_STRING_SORT_ORDER_ASCENDING,
        ITTY_BIT_STRING_SORT_ORDER_DESCENDING
} itty_bit_string_sort_order_t;

struct itty_bit_string_iterator_t {
        itty_bit_string_t *bit_string;
        size_t             current_index;
};

itty_bit_string_t *itty_bit_string_new (void);
itty_bit_string_t *itty_bit_string_duplicate (itty_bit_string_t *bit_string);

void itty_bit_string_free (itty_bit_string_t *bit_string);

void itty_bit_string_append_word (itty_bit_string_t *bit_string,
                                  size_t             word);

void itty_bit_string_append_zeros (itty_bit_string_t *bit_string,
                                   size_t             count);

itty_bit_string_t *itty_bit_string_exclusive_nor (itty_bit_string_t *a,
                                                  itty_bit_string_t *b);
itty_bit_string_t *itty_bit_string_exclusive_or (itty_bit_string_t *a,
                                                 itty_bit_string_t *b);

itty_bit_string_t *itty_bit_string_combine (itty_bit_string_t *a,
                                            itty_bit_string_t *b);
itty_bit_string_t *itty_bit_string_mask (itty_bit_string_t *a,
                                         itty_bit_string_t *b);

size_t itty_bit_string_get_pop_count (itty_bit_string_t *bit_string);
size_t itty_bit_string_get_length (itty_bit_string_t *bit_string);

size_t itty_bit_string_evaluate_similarity (itty_bit_string_t *a,
                                            itty_bit_string_t *b);

int itty_bit_string_compare (itty_bit_string_t *a,
                             itty_bit_string_t *b);

int itty_bit_string_compare_by_pop_count (itty_bit_string_t *a,
                                          itty_bit_string_t *b);

void *itty_bit_string_get_words (itty_bit_string_t *bit_string);
size_t itty_bit_string_get_number_of_words (itty_bit_string_t *bit_string);

itty_bit_string_list_t *itty_bit_string_split (itty_bit_string_t *bit_string,
                                               size_t             number_of_bit_strings);

itty_bit_string_t *itty_bit_string_concatenate (itty_bit_string_t *a,
                                                itty_bit_string_t *b);
itty_bit_string_t *itty_bit_string_double (itty_bit_string_t *bit_string);
itty_bit_string_t *itty_bit_string_reduce_by_half (itty_bit_string_t *bit_string);
char *itty_bit_string_present (itty_bit_string_t                     *bit_string,
                               itty_bit_string_presentation_format_t  format);
void itty_bit_string_iterator_init (itty_bit_string_t          *bit_string,
                                    itty_bit_string_iterator_t *iterator);
void itty_bit_string_iterator_init_at_word_offset (itty_bit_string_t          *bit_string,
                                                   itty_bit_string_iterator_t *iterator,
                                                   size_t                     word_offset);

bool itty_bit_string_iterator_next (itty_bit_string_iterator_t *iterator,
                                    size_t                     *word);
