#pragma once

#include "itty-bit-string.h"
#include <stddef.h>
#include <stdbool.h>

typedef struct itty_bit_string_list_t itty_bit_string_list_t;
typedef struct itty_bit_string_list_iterator_t itty_bit_string_list_iterator_t;

struct itty_bit_string_list_iterator_t {
        itty_bit_string_list_t *list;
        size_t                  current_index;
};

itty_bit_string_list_t *itty_bit_string_list_new (void);

void itty_bit_string_list_free (itty_bit_string_list_t *list);

void itty_bit_string_list_append (itty_bit_string_list_t *list,
                                  itty_bit_string_t      *bit_string);

size_t itty_bit_string_list_get_length (itty_bit_string_list_t *list);
size_t itty_bit_string_list_get_bit_length (itty_bit_string_list_t *list);

itty_bit_string_list_t *itty_bit_string_list_exclusive_or (itty_bit_string_list_t *list_a,
                                                           itty_bit_string_list_t *list_b);

itty_bit_string_t *itty_bit_string_list_fetch (itty_bit_string_list_t *list,
                                               size_t                  index);

itty_bit_string_t *itty_bit_string_list_condense (itty_bit_string_list_t *list);

itty_bit_string_list_t *itty_bit_string_list_transpose (itty_bit_string_list_t *list);

size_t itty_bit_string_list_get_max_number_of_words (itty_bit_string_list_t *list);

char *itty_bit_string_list_present (itty_bit_string_list_t                *bit_string_list,
                                    itty_bit_string_presentation_format_t  format);
itty_bit_string_list_t *itty_bit_string_list_popcount_softmax (itty_bit_string_list_t *list,
                                                               size_t                  num_words);
bool itty_bit_string_list_popcount_argmax (itty_bit_string_list_t *list,
                                           size_t                  num_words,
                                           size_t                 *index);
void itty_bit_string_list_sort (itty_bit_string_list_t      *list,
                                itty_bit_string_sort_order_t order);

void itty_bit_string_list_iterator_init (itty_bit_string_list_t          *list,
                                         itty_bit_string_list_iterator_t *iterator);

void itty_bit_string_list_iterator_init_at_index (itty_bit_string_list_t          *list,
                                                  itty_bit_string_list_iterator_t *iterator,
                                                  size_t                          index);

bool itty_bit_string_list_iterator_next (itty_bit_string_list_iterator_t  *iterator,
                                         itty_bit_string_t               **bit_string);

