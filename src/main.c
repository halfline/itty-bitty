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

        itty_bit_string_t *bit_string;
        size_t i = 0;
        while (itty_bit_string_list_iterator_next (&list_iterator, &bit_string)) {
                printf ("Bit string %zu:\n", i + 1);
                itty_bit_string_iterator_t bit_string_iterator;
                itty_bit_string_iterator_init (bit_string, &bit_string_iterator);
                size_t word;
                size_t j = 0;
                while (itty_bit_string_iterator_next (&bit_string_iterator, &word)) {
                        printf ("  Word %zu: %zu\n", j + 1, word);
                        j++;
                }

                char *binary_representation = itty_bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
                if (binary_representation) {
                        printf ("  Binary: %s\n", binary_representation);
                        free (binary_representation);
                }

                char *hexadecimal_representation = itty_bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
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
        while (itty_bit_string_list_iterator_next (&list_iterator, &bit_string)) {
                printf ("Softmax bit string %zu:\n", i + 1);
                itty_bit_string_iterator_t bit_string_iterator;
                itty_bit_string_iterator_init (bit_string, &bit_string_iterator);
                size_t word;
                size_t j = 0;
                while (itty_bit_string_iterator_next (&bit_string_iterator, &word)) {
                        printf ("  Word %zu: %zu\n", j + 1, word);
                        j++;
                }

                char *binary_representation = itty_bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
                if (binary_representation) {
                        printf ("  Binary: %s\n", binary_representation);
                        free (binary_representation);
                }

                char *hexadecimal_representation = itty_bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
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
        while (itty_bit_string_list_iterator_next (&list_iterator, &bit_string)) {
                printf ("Sorted Softmax bit string %zu:\n", i + 1);
                itty_bit_string_iterator_t bit_string_iterator;
                itty_bit_string_iterator_init (bit_string, &bit_string_iterator);
                size_t word;
                size_t j = 0;
                while (itty_bit_string_iterator_next (&bit_string_iterator, &word)) {
                        printf ("  Word %zu: %zu\n", j + 1, word);
                        j++;
                }

                char *binary_representation = itty_bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
                if (binary_representation) {
                        printf ("  Binary: %s\n", binary_representation);
                        free (binary_representation);
                }

                char *hexadecimal_representation = itty_bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
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
                        while (itty_bit_string_list_iterator_next (&list_iterator, &bit_string)) {
                                printf ("Bit string from file:\n");
                                itty_bit_string_iterator_t bit_string_iterator;
                                itty_bit_string_iterator_init (bit_string, &bit_string_iterator);
                                size_t word;
                                size_t j = 0;
                                while (itty_bit_string_iterator_next (&bit_string_iterator, &word)) {
                                        printf ("  Word %zu: %zu\n", j + 1, word);
                                        j++;
                                }

                                char *binary_representation = itty_bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
                                if (binary_representation) {
                                        printf ("  Binary: %s\n", binary_representation);
                                        free (binary_representation);
                                }

                                char *hexadecimal_representation = itty_bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
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
