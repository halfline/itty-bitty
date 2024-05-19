#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "itty-bit-string.c"

void
test_itty_bit_string_new (void)
{
        itty_bit_string_t *bit_string = itty_bit_string_new ();
        assert (bit_string != NULL);
        assert (bit_string->words == NULL);
        assert (bit_string->number_of_words == 0);
        assert (bit_string->pop_count == 0);
        assert (bit_string->pop_count_computed == false);
        itty_bit_string_free (bit_string);
}

void
test_itty_bit_string_duplicate (void)
{
        itty_bit_string_t *original = itty_bit_string_new ();
        itty_bit_string_append_word (original, 1);
        itty_bit_string_t *duplicate = itty_bit_string_duplicate (original);
        assert (duplicate != NULL);
        char *original_representation = itty_bit_string_present (original, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char *duplicate_representation = itty_bit_string_present (duplicate, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        assert (strcmp (original_representation, duplicate_representation) == 0);
        free (original_representation);
        free (duplicate_representation);
        itty_bit_string_free (original);
        itty_bit_string_free (duplicate);
}

void
test_itty_bit_string_append_word (void)
{
        itty_bit_string_t *bit_string = itty_bit_string_new ();
        itty_bit_string_append_word (bit_string, 1);
        assert (bit_string->number_of_words == 1);
        char *representation = itty_bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        assert (strcmp (representation, "0000000000000000000000000000000000000000000000000000000000000001") == 0);
        free (representation);
        itty_bit_string_free (bit_string);
}

void
test_itty_bit_string_append_zeros (void)
{
        itty_bit_string_t *bit_string = itty_bit_string_new ();
        itty_bit_string_append_zeros (bit_string, 3);
        assert (bit_string->number_of_words == 3);
        char *representation = itty_bit_string_present (bit_string, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char expected[] = "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000";
        assert (strcmp (representation, expected) == 0);
        free (representation);
        itty_bit_string_free (bit_string);
}

void
test_itty_bit_string_exclusive_nor (void)
{
        itty_bit_string_t *a = itty_bit_string_new ();
        itty_bit_string_t *b = itty_bit_string_new ();
        itty_bit_string_append_word (a, 0b1100);
        itty_bit_string_append_word (b, 0b1010);
        itty_bit_string_t *result = itty_bit_string_exclusive_nor (a, b);
        assert (result != NULL);
        char *representation = itty_bit_string_present (result, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char expected[] = "1111111111111111111111111111111111111111111111111111111111111001";
        assert (strcmp (representation, expected) == 0);
        free (representation);
        itty_bit_string_free (a);
        itty_bit_string_free (b);
        itty_bit_string_free (result);
}

void
test_itty_bit_string_exclusive_or (void)
{
        itty_bit_string_t *a = itty_bit_string_new ();
        itty_bit_string_t *b = itty_bit_string_new ();
        itty_bit_string_append_word (a, 0b1100);
        itty_bit_string_append_word (b, 0b1010);
        itty_bit_string_t *result = itty_bit_string_exclusive_or (a, b);
        assert (result != NULL);
        char *representation = itty_bit_string_present (result, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char expected[] = "0000000000000000000000000000000000000000000000000000000000000110";
        assert (strcmp (representation, expected) == 0);
        free (representation);
        itty_bit_string_free (a);
        itty_bit_string_free (b);
        itty_bit_string_free (result);
}

void
test_itty_bit_string_combine (void)
{
        itty_bit_string_t *a = itty_bit_string_new ();
        itty_bit_string_t *b = itty_bit_string_new ();
        itty_bit_string_append_word (a, 0b1100);
        itty_bit_string_append_word (b, 0b1010);
        itty_bit_string_t *result = itty_bit_string_combine (a, b);
        assert (result != NULL);
        char *representation = itty_bit_string_present (result, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char expected[] = "0000000000000000000000000000000000000000000000000000000000001110";
        assert (strcmp (representation, expected) == 0);
        free (representation);
        itty_bit_string_free (a);
        itty_bit_string_free (b);
        itty_bit_string_free (result);
}

void
test_itty_bit_string_get_pop_count (void)
{
        itty_bit_string_t *bit_string = itty_bit_string_new ();
        itty_bit_string_append_word (bit_string, 0b1101);
        size_t pop_count = itty_bit_string_get_pop_count (bit_string);
        assert (pop_count == 3);
        itty_bit_string_free (bit_string);
}

void
test_itty_bit_string_evaluate_similarity (void)
{
        itty_bit_string_t *a = itty_bit_string_new ();
        itty_bit_string_t *b = itty_bit_string_new ();
        itty_bit_string_append_word (a, 0b0000000000000000000000000000000000000000000000000000000000001100);
        itty_bit_string_append_word (b, 0b0000000000000000000000000000000000000000000000000000000000001101);
        size_t similarity = itty_bit_string_evaluate_similarity (a, b);
        assert (similarity == 63);
        itty_bit_string_free (a);
        itty_bit_string_free (b);
}

void
test_itty_bit_string_compare (void)
{
        itty_bit_string_t *a = itty_bit_string_new ();
        itty_bit_string_t *b = itty_bit_string_new ();
        itty_bit_string_append_word (a, 0b1100);
        itty_bit_string_append_word (b, 0b1001);
        int cmp = itty_bit_string_compare (a, b);
        assert (cmp == 0);
        itty_bit_string_append_word (a, 0b1111);
        cmp = itty_bit_string_compare (a, b);
        assert (cmp > 0);
        itty_bit_string_free (a);
        itty_bit_string_free (b);
}

void
test_itty_bit_string_double (void)
{
        itty_bit_string_t *bit_string = itty_bit_string_new ();
        itty_bit_string_append_word (bit_string, 0b1100);
        itty_bit_string_t *doubled = itty_bit_string_double (bit_string);
        assert (doubled != NULL);
        char *representation = itty_bit_string_present (doubled, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char expected[] = "0000000000000000000000000000000000000000000000000000000000001100"
                          "0000000000000000000000000000000000000000000000000000000000001100";
        assert (strcmp (representation, expected) == 0);
        free (representation);
        itty_bit_string_free (bit_string);
        itty_bit_string_free (doubled);
}

void
test_itty_bit_string_reduce_by_half (void)
{
        itty_bit_string_t *bit_string = itty_bit_string_new ();
        itty_bit_string_append_word (bit_string, 0b1100);
        itty_bit_string_append_word (bit_string, 0b1010);
        itty_bit_string_t *reduced = itty_bit_string_reduce_by_half (bit_string);
        assert (reduced != NULL);
        char *representation = itty_bit_string_present (reduced, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char expected[] = "1111111111111111111111111111111111111111111111111111111111110011";
        assert (strcmp (representation, expected) == 0);
        free (representation);
        itty_bit_string_free (bit_string);
        itty_bit_string_free (reduced);
}

void
test_itty_bit_string_list_new (void)
{
        itty_bit_string_list_t *list = itty_bit_string_list_new ();
        assert (list != NULL);
        assert (list->bit_strings == NULL);
        assert (list->count == 0);
        assert (list->max_number_of_words == 0);
        itty_bit_string_list_free (list);
}

void
test_itty_bit_string_list_append (void)
{
        itty_bit_string_list_t *list = itty_bit_string_list_new ();
        itty_bit_string_t *bit_string = itty_bit_string_new ();
        itty_bit_string_append_word (bit_string, 0b1100);
        itty_bit_string_list_append (list, bit_string);
        assert (list->count == 1);
        assert (list->bit_strings[0] == bit_string);
        itty_bit_string_list_free (list); // itty_bit_string_list_free will also free bit_string
}

void
test_itty_bit_string_list_exclusive_or (void)
{
        itty_bit_string_list_t *list_a = itty_bit_string_list_new ();
        itty_bit_string_list_t *list_b = itty_bit_string_list_new ();
        itty_bit_string_t *bit_string_a = itty_bit_string_new ();
        itty_bit_string_t *bit_string_b = itty_bit_string_new ();
        itty_bit_string_append_word (bit_string_a, 0b1100);
        itty_bit_string_append_word (bit_string_b, 0b1010);
        itty_bit_string_list_append (list_a, bit_string_a);
        itty_bit_string_list_append (list_b, bit_string_b);
        itty_bit_string_list_t *result_list = itty_bit_string_list_exclusive_or (list_a, list_b);
        assert (result_list != NULL);
        assert (result_list->count == 1);
        char *representation = itty_bit_string_present (result_list->bit_strings[0], BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char expected[] = "0000000000000000000000000000000000000000000000000000000000000110";
        assert (strcmp (representation, expected) == 0);
        free (representation);
        itty_bit_string_list_free (list_a);
        itty_bit_string_list_free (list_b);
        itty_bit_string_list_free (result_list);
}

void
test_itty_bit_string_list_condense (void)
{
        itty_bit_string_list_t *list = itty_bit_string_list_new ();
        itty_bit_string_t *bit_string_1 = itty_bit_string_new ();
        itty_bit_string_t *bit_string_2 = itty_bit_string_new ();
        itty_bit_string_t *bit_string_3 = itty_bit_string_new ();
        itty_bit_string_append_word (bit_string_1, 0b1100);
        itty_bit_string_append_word (bit_string_2, 0b1010);
        itty_bit_string_append_word (bit_string_3, 0b1111);
        itty_bit_string_list_append (list, bit_string_1);
        itty_bit_string_list_append (list, bit_string_2);
        itty_bit_string_list_append (list, bit_string_3);
        itty_bit_string_t *condensed = itty_bit_string_list_condense (list);
        assert (condensed != NULL);
        char *representation = itty_bit_string_present (condensed, BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char expected[] = "0000000000000000000000000000000000000000000000000000000000001110";
        assert (strcmp (representation, expected) == 0);
        free (representation);
        itty_bit_string_free (condensed);
        itty_bit_string_list_free (list);
}

void
test_itty_bit_string_list_sort (void)
{
        itty_bit_string_list_t *list = itty_bit_string_list_new ();
        itty_bit_string_t *bit_string_1 = itty_bit_string_new ();
        itty_bit_string_t *bit_string_2 = itty_bit_string_new ();
        itty_bit_string_t *bit_string_3 = itty_bit_string_new ();
        itty_bit_string_append_word (bit_string_1, 0b1100);
        itty_bit_string_append_word (bit_string_2, 0b1000);
        itty_bit_string_append_word (bit_string_3, 0b1111);
        itty_bit_string_list_append (list, bit_string_1);
        itty_bit_string_list_append (list, bit_string_2);
        itty_bit_string_list_append (list, bit_string_3);

        itty_bit_string_list_sort (list, BIT_STRING_SORT_ORDER_ASCENDING);

        assert (list->bit_strings[0] == bit_string_2);
        assert (list->bit_strings[1] == bit_string_1);
        assert (list->bit_strings[2] == bit_string_3);
        itty_bit_string_list_free (list);
}

void
test_itty_bit_string_list_transpose (void)
{
        itty_bit_string_list_t *list = itty_bit_string_list_new ();
        itty_bit_string_t *bit_string_1 = itty_bit_string_new ();
        itty_bit_string_t *bit_string_2 = itty_bit_string_new ();
        itty_bit_string_t *bit_string_3 = itty_bit_string_new ();
        itty_bit_string_append_word (bit_string_1, 0b101);
        itty_bit_string_append_word (bit_string_2, 0b011);
        itty_bit_string_append_word (bit_string_3, 0b110);
        itty_bit_string_list_append (list, bit_string_1);
        itty_bit_string_list_append (list, bit_string_2);
        itty_bit_string_list_append (list, bit_string_3);

        itty_bit_string_list_t *transposed_list = itty_bit_string_list_transpose (list);
        assert (transposed_list != NULL);
        assert (transposed_list->count == WORD_SIZE_IN_BITS);

        // Verify the transposed bits
        char *representation_1 = itty_bit_string_present (transposed_list->bit_strings[WORD_SIZE_IN_BITS - 1], BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char *representation_2 = itty_bit_string_present (transposed_list->bit_strings[WORD_SIZE_IN_BITS - 2], BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char *representation_3 = itty_bit_string_present (transposed_list->bit_strings[WORD_SIZE_IN_BITS - 3], BIT_STRING_PRESENTATION_FORMAT_BINARY);
        assert (strcmp (representation_1, "0000000000000000000000000000000000000000000000000000000000000110") == 0);
        assert (strcmp (representation_2, "0000000000000000000000000000000000000000000000000000000000000011") == 0);
        assert (strcmp (representation_3, "0000000000000000000000000000000000000000000000000000000000000101") == 0);

        free (representation_1);
        free (representation_2);
        free (representation_3);
        itty_bit_string_list_free (transposed_list);
        itty_bit_string_list_free (list);
}

int
main (void)
{
        test_itty_bit_string_new ();
        test_itty_bit_string_duplicate ();
        test_itty_bit_string_append_word ();
        test_itty_bit_string_append_zeros ();
        test_itty_bit_string_exclusive_nor ();
        test_itty_bit_string_exclusive_or ();
        test_itty_bit_string_combine ();
        test_itty_bit_string_get_pop_count ();
        test_itty_bit_string_evaluate_similarity ();
        test_itty_bit_string_compare ();
        test_itty_bit_string_list_new ();
        test_itty_bit_string_list_append ();
        test_itty_bit_string_list_exclusive_or ();
        test_itty_bit_string_list_condense ();
        test_itty_bit_string_list_sort ();
        test_itty_bit_string_list_transpose ();

        printf ("All tests passed.\n");
        return 0;
}

