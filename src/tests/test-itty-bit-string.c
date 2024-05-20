#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "itty-bit-string.h"
#include "itty-bit-string-private.h"

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
        char *original_representation = itty_bit_string_present (original, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char *duplicate_representation = itty_bit_string_present (duplicate, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
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
        char *representation = itty_bit_string_present (bit_string, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
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
        char *representation = itty_bit_string_present (bit_string, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
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
        char *representation = itty_bit_string_present (result, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
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
        char *representation = itty_bit_string_present (result, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
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
        char *representation = itty_bit_string_present (result, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
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
test_itty_bit_string_compare_by_pop_count (void)
{
        itty_bit_string_t *a = itty_bit_string_new ();
        itty_bit_string_t *b = itty_bit_string_new ();
        itty_bit_string_append_word (a, 0b1100);
        itty_bit_string_append_word (b, 0b1001);
        int cmp = itty_bit_string_compare_by_pop_count (a, b);
        assert (cmp == 0);
        itty_bit_string_append_word (a, 0b1111);
        cmp = itty_bit_string_compare_by_pop_count (a, b);
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
        char *representation = itty_bit_string_present (doubled, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
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
        itty_bit_string_append_word (bit_string, 0b1100);
        itty_bit_string_t *reduced = itty_bit_string_reduce_by_half (bit_string);
        assert (reduced != NULL);
        char *representation = itty_bit_string_present (reduced, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char expected[] = "0000000000000000000000000000000000000000000000000000000000001100";
        assert (strcmp (representation, expected) == 0);
        free (representation);
        itty_bit_string_free (bit_string);
        itty_bit_string_free (reduced);
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
        test_itty_bit_string_compare_by_pop_count ();
        test_itty_bit_string_double ();
        test_itty_bit_string_reduce_by_half ();

        printf ("All itty-bit-string tests passed.\n");
        return 0;
}

