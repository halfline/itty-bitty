#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "itty-bit-string.h"
#include "itty-bit-string-private.h"
#include "itty-bit-string-list.h"
#include "itty-bit-string-list-private.h"

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
        char *representation = itty_bit_string_present (result_list->bit_strings[0], ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
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
        char *representation = itty_bit_string_present (condensed, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
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

        itty_bit_string_list_sort (list, ITTY_BIT_STRING_SORT_ORDER_ASCENDING);

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
        itty_bit_string_append_word (bit_string_1, 0b1100);
        itty_bit_string_append_word (bit_string_2, 0b1010);
        itty_bit_string_append_word (bit_string_3, 0b1111);
        printf ("%s\n", itty_bit_string_present (bit_string_1, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY));
        printf ("%s\n", itty_bit_string_present (bit_string_2, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY));
        printf ("%s\n", itty_bit_string_present (bit_string_3, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY));
        itty_bit_string_list_append (list, bit_string_1);
        itty_bit_string_list_append (list, bit_string_2);
        itty_bit_string_list_append (list, bit_string_3);

        itty_bit_string_list_t *transposed_list = itty_bit_string_list_transpose (list);
        assert (transposed_list != NULL);
        assert (transposed_list->count == 4);

        // Verify the transposed bits
        char *representation_1 = itty_bit_string_present (transposed_list->bit_strings[0], ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char *representation_2 = itty_bit_string_present (transposed_list->bit_strings[1], ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char *representation_3 = itty_bit_string_present (transposed_list->bit_strings[2], ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
        char *representation_4 = itty_bit_string_present (transposed_list->bit_strings[3], ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY);
        assert (strcmp (representation_1, "0000000000000000000000000000000000000000000000000000000000000111") == 0);
        assert (strcmp (representation_2, "0000000000000000000000000000000000000000000000000000000000000101") == 0);
        assert (strcmp (representation_3, "0000000000000000000000000000000000000000000000000000000000000110") == 0);
        assert (strcmp (representation_4, "0000000000000000000000000000000000000000000000000000000000000100") == 0);

        free (representation_1);
        free (representation_2);
        free (representation_3);
        free (representation_4);
        itty_bit_string_list_free (transposed_list);
        itty_bit_string_list_free (list);
}

int
main (void)
{
        test_itty_bit_string_list_new ();
        test_itty_bit_string_list_append ();
        test_itty_bit_string_list_exclusive_or ();
        test_itty_bit_string_list_transpose ();
        test_itty_bit_string_list_condense ();
        test_itty_bit_string_list_sort ();

        printf ("All itty-bit-string-list tests passed.\n");
        return 0;
}

