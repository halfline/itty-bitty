#include "itty-position.h"
#include "itty-bit-string-private.h"
#include <assert.h>
#include <stdio.h>

static void
test_itty_position_locality_bonus (void)
{
        assert (itty_position_locality_bonus (4, 4, 4) == 4);
        assert (itty_position_locality_bonus (4, 3, 4) == 3);
        assert (itty_position_locality_bonus (4, 6, 4) == 2);
        assert (itty_position_locality_bonus (4, 8, 4) == 0);
        assert (itty_position_locality_bonus (4, 9, 4) == 0);
        assert (itty_position_locality_bonus (4, 4, 0) == 0);
}

static void
test_itty_position_gray_code (void)
{
        assert (itty_position_gray_code (0) == 0);
        assert (itty_position_gray_code (1) == 1);
        assert (itty_position_gray_code (2) == 3);
        assert (itty_position_gray_code (3) == 2);
        assert (itty_position_gray_code (4) == 6);
}

static void
test_itty_position_gray_encode (void)
{
        itty_bit_string_t *position = itty_position_gray_encode (4, 2);
        assert (position != NULL);
        assert (position->number_of_words == 2);
        assert (position->words[0] == 6);
        assert (position->words[1] == 0);
        itty_bit_string_free (position);

        assert (itty_position_gray_encode (4, 0) == NULL);
}

static void
test_itty_position_gray_similarity (void)
{
        assert (itty_position_gray_similarity (0, 0, 3) == 3);
        assert (itty_position_gray_similarity (0, 1, 3) == 2);
        assert (itty_position_gray_similarity (0, 2, 3) == 1);

        /* Gray code is locally smooth, not a monotonic distance metric. */
        assert (itty_position_gray_similarity (0, 7, 3) == 2);
}

int
main (void)
{
        test_itty_position_locality_bonus ();
        test_itty_position_gray_code ();
        test_itty_position_gray_encode ();
        test_itty_position_gray_similarity ();
        printf ("All itty-position tests passed.\n");
        return 0;
}
