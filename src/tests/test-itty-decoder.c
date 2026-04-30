#include "itty-bit-string-list.h"
#include "itty-bit-string-private.h"
#include "itty-bit-string.h"
#include "itty-decoder.h"

#include <assert.h>
#include <stdio.h>

static itty_bit_string_t *
create_activation (size_t payload_start,
                   size_t score_bits,
                   size_t payload_bits)
{
        itty_bit_string_t *activation = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        size_t gray_header = itty_decoder_gray_code (payload_start);

        itty_bit_string_append_word (activation, 0);
        itty_bit_string_append_word (activation, 0);

        for (size_t bit_index = 0; bit_index < ITTY_DECODER_HEADER_BITS; bit_index++)
                itty_bit_string_set_bit (activation,
                                         bit_index,
                                         (gray_header >> bit_index) & 1U);

        for (size_t bit_index = 0; bit_index < ITTY_DECODER_SCORE_BITS; bit_index++)
                itty_bit_string_set_bit (activation,
                                         payload_start + bit_index,
                                         (score_bits >> bit_index) & 1U);

        for (size_t bit_index = 0; bit_index < 8; bit_index++)
                itty_bit_string_set_bit (activation,
                                         payload_start + ITTY_DECODER_SCORE_BITS + bit_index,
                                         (payload_bits >> bit_index) & 1U);

        return activation;
}

static void
test_itty_decoder_select_output_uses_decoded_score (void)
{
        itty_bit_string_list_t *outputs = itty_bit_string_list_new ();
        size_t selected_index = 99;

        itty_bit_string_list_append (outputs, create_activation (64, 0b00000111, 0xaa));
        itty_bit_string_list_append (outputs, create_activation (64, 0b00011111, 0x55));

        assert (itty_decoder_select_output (outputs, 16, &selected_index));
        assert (selected_index == 1);

        itty_bit_string_list_free (outputs);
}

static void
test_itty_decoder_decode_exposes_score_bits (void)
{
        itty_bit_string_t *activation = create_activation (64, 0b00001111, 0xc3);
        itty_decoder_result_t *result = itty_decoder_decode (activation, 16);
        itty_bit_string_t *score;

        assert (result != NULL);
        assert (itty_decoder_result_header_valid (result));
        assert (itty_decoder_result_get_payload_start (result) == 64);
        assert (itty_decoder_result_get_score_pop_count (result) == 4);

        score = itty_decoder_result_get_score (result);
        assert (score != NULL);
        assert (itty_bit_string_get_pop_count (score) == 4);

        itty_decoder_result_free (result);
        itty_bit_string_free (activation);
}

int
main (void)
{
        test_itty_decoder_select_output_uses_decoded_score ();
        test_itty_decoder_decode_exposes_score_bits ();
        printf ("All itty-decoder tests passed.\n");
        return 0;
}
