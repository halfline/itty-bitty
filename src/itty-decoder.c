#include "itty-decoder.h"

#include "itty-bit-string-list.h"
#include "itty-bit-string-private.h"

#include <stdlib.h>

struct itty_decoder_result_t {
        bool               header_valid;
        size_t             payload_start;
        itty_bit_string_t *score;
        itty_bit_string_t *payload;
};

static size_t
word_count_for_bit_count (size_t bit_count)
{
        return (bit_count + ITTY_BIT_STRING_WORD_SIZE_IN_BITS - 1) / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
}

static itty_bit_string_t *
new_zeroed_bit_string (size_t bit_count)
{
        size_t word_count = word_count_for_bit_count (bit_count);
        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);

        if (!bit_string)
                return NULL;

        for (size_t word_index = 0; word_index < word_count; word_index++)
                itty_bit_string_append_word (bit_string, 0);

        return bit_string;
}

static size_t
score_bit_count_for_payload_bit_count (size_t payload_bit_count)
{
        return payload_bit_count < ITTY_DECODER_SCORE_BITS ? payload_bit_count : ITTY_DECODER_SCORE_BITS;
}

size_t
itty_decoder_gray_code (size_t value)
{
        return value ^ (value >> 1);
}

size_t
itty_decoder_gray_decode (size_t gray_code)
{
        size_t value = gray_code;

        for (size_t shift = 1; shift < ITTY_DECODER_HEADER_BITS; shift <<= 1)
                value ^= value >> shift;

        return value;
}

size_t
itty_decoder_get_score_vote_count (itty_bit_string_t *activation)
{
        size_t activation_bit_count;
        size_t gray_code = 0;
        size_t payload_start;
        size_t score_pop_count = 0;

        if (!activation)
                return 1;

        activation_bit_count = itty_bit_string_get_length (activation);
        for (size_t bit_index = 0; bit_index < ITTY_DECODER_HEADER_BITS; bit_index++) {
                if (itty_bit_string_get_bit (activation, bit_index))
                        gray_code |= ((size_t) 1) << bit_index;
        }

        payload_start = itty_decoder_gray_decode (gray_code);
        if (payload_start + ITTY_DECODER_SCORE_BITS > activation_bit_count)
                return 1;

        for (size_t bit_index = 0; bit_index < ITTY_DECODER_SCORE_BITS; bit_index++) {
                if (itty_bit_string_get_bit (activation, payload_start + bit_index))
                        score_pop_count++;
        }

        return score_pop_count;
}

itty_decoder_result_t *
itty_decoder_decode (itty_bit_string_t *activation,
                     size_t             payload_bit_count)
{
        size_t gray_code = 0;
        size_t activation_bit_count;
        size_t payload_start;
        size_t score_bit_count;
        itty_bit_string_t *score;
        itty_bit_string_t *payload;
        itty_decoder_result_t *result;

        if (!activation || payload_bit_count == 0)
                return NULL;

        activation_bit_count = itty_bit_string_get_length (activation);
        score_bit_count = score_bit_count_for_payload_bit_count (payload_bit_count);
        for (size_t bit_index = 0; bit_index < ITTY_DECODER_HEADER_BITS; bit_index++) {
                if (itty_bit_string_get_bit (activation, bit_index))
                        gray_code |= ((size_t) 1) << bit_index;
        }

        payload_start = itty_decoder_gray_decode (gray_code);
        score = new_zeroed_bit_string (score_bit_count);
        if (!score)
                return NULL;
        payload = new_zeroed_bit_string (payload_bit_count);
        if (!payload) {
                itty_bit_string_free (score);
                return NULL;
        }

        result = malloc (sizeof (itty_decoder_result_t));
        if (!result) {
                itty_bit_string_free (score);
                itty_bit_string_free (payload);
                return NULL;
        }

        result->payload_start = payload_start;
        result->header_valid = payload_start + payload_bit_count <= activation_bit_count;
        result->score = score;
        result->payload = payload;

        if (!result->header_valid)
                return result;

        for (size_t bit_index = 0; bit_index < payload_bit_count; bit_index++)
                itty_bit_string_set_bit (payload,
                                         bit_index,
                                         itty_bit_string_get_bit (activation, payload_start + bit_index));

        for (size_t bit_index = 0; bit_index < score_bit_count; bit_index++)
                itty_bit_string_set_bit (score,
                                         bit_index,
                                         itty_bit_string_get_bit (payload, bit_index));

        return result;
}

bool
itty_decoder_select_output (itty_bit_string_list_t *outputs,
                            size_t                  payload_bit_count,
                            size_t                 *index)
{
        bool found_one = false;
        size_t best_index = 0;
        size_t best_score_pop_count = 0;

        if (!outputs || !index)
                return false;

        for (size_t current_index = 0; current_index < itty_bit_string_list_get_length (outputs); current_index++) {
                itty_bit_string_t *activation = itty_bit_string_list_fetch (outputs, current_index);
                itty_decoder_result_t *result = itty_decoder_decode (activation, payload_bit_count);
                size_t current_score_pop_count;

                if (!result)
                        continue;

                if (!itty_decoder_result_header_valid (result)) {
                        itty_decoder_result_free (result);
                        continue;
                }

                current_score_pop_count = itty_decoder_result_get_score_pop_count (result);
                if (!found_one || current_score_pop_count > best_score_pop_count) {
                        found_one = true;
                        best_index = current_index;
                        best_score_pop_count = current_score_pop_count;
                }

                itty_decoder_result_free (result);
        }

        if (!found_one)
                return false;

        *index = best_index;
        return true;
}

void
itty_decoder_result_free (itty_decoder_result_t *result)
{
        if (!result)
                return;

        itty_bit_string_free (result->score);
        itty_bit_string_free (result->payload);
        free (result);
}

bool
itty_decoder_result_header_valid (itty_decoder_result_t *result)
{
        return result->header_valid;
}

size_t
itty_decoder_result_get_payload_start (itty_decoder_result_t *result)
{
        return result->payload_start;
}

itty_bit_string_t *
itty_decoder_result_get_score (itty_decoder_result_t *result)
{
        return result->score;
}

size_t
itty_decoder_result_get_score_pop_count (itty_decoder_result_t *result)
{
        if (!result || !result->score)
                return 0;

        return itty_bit_string_get_pop_count (result->score);
}

itty_bit_string_t *
itty_decoder_result_get_payload (itty_decoder_result_t *result)
{
        return result->payload;
}

size_t
itty_decoder_measure_distance (itty_decoder_result_t *result,
                               itty_bit_string_t     *expected_payload)
{
        itty_bit_string_t *difference;
        size_t distance;

        if (!result || !expected_payload || !result->header_valid)
                return itty_bit_string_get_length (expected_payload);

        difference = itty_bit_string_exclusive_or (result->payload, expected_payload);
        if (!difference)
                return itty_bit_string_get_length (expected_payload);

        distance = itty_bit_string_get_pop_count (difference);
        itty_bit_string_free (difference);
        return distance;
}
