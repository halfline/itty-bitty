#include "itty-self-describing-gray-payload.h"

#include "itty-bit-string-private.h"
#include <stdlib.h>

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

size_t
itty_self_describing_gray_payload_gray_code (size_t value)
{
        return value ^ (value >> 1);
}

size_t
itty_self_describing_gray_payload_gray_decode (size_t gray_code)
{
        size_t value = gray_code;

        for (size_t shift = 1; shift < ITTY_SELF_DESCRIBING_GRAY_PAYLOAD_HEADER_BITS; shift <<= 1)
                value ^= value >> shift;

        return value;
}

itty_bit_string_t *
itty_self_describing_gray_payload_build_activation (itty_bit_string_t *payload,
                                                    size_t             payload_bit_count,
                                                    size_t             payload_start,
                                                    size_t             activation_bit_count)
{
        itty_bit_string_t *activation;
        size_t gray_code;

        if (!payload || payload_bit_count == 0 || activation_bit_count < payload_start + payload_bit_count)
                return NULL;

        activation = new_zeroed_bit_string (activation_bit_count);
        if (!activation)
                return NULL;

        gray_code = itty_self_describing_gray_payload_gray_code (payload_start & 0xff);
        for (size_t bit_index = 0; bit_index < ITTY_SELF_DESCRIBING_GRAY_PAYLOAD_HEADER_BITS; bit_index++)
                itty_bit_string_set_bit (activation, bit_index, ((gray_code >> bit_index) & 1U) != 0);

        for (size_t bit_index = 0; bit_index < payload_bit_count; bit_index++)
                itty_bit_string_set_bit (activation,
                                         payload_start + bit_index,
                                         itty_bit_string_get_bit (payload, bit_index));

        return activation;
}

itty_bit_string_t *
itty_self_describing_gray_payload_build_target (itty_bit_string_t *payload,
                                                size_t             payload_bit_count)
{
        itty_bit_string_t *target;

        if (!payload || payload_bit_count == 0)
                return NULL;

        target = new_zeroed_bit_string (payload_bit_count);
        if (!target)
                return NULL;

        for (size_t bit_index = 0; bit_index < payload_bit_count; bit_index++)
                itty_bit_string_set_bit (target,
                                         bit_index,
                                         itty_bit_string_get_bit (payload, bit_index));

        return target;
}

bool
itty_self_describing_gray_payload_measure (itty_bit_string_t                           *activation,
                                           itty_bit_string_t                           *target,
                                           size_t                                       payload_bit_count,
                                           itty_self_describing_gray_payload_result_t  *result)
{
        size_t gray_code = 0;
        size_t payload_start;
        size_t activation_bit_count;
        size_t distance = 0;

        if (!activation || !target || !result || payload_bit_count == 0)
                return false;

        activation_bit_count = itty_bit_string_get_length (activation);
        for (size_t bit_index = 0; bit_index < ITTY_SELF_DESCRIBING_GRAY_PAYLOAD_HEADER_BITS; bit_index++) {
                if (itty_bit_string_get_bit (activation, bit_index))
                        gray_code |= ((size_t) 1) << bit_index;
        }

        payload_start = itty_self_describing_gray_payload_gray_decode (gray_code);
        *result = (itty_self_describing_gray_payload_result_t) {
                .decoded_payload_start = payload_start,
                .header_valid = payload_start + payload_bit_count <= activation_bit_count,
                .measured_distance = payload_bit_count,
        };

        if (!result->header_valid)
                return true;

        for (size_t bit_index = 0; bit_index < payload_bit_count; bit_index++) {
                bool actual_bit = itty_bit_string_get_bit (activation, payload_start + bit_index);
                bool target_bit = itty_bit_string_get_bit (target, bit_index);

                if (actual_bit != target_bit)
                        distance++;
        }

        result->measured_distance = distance;
        return true;
}
