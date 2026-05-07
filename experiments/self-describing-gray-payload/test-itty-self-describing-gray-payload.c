#include "itty-self-describing-gray-payload.h"

#include "itty-bit-string-private.h"
#include <assert.h>
#include <stdio.h>

static itty_bit_string_t *
create_zero_bit_string (size_t word_count)
{
        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);

        for (size_t word_index = 0; word_index < word_count; word_index++)
                itty_bit_string_append_word (bit_string, 0);

        return bit_string;
}

static void
set_bits_from_word (itty_bit_string_t *bit_string,
                    size_t             value,
                    size_t             bit_offset,
                    size_t             bit_count)
{
        for (size_t bit_index = 0; bit_index < bit_count; bit_index++)
                itty_bit_string_set_bit (bit_string,
                                         bit_offset + bit_index,
                                         ((value >> bit_index) & 1U) != 0);
}

static void
print_metric (const char                                   *label,
              itty_self_describing_gray_payload_result_t   *result)
{
        printf ("%-16s start=%zu header_valid=%s measured_distance=%zu\n",
                label,
                result->decoded_payload_start,
                result->header_valid ? "yes" : "no",
                result->measured_distance);
}

static void
test_itty_self_describing_gray_payload_examples (void)
{
        size_t starts[4] = { 12, 20, 28, 36 };
        size_t payloads[4] = { 0b1001, 0b0110, 0b1110, 0b0101 };

        for (size_t example_index = 0; example_index < 4; example_index++) {
                itty_bit_string_t *payload = create_zero_bit_string (1);
                itty_bit_string_t *activation;
                itty_bit_string_t *target;
                itty_self_describing_gray_payload_result_t result;

                set_bits_from_word (payload, payloads[example_index], 0, 4);
                activation = itty_self_describing_gray_payload_build_activation (payload,
                                                                                 4,
                                                                                 starts[example_index],
                                                                                 64);
                target = itty_self_describing_gray_payload_build_target (payload, 4);

                assert (activation != NULL);
                assert (target != NULL);
                assert (itty_self_describing_gray_payload_measure (activation, target, 4, &result));
                assert (result.header_valid);
                assert (result.decoded_payload_start == starts[example_index]);
                assert (result.measured_distance == 0);

                if (example_index == 0)
                        print_metric ("example A", &result);
                else if (example_index == 1)
                        print_metric ("example B", &result);
                else if (example_index == 2)
                        print_metric ("example C", &result);
                else
                        print_metric ("example D", &result);

                itty_bit_string_free (target);
                itty_bit_string_free (activation);
                itty_bit_string_free (payload);
        }
}

static void
test_itty_self_describing_gray_payload_detects_bad_payload (void)
{
        itty_bit_string_t *payload = create_zero_bit_string (1);
        itty_bit_string_t *activation;
        itty_bit_string_t *target;
        itty_self_describing_gray_payload_result_t result;

        set_bits_from_word (payload, 0b1011, 0, 4);
        activation = itty_self_describing_gray_payload_build_activation (payload, 4, 18, 64);
        target = itty_self_describing_gray_payload_build_target (payload, 4);
        assert (activation != NULL);
        assert (target != NULL);

        itty_bit_string_set_bit (activation, 18, false);
        assert (itty_self_describing_gray_payload_measure (activation, target, 4, &result));
        assert (result.header_valid);
        assert (result.decoded_payload_start == 18);
        assert (result.measured_distance == 1);
        print_metric ("bad payload", &result);

        itty_bit_string_free (target);
        itty_bit_string_free (activation);
        itty_bit_string_free (payload);
}

static void
test_itty_self_describing_gray_payload_rejects_bad_header (void)
{
        itty_bit_string_t *payload = create_zero_bit_string (1);
        itty_bit_string_t *activation = create_zero_bit_string (1);
        itty_bit_string_t *target;
        itty_self_describing_gray_payload_result_t result;

        set_bits_from_word (payload, 0b1111, 0, 4);
        target = itty_self_describing_gray_payload_build_target (payload, 4);
        assert (target != NULL);

        set_bits_from_word (activation, itty_self_describing_gray_payload_gray_code (63), 0, 8);
        assert (itty_self_describing_gray_payload_measure (activation, target, 4, &result));
        assert (!result.header_valid);
        assert (result.decoded_payload_start == 63);
        print_metric ("bad header", &result);

        itty_bit_string_free (target);
        itty_bit_string_free (activation);
        itty_bit_string_free (payload);
}

int
main (void)
{
        printf ("Self-describing Gray payload metrics\n");
        test_itty_self_describing_gray_payload_examples ();
        test_itty_self_describing_gray_payload_detects_bad_payload ();
        test_itty_self_describing_gray_payload_rejects_bad_header ();
        printf ("All itty-self-describing-gray-payload tests passed.\n");
        return 0;
}
