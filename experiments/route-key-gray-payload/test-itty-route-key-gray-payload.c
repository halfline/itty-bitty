#include "itty-route-key-gray-payload.h"

#include "itty-bit-string-private.h"
#include "itty-self-describing-gray-payload.h"

#include <assert.h>
#include <stdio.h>
#include <stdint.h>

static itty_bit_string_t *
create_bit_string (size_t word)
{
        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        itty_bit_string_append_word (bit_string, word);
        return bit_string;
}

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

static itty_bit_string_t *
create_target_for_payload (size_t payload_bits,
                           size_t payload_value)
{
        itty_bit_string_t *payload = create_zero_bit_string (1);
        itty_bit_string_t *target;

        set_bits_from_word (payload, payload_value, 0, payload_bits);
        target = itty_self_describing_gray_payload_build_target (payload, payload_bits);
        itty_bit_string_free (payload);
        return target;
}

static itty_bit_string_t *
create_activation_for_payload (size_t payload_bits,
                               size_t payload_start,
                               size_t payload_value)
{
        itty_bit_string_t *payload = create_zero_bit_string (1);
        itty_bit_string_t *activation;

        set_bits_from_word (payload, payload_value, 0, payload_bits);
        activation = itty_self_describing_gray_payload_build_activation (payload,
                                                                         payload_bits,
                                                                         payload_start,
                                                                         64);
        itty_bit_string_free (payload);
        return activation;
}

static void
print_result_metric (const char                              *label,
                     itty_route_key_gray_payload_result_t    *result)
{
        char committed_distance[32];

        if (result->committed_distance == SIZE_MAX)
                snprintf (committed_distance, sizeof (committed_distance), "unset");
        else
                snprintf (committed_distance, sizeof (committed_distance), "%zu", result->committed_distance);

        printf ("%-18s route=%zu gap=%zu start=%zu header_valid=%s measured_distance=%zu committed_distance=%s\n",
                label,
                result->selected_route,
                result->selected_gap,
                result->decoded_payload_start,
                result->header_valid ? "yes" : "no",
                result->measured_distance,
                committed_distance);
}

static void
test_itty_route_key_gray_payload_abcd_chain (void)
{
        itty_route_key_gray_payload_table_t *table = itty_route_key_gray_payload_table_new ();
        itty_bit_string_t *probe_a = create_bit_string (0x0f0f0f0f);
        itty_bit_string_t *probe_b = create_bit_string (0xf0f0f0f0);
        itty_bit_string_t *probe_c = create_bit_string (0x33333333);
        itty_bit_string_t *probe_d = create_bit_string (0x5555aaaa);
        itty_bit_string_t *target_a = create_target_for_payload (4, 0b1001);
        itty_bit_string_t *target_b = create_target_for_payload (4, 0b0110);
        itty_bit_string_t *target_c = create_target_for_payload (4, 0b1110);
        itty_bit_string_t *target_d = create_target_for_payload (4, 0b0101);
        itty_bit_string_t *activation_a = create_activation_for_payload (4, 12, 0b1001);
        itty_bit_string_t *activation_b = create_activation_for_payload (4, 20, 0b0110);
        itty_bit_string_t *activation_c = create_activation_for_payload (4, 28, 0b1110);
        itty_bit_string_t *activation_d = create_activation_for_payload (4, 36, 0b0101);
        itty_route_key_gray_payload_result_t result;

        assert (table != NULL);
        assert (itty_route_key_gray_payload_table_append_route (table, probe_b, target_b, 4));
        assert (itty_route_key_gray_payload_table_append_route (table, probe_a, target_a, 4));
        assert (itty_route_key_gray_payload_table_measure (table, probe_a, activation_a, &result));
        assert (result.selected_route == 1);
        assert (result.selected_gap > 0);
        assert (result.header_valid);
        assert (result.decoded_payload_start == 12);
        assert (result.measured_distance == 0);
        print_result_metric ("A after AB", &result);
        assert (itty_route_key_gray_payload_table_measure (table, probe_b, activation_b, &result));
        assert (result.selected_route == 0);
        assert (result.header_valid);
        assert (result.decoded_payload_start == 20);
        assert (result.measured_distance == 0);
        print_result_metric ("B after AB", &result);

        assert (itty_route_key_gray_payload_table_append_route (table, probe_c, target_c, 4));
        assert (itty_route_key_gray_payload_table_measure (table, probe_a, activation_a, &result));
        assert (result.selected_route == 1);
        print_result_metric ("A after ABC", &result);
        assert (itty_route_key_gray_payload_table_measure (table, probe_b, activation_b, &result));
        assert (result.selected_route == 0);
        print_result_metric ("B after ABC", &result);
        assert (itty_route_key_gray_payload_table_measure (table, probe_c, activation_c, &result));
        assert (result.selected_route == 2);
        assert (result.header_valid);
        assert (result.decoded_payload_start == 28);
        assert (result.measured_distance == 0);
        print_result_metric ("C after ABC", &result);

        assert (itty_route_key_gray_payload_table_append_route (table, probe_d, target_d, 4));
        assert (itty_route_key_gray_payload_table_measure (table, probe_a, activation_a, &result));
        assert (result.selected_route == 1);
        print_result_metric ("A after ABCD", &result);
        assert (itty_route_key_gray_payload_table_measure (table, probe_b, activation_b, &result));
        assert (result.selected_route == 0);
        print_result_metric ("B after ABCD", &result);
        assert (itty_route_key_gray_payload_table_measure (table, probe_c, activation_c, &result));
        assert (result.selected_route == 2);
        print_result_metric ("C after ABCD", &result);
        assert (itty_route_key_gray_payload_table_measure (table, probe_d, activation_d, &result));
        assert (result.selected_route == 3);
        assert (result.header_valid);
        assert (result.decoded_payload_start == 36);
        assert (result.measured_distance == 0);
        print_result_metric ("D after ABCD", &result);

        itty_bit_string_free (probe_a);
        itty_bit_string_free (probe_b);
        itty_bit_string_free (probe_c);
        itty_bit_string_free (probe_d);
        itty_bit_string_free (target_a);
        itty_bit_string_free (target_b);
        itty_bit_string_free (target_c);
        itty_bit_string_free (target_d);
        itty_bit_string_free (activation_a);
        itty_bit_string_free (activation_b);
        itty_bit_string_free (activation_c);
        itty_bit_string_free (activation_d);
        itty_route_key_gray_payload_table_free (table);
}

static void
test_itty_route_key_gray_payload_commit_selected_route (void)
{
        itty_route_key_gray_payload_table_t *table = itty_route_key_gray_payload_table_new ();
        itty_bit_string_t *probe = create_bit_string (0x12345678);
        itty_bit_string_t *target = create_target_for_payload (4, 0b1011);
        itty_bit_string_t *activation = create_activation_for_payload (4, 18, 0b1011);
        itty_route_key_gray_payload_result_t result;

        assert (table != NULL);
        assert (itty_route_key_gray_payload_table_append_route (table, probe, target, 4));
        assert (itty_route_key_gray_payload_table_measure (table, probe, activation, &result));
        assert (result.selected_route == 0);
        assert (result.header_valid);
        assert (result.decoded_payload_start == 18);
        assert (result.measured_distance == 0);
        assert (result.committed_distance == SIZE_MAX);
        print_result_metric ("Commit before", &result);

        assert (itty_route_key_gray_payload_table_commit (table, probe, activation, &result));
        assert (result.selected_route == 0);
        assert (result.header_valid);
        assert (result.decoded_payload_start == 18);
        assert (result.measured_distance == 0);
        assert (result.committed_distance == 0);
        print_result_metric ("Commit after", &result);

        itty_bit_string_set_bit (activation, 18, false);
        assert (itty_route_key_gray_payload_table_measure (table, probe, activation, &result));
        assert (result.measured_distance == 1);
        assert (result.committed_distance == 0);
        print_result_metric ("Commit damaged", &result);

        itty_bit_string_free (probe);
        itty_bit_string_free (target);
        itty_bit_string_free (activation);
        itty_route_key_gray_payload_table_free (table);
}

int
main (void)
{
        printf ("Route-key + self-describing Gray payload metrics\n");
        test_itty_route_key_gray_payload_abcd_chain ();
        test_itty_route_key_gray_payload_commit_selected_route ();
        printf ("All itty-route-key-gray-payload tests passed.\n");
        return 0;
}
