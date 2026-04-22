#include "itty-bit-string.h"
#include "itty-bit-string-list.h"
#include "itty-manager.h"
#include "itty-network.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static itty_bit_string_t *
create_bit_string (size_t word)
{
        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        itty_bit_string_append_word (bit_string, word);
        return bit_string;
}

static itty_bit_string_list_t *
create_input_list (void)
{
        itty_bit_string_list_t *input = itty_bit_string_list_new ();
        itty_bit_string_list_append (input, create_bit_string (0b1111));
        itty_bit_string_list_append (input, create_bit_string (0b1111));
        return input;
}

static itty_bit_string_list_t *
create_input_list_with_count (size_t count)
{
        itty_bit_string_list_t *input = itty_bit_string_list_new ();

        for (size_t i = 0; i < count; i++)
                itty_bit_string_list_append (input, create_bit_string (0b1111 + i));

        return input;
}

static itty_bit_string_list_t *
create_masks_with_count (size_t count)
{
        itty_bit_string_list_t *masks = itty_bit_string_list_new ();

        for (size_t i = 0; i < count; i++)
                itty_bit_string_list_append (masks, create_bit_string (i));

        return masks;
}

static itty_bit_string_list_t *
create_masks (void)
{
        return create_masks_with_count (2);
}

static itty_bit_string_list_t *
create_affinity_traits (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_append (traits, create_bit_string (~0UL));
        itty_bit_string_list_append (traits, create_bit_string (0));
        return traits;
}

static itty_bit_string_list_t *
create_affinity_imprints (void)
{
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_list_append (imprints, create_bit_string (0b1010));
        itty_bit_string_list_append (imprints, create_bit_string (0b0101));
        return imprints;
}

static void
test_itty_network_affinity_node_new_rejects_shape_mismatch (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_list_append (traits, create_bit_string (~0UL));
        itty_bit_string_list_append (traits, create_bit_string (0));
        itty_bit_string_list_append (imprints, create_bit_string (0b1010));

        itty_network_node_t *node = itty_network_affinity_node_new (traits, imprints, NULL);
        assert (node == NULL);

        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static itty_network_t *
create_network (void)
{
        itty_network_t *network = itty_network_new ();
        itty_network_layer_t *layer = itty_network_layer_new ();

        itty_network_layer_append (layer, itty_network_feed_node_new (create_masks ()));
        itty_network_layer_append (layer, itty_network_feed_node_new (create_masks ()));
        itty_network_append (network, layer);

        return network;
}

static itty_network_t *
create_affinity_network (void)
{
        itty_network_t *network = itty_network_new ();
        itty_network_layer_t *layer = itty_network_layer_new ();
        itty_affinity_probe_options_t options = {
                .total_votes = 7,
                .score_bit_length = ITTY_BIT_STRING_WORD_SIZE_IN_BITS
        };

        itty_network_layer_append (layer, itty_network_affinity_node_new (create_affinity_traits (),
                                                                          create_affinity_imprints (),
                                                                          &options));
        itty_network_append (network, layer);

        return network;
}

static itty_network_t *
create_too_wide_affinity_network (void)
{
        itty_network_t *network = itty_network_new ();
        itty_network_layer_t *layer = itty_network_layer_new ();
        itty_affinity_probe_options_t options = {
                .total_votes = 7,
                .score_bit_length = ITTY_BIT_STRING_WORD_SIZE_IN_BITS + 1
        };

        itty_network_layer_append (layer, itty_network_affinity_node_new (create_affinity_traits (),
                                                                          create_affinity_imprints (),
                                                                          &options));
        itty_network_append (network, layer);

        return network;
}

static itty_network_t *
create_two_layer_network (void)
{
        itty_network_t *network = itty_network_new ();
        itty_network_layer_t *first_layer = itty_network_layer_new ();
        itty_network_layer_t *second_layer = itty_network_layer_new ();

        itty_network_layer_append (first_layer, itty_network_feed_node_new (create_masks ()));
        itty_network_layer_append (first_layer, itty_network_feed_node_new (create_masks ()));
        itty_network_layer_append (second_layer, itty_network_feed_node_new (create_masks ()));

        itty_network_append (network, first_layer);
        itty_network_append (network, second_layer);

        return network;
}

static itty_network_t *
create_network_with_mask_counts (size_t const *mask_counts,
                                 size_t        node_count)
{
        itty_network_t *network = itty_network_new ();
        itty_network_layer_t *layer = itty_network_layer_new ();

        for (size_t i = 0; i < node_count; i++)
                itty_network_layer_append (layer, itty_network_feed_node_new (create_masks_with_count (mask_counts[i])));

        itty_network_append (network, layer);

        return network;
}

static void
free_feed_result (itty_bit_string_list_t *input,
                  itty_bit_string_list_t *output)
{
        if (output && output != input)
                itty_bit_string_list_free (output);
        itty_bit_string_list_free (input);
}

static void
assert_bit_string_lists_equal (itty_bit_string_list_t *a,
                               itty_bit_string_list_t *b)
{
        assert (a != NULL);
        assert (b != NULL);
        assert (itty_bit_string_list_get_length (a) == itty_bit_string_list_get_length (b));

        for (size_t i = 0; i < itty_bit_string_list_get_length (a); i++) {
                itty_bit_string_t *a_bit_string = itty_bit_string_list_fetch (a, i);
                itty_bit_string_t *b_bit_string = itty_bit_string_list_fetch (b, i);
                assert (itty_bit_string_compare (a_bit_string, b_bit_string) == 0);
        }
}

static void
assert_network_manager_matches_sync (itty_network_t *network,
                                     size_t          input_count)
{
        itty_bit_string_list_t *sync_input = create_input_list_with_count (input_count);
        itty_bit_string_list_t *async_input = create_input_list_with_count (input_count);
        itty_manager_t *manager = itty_manager_new ();

        itty_bit_string_list_t *sync_output = itty_network_feed (network, sync_input);
        itty_bit_string_list_t *async_output = itty_network_feed_with_manager (network, async_input, manager);

        assert_bit_string_lists_equal (sync_output, async_output);

        free_feed_result (sync_input, sync_output);
        free_feed_result (async_input, async_output);
        itty_manager_free (manager);
}

static size_t
count_occurrences (const char *text,
                   const char *needle)
{
        size_t count = 0;
        const char *match = text;

        while ((match = strstr (match, needle)) != NULL) {
                count++;
                match += strlen (needle);
        }

        return count;
}

static void
test_itty_network_feed_with_manager (void)
{
        itty_network_t *network = create_network ();
        itty_bit_string_list_t *sync_input = create_input_list ();
        itty_bit_string_list_t *async_input = create_input_list ();
        itty_manager_t *manager = itty_manager_new ();

        itty_bit_string_list_t *sync_output = itty_network_feed (network, sync_input);
        itty_bit_string_list_t *async_output = itty_network_feed_with_manager (network, async_input, manager);

        assert_bit_string_lists_equal (sync_output, async_output);

        free_feed_result (sync_input, sync_output);
        free_feed_result (async_input, async_output);
        itty_manager_free (manager);
        itty_network_free (network);
}

static void
test_itty_network_feed_with_manager_edge_cases (void)
{
        size_t zero_mask_counts[] = { 0 };
        itty_network_t *zero_mask_network = create_network_with_mask_counts (zero_mask_counts, 1);
        assert_network_manager_matches_sync (zero_mask_network, 2);
        itty_network_free (zero_mask_network);

        size_t short_mask_counts[] = { 1 };
        itty_network_t *short_mask_network = create_network_with_mask_counts (short_mask_counts, 1);
        assert_network_manager_matches_sync (short_mask_network, 3);
        itty_network_free (short_mask_network);

        size_t long_mask_counts[] = { 3 };
        itty_network_t *long_mask_network = create_network_with_mask_counts (long_mask_counts, 1);
        assert_network_manager_matches_sync (long_mask_network, 1);
        itty_network_free (long_mask_network);

        size_t regular_mask_counts[] = { 2 };
        itty_network_t *empty_input_network = create_network_with_mask_counts (regular_mask_counts, 1);
        assert_network_manager_matches_sync (empty_input_network, 0);
        itty_network_free (empty_input_network);

        itty_network_t *empty_layer_network = itty_network_new ();
        itty_network_append (empty_layer_network, itty_network_layer_new ());
        assert_network_manager_matches_sync (empty_layer_network, 2);
        itty_network_free (empty_layer_network);

        itty_network_t *empty_network = itty_network_new ();
        assert_network_manager_matches_sync (empty_network, 2);
        itty_network_free (empty_network);
}

static void
test_itty_network_affinity_node_matches_affinity_probe_list (void)
{
        itty_network_t *network = create_affinity_network ();
        itty_bit_string_list_t *network_input = itty_bit_string_list_new ();
        itty_bit_string_list_append (network_input, create_bit_string (~0UL));
        itty_bit_string_list_append (network_input, create_bit_string (0));

        itty_bit_string_list_t *traits = create_affinity_traits ();
        itty_bit_string_list_t *imprints = create_affinity_imprints ();
        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        assert (affinity != NULL);
        itty_bit_string_list_t *affinity_input = itty_bit_string_list_new ();
        itty_bit_string_list_append (affinity_input, create_bit_string (~0UL));
        itty_bit_string_list_append (affinity_input, create_bit_string (0));
        itty_affinity_probe_options_t options = {
                .total_votes = 7,
                .score_bit_length = ITTY_BIT_STRING_WORD_SIZE_IN_BITS
        };

        itty_bit_string_list_t *network_output = itty_network_feed (network, network_input);
        itty_bit_string_list_t *affinity_output = itty_affinity_probe_list (affinity,
                                                                            affinity_input,
                                                                            &options);
        assert_bit_string_lists_equal (network_output, affinity_output);

        itty_bit_string_list_free (affinity_output);
        itty_bit_string_list_free (affinity_input);
        itty_affinity_free (affinity);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
        free_feed_result (network_input, network_output);
        itty_network_free (network);
}

static void
test_itty_network_affinity_node_with_manager_matches_sync (void)
{
        itty_network_t *network = create_affinity_network ();
        itty_bit_string_list_t *sync_input = itty_bit_string_list_new ();
        itty_bit_string_list_t *manager_input = itty_bit_string_list_new ();
        itty_manager_t *manager = itty_manager_new ();
        itty_bit_string_list_append (sync_input, create_bit_string (~0UL));
        itty_bit_string_list_append (sync_input, create_bit_string (0));
        itty_bit_string_list_append (manager_input, create_bit_string (~0UL));
        itty_bit_string_list_append (manager_input, create_bit_string (0));

        itty_bit_string_list_t *sync_output = itty_network_feed (network, sync_input);
        itty_bit_string_list_t *manager_output = itty_network_feed_with_manager (network,
                                                                                 manager_input,
                                                                                 manager);
        assert_bit_string_lists_equal (sync_output, manager_output);

        free_feed_result (sync_input, sync_output);
        free_feed_result (manager_input, manager_output);
        itty_manager_free (manager);
        itty_network_free (network);
}

static void
test_itty_network_affinity_node_failure_returns_null (void)
{
        itty_network_t *sync_network = create_too_wide_affinity_network ();
        itty_bit_string_list_t *sync_input = itty_bit_string_list_new ();
        itty_bit_string_list_append (sync_input, create_bit_string (~0UL));
        itty_bit_string_list_append (sync_input, create_bit_string (0));

        itty_bit_string_list_t *sync_output = itty_network_feed (sync_network, sync_input);
        assert (sync_output == NULL);
        itty_bit_string_list_free (sync_input);
        itty_network_free (sync_network);

        itty_network_t *manager_network = create_too_wide_affinity_network ();
        itty_bit_string_list_t *manager_input = itty_bit_string_list_new ();
        itty_manager_t *manager = itty_manager_new ();
        itty_bit_string_list_append (manager_input, create_bit_string (~0UL));
        itty_bit_string_list_append (manager_input, create_bit_string (0));

        itty_bit_string_list_t *manager_output = itty_network_feed_with_manager (manager_network,
                                                                                 manager_input,
                                                                                 manager);
        assert (manager_output == NULL);
        itty_manager_free (manager);
        itty_bit_string_list_free (manager_input);
        itty_network_free (manager_network);
}

static void
test_itty_network_select_output (void)
{
        itty_bit_string_list_t *outputs = itty_bit_string_list_new ();
        size_t index = 0;

        itty_bit_string_list_append (outputs, create_bit_string (0b0011));
        itty_bit_string_list_append (outputs, create_bit_string (0b1111));
        itty_bit_string_list_append (outputs, create_bit_string (0b0001));

        assert (itty_network_select_output (outputs, &index));
        assert (index == 1);

        itty_bit_string_list_free (outputs);
}

static void
test_itty_network_select_output_rejects_empty_outputs (void)
{
        itty_bit_string_list_t *outputs = itty_bit_string_list_new ();
        size_t index = 17;

        assert (!itty_network_select_output (outputs, &index));
        assert (index == 17);

        itty_bit_string_list_free (outputs);
}

static void
test_itty_network_layer_present_feed_layer (void)
{
        itty_bit_string_list_t *input = create_input_list ();
        itty_network_layer_t *layer = itty_network_layer_new ();
        itty_network_layer_append (layer, itty_network_feed_node_new (create_masks ()));

        char *presentation = itty_network_layer_present (layer, input);
        assert (presentation != NULL);
        assert (strstr (presentation, "network layer 0") != NULL);
        assert (strstr (presentation, "stage 0: name=network modulate inputs") != NULL);
        assert (strstr (presentation, "stage 1: name=network condense nodes") != NULL);
        assert (strstr (presentation, "stage 2: name=network double outputs") != NULL);
        assert (strstr (presentation, "name=network input") != NULL);
        assert (strstr (presentation, "name=network mask") != NULL);
        assert (strstr (presentation, "name=network modulated input") != NULL);
        assert (strstr (presentation, "name=network condensed output") != NULL);
        assert (strstr (presentation, "name=network layer output") != NULL);
        assert (strstr (presentation, "command 0: XOR") != NULL);
        assert (strstr (presentation, "command 2: CONDENSE") != NULL);
        assert (strstr (presentation, "command 3: DOUBLE") != NULL);
        free (presentation);

        itty_network_layer_free (layer);
        itty_bit_string_list_free (input);
}

static void
test_itty_network_layer_present_affinity_layer (void)
{
        itty_bit_string_list_t *input = itty_bit_string_list_new ();
        itty_network_layer_t *layer = itty_network_layer_new ();
        itty_affinity_probe_options_t options = {
                .total_votes = 7,
                .score_bit_length = ITTY_BIT_STRING_WORD_SIZE_IN_BITS
        };
        itty_bit_string_list_append (input, create_bit_string (~0UL));
        itty_bit_string_list_append (input, create_bit_string (0));
        itty_network_layer_append (layer, itty_network_affinity_node_new (create_affinity_traits (),
                                                                          create_affinity_imprints (),
                                                                          &options));

        char *presentation = itty_network_layer_present (layer, input);
        assert (presentation != NULL);
        assert (strstr (presentation, "network layer 0") != NULL);
        assert (strstr (presentation, "network affinity node 0") != NULL);
        assert (strstr (presentation, "stage 0: name=affinity match") != NULL);
        assert (strstr (presentation, "command 5: WEIGHTED_CONDENSE") != NULL);
        free (presentation);

        itty_network_layer_free (layer);
        itty_bit_string_list_free (input);
}

static void
test_itty_network_present_plan_includes_affinity_node (void)
{
        itty_network_t *network = create_affinity_network ();
        itty_bit_string_list_t *input = itty_bit_string_list_new ();
        itty_bit_string_list_append (input, create_bit_string (~0UL));
        itty_bit_string_list_append (input, create_bit_string (0));

        char *presentation = itty_network_present_plan (network, input);
        assert (presentation != NULL);
        assert (strstr (presentation, "network plan: layers=1") != NULL);
        assert (strstr (presentation, "network layer 0") != NULL);
        assert (strstr (presentation, "network affinity node 0") != NULL);
        assert (strstr (presentation, "stage 0: name=affinity match") != NULL);
        assert (strstr (presentation, "stage 1: name=affinity score") != NULL);
        assert (strstr (presentation, "stage 2: name=affinity causal mask") != NULL);
        assert (strstr (presentation, "stage 3: name=affinity output") != NULL);
        assert (strstr (presentation, "command 0: XNOR") != NULL);
        assert (strstr (presentation, "command 2: POPCOUNT") != NULL);
        assert (strstr (presentation, "command 5: WEIGHTED_CONDENSE") != NULL);
        free (presentation);

        itty_bit_string_list_free (input);
        itty_network_free (network);
}

static void
test_itty_network_present_plan (void)
{
        itty_network_t *network = create_two_layer_network ();
        itty_bit_string_list_t *input = create_input_list ();

        char *presentation = itty_network_present_plan (network, input);
        assert (presentation != NULL);
        assert (strstr (presentation, "network plan: layers=2") != NULL);
        assert (strstr (presentation, "network layer 0") != NULL);
        assert (strstr (presentation, "network layer 1") != NULL);
        assert (count_occurrences (presentation, "stage 0: name=network modulate inputs") == 2);
        assert (count_occurrences (presentation, "stage 1: name=network condense nodes") == 2);
        assert (count_occurrences (presentation, "stage 2: name=network double outputs") == 2);
        free (presentation);

        itty_bit_string_list_free (input);
        itty_network_free (network);
}

static void
test_itty_network_present_plan_empty_network (void)
{
        itty_network_t *network = itty_network_new ();
        itty_bit_string_list_t *input = create_input_list ();

        char *presentation = itty_network_present_plan (network, input);
        assert (presentation != NULL);
        assert (strcmp (presentation, "network plan: layers=0\n") == 0);
        free (presentation);

        itty_bit_string_list_free (input);
        itty_network_free (network);
}

int
main (void)
{
        test_itty_network_affinity_node_new_rejects_shape_mismatch ();
        test_itty_network_feed_with_manager ();
        test_itty_network_feed_with_manager_edge_cases ();
        test_itty_network_affinity_node_matches_affinity_probe_list ();
        test_itty_network_affinity_node_with_manager_matches_sync ();
        test_itty_network_affinity_node_failure_returns_null ();
        test_itty_network_select_output ();
        test_itty_network_select_output_rejects_empty_outputs ();
        test_itty_network_layer_present_feed_layer ();
        test_itty_network_layer_present_affinity_layer ();
        test_itty_network_present_plan_includes_affinity_node ();
        test_itty_network_present_plan ();
        test_itty_network_present_plan_empty_network ();
        printf ("All itty-network tests passed.\n");
        return 0;
}
