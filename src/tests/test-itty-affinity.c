#include "itty-affinity.h"
#include "itty-bit-string-private.h"
#include "itty-manager.h"
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

static void
test_itty_affinity_probe_exact_match_dominates (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_t *probe = create_bit_string (~0UL);

        itty_bit_string_list_append (traits, create_bit_string (~0UL));
        itty_bit_string_list_append (traits, create_bit_string (~0UL << (ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2)));
        itty_bit_string_list_append (traits, create_bit_string (0));

        itty_bit_string_list_append (imprints, create_bit_string (0b1010));
        itty_bit_string_list_append (imprints, create_bit_string (0b1100));
        itty_bit_string_list_append (imprints, create_bit_string (0b0111));

        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        assert (affinity != NULL);

        itty_bit_string_t *output = itty_affinity_probe (affinity, probe, 12);
        assert (output != NULL);
        assert (output->number_of_words == 1);
        assert (output->words[0] == 0b1010);

        itty_bit_string_free (output);
        itty_affinity_free (affinity);
        itty_bit_string_free (probe);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static void
test_itty_affinity_rejects_shape_mismatch (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();

        itty_bit_string_list_append (traits, create_bit_string (0));
        itty_bit_string_list_append (traits, create_bit_string (1));
        itty_bit_string_list_append (imprints, create_bit_string (0));

        assert (itty_affinity_new (traits, imprints) == NULL);

        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static void
test_itty_affinity_locality_breaks_content_tie (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_t *probe = create_bit_string (~0UL);

        itty_bit_string_list_append (traits, create_bit_string (~0UL << (ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2)));
        itty_bit_string_list_append (traits, create_bit_string (~0UL));

        itty_bit_string_list_append (imprints, create_bit_string (0b1010));
        itty_bit_string_list_append (imprints, create_bit_string (0b0101));

        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        assert (affinity != NULL);

        itty_bit_string_t *output = itty_affinity_probe_at (affinity, probe, 1, 5, 3);
        assert (output != NULL);
        assert (output->number_of_words == 1);
        assert (output->words[0] == 0b0101);

        itty_bit_string_free (output);
        itty_affinity_free (affinity);
        itty_bit_string_free (probe);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static void
test_itty_affinity_options_match_probe_at (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_t *probe = create_bit_string (~0UL);

        itty_bit_string_list_append (traits, create_bit_string (~0UL << (ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2)));
        itty_bit_string_list_append (traits, create_bit_string (~0UL));

        itty_bit_string_list_append (imprints, create_bit_string (0b1010));
        itty_bit_string_list_append (imprints, create_bit_string (0b0101));

        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        assert (affinity != NULL);

        itty_affinity_probe_options_t options = {
                .total_votes = 5,
                .probe_index = 1,
                .locality_window = 3
        };
        itty_bit_string_t *probe_at_output = itty_affinity_probe_at (affinity, probe, 1, 5, 3);
        itty_bit_string_t *options_output = itty_affinity_probe_with_options (affinity, probe, &options);
        assert (probe_at_output != NULL);
        assert (options_output != NULL);
        assert (itty_bit_string_compare (probe_at_output, options_output) == 0);

        itty_bit_string_free (options_output);
        itty_bit_string_free (probe_at_output);
        itty_affinity_free (affinity);
        itty_bit_string_free (probe);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static void
test_itty_affinity_causal_masks_future_traits (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_t *probe = create_bit_string (~0UL);

        itty_bit_string_list_append (traits, create_bit_string (~0UL << (ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2)));
        itty_bit_string_list_append (traits, create_bit_string (~0UL));

        itty_bit_string_list_append (imprints, create_bit_string (0b1010));
        itty_bit_string_list_append (imprints, create_bit_string (0b0101));

        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        assert (affinity != NULL);

        itty_affinity_probe_options_t options = {
                .total_votes = 5,
                .probe_index = 0
        };
        itty_bit_string_t *output = itty_affinity_probe_with_options (affinity, probe, &options);
        assert (output != NULL);
        assert (output->words[0] == 0b0101);
        itty_bit_string_free (output);

        options.causal = true;
        output = itty_affinity_probe_with_options (affinity, probe, &options);
        assert (output != NULL);
        assert (output->words[0] == 0b1010);

        itty_bit_string_free (output);
        itty_affinity_free (affinity);
        itty_bit_string_free (probe);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static void
test_itty_affinity_gray_position_score_breaks_content_tie (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_t *probe = create_bit_string (~0UL);

        itty_bit_string_list_append (traits, create_bit_string (~0UL));
        itty_bit_string_list_append (traits, create_bit_string (~0UL));
        itty_bit_string_list_append (traits, create_bit_string (~0UL));

        itty_bit_string_list_append (imprints, create_bit_string (0b1010));
        itty_bit_string_list_append (imprints, create_bit_string (0b0101));
        itty_bit_string_list_append (imprints, create_bit_string (0b1100));

        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        assert (affinity != NULL);

        itty_affinity_probe_options_t options = {
                .total_votes = 7,
                .probe_index = 2,
                .gray_position_bits = 2,
                .gray_position_weight = 2
        };
        itty_bit_string_t *output = itty_affinity_probe_with_options (affinity, probe, &options);
        assert (output != NULL);
        assert (output->words[0] == 0b1100);

        itty_bit_string_free (output);
        itty_affinity_free (affinity);
        itty_bit_string_free (probe);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static void
test_itty_affinity_scores_over_probe_length (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_t *probe = create_bit_string (0b1010);

        itty_bit_string_list_append (traits, create_bit_string (0b00001010));
        itty_bit_string_list_append (traits, create_bit_string (0b11111010));

        itty_bit_string_list_append (imprints, create_bit_string (0b0001));
        itty_bit_string_list_append (imprints, create_bit_string (0b0010));

        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        assert (affinity != NULL);

        itty_bit_string_t *output = itty_affinity_probe (affinity, probe, 5);
        assert (output != NULL);
        assert (output->words[0] == 0b0001);

        itty_bit_string_free (output);
        itty_affinity_free (affinity);
        itty_bit_string_free (probe);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static void
test_itty_affinity_allows_explicit_score_length (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_t *probe = create_bit_string (0b11001010);

        itty_bit_string_list_append (traits, create_bit_string (0b00001010));
        itty_bit_string_list_append (traits, create_bit_string (0b11001010));

        itty_bit_string_list_append (imprints, create_bit_string (0b0001));
        itty_bit_string_list_append (imprints, create_bit_string (0b0010));

        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        assert (affinity != NULL);

        itty_affinity_probe_options_t options = {
                .total_votes = 5,
                .score_bit_length = 4
        };
        itty_bit_string_t *output = itty_affinity_probe_with_options (affinity, probe, &options);
        assert (output != NULL);
        assert (output->words[0] == 0b0001);
        itty_bit_string_free (output);

        options.score_bit_length = 8;
        output = itty_affinity_probe_with_options (affinity, probe, &options);
        assert (output != NULL);
        assert (output->words[0] == 0b0010);

        itty_bit_string_free (output);
        itty_affinity_free (affinity);
        itty_bit_string_free (probe);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static void
test_itty_affinity_plan_rejects_probe_shorter_than_score_length (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_t *probe = create_bit_string (0b1010);

        itty_bit_string_list_append (traits, create_bit_string (0b1010));
        itty_bit_string_list_append (imprints, create_bit_string (0b0001));

        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        assert (affinity != NULL);

        itty_affinity_plan_t *plan = itty_affinity_plan_new (affinity, 8);
        assert (plan != NULL);
        itty_affinity_probe_options_t options = {
                .total_votes = 3
        };
        assert (itty_affinity_plan_probe (plan, probe, &options) == NULL);

        itty_affinity_plan_free (plan);
        itty_affinity_free (affinity);
        itty_bit_string_free (probe);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static void
test_itty_affinity_probe_list_uses_position_per_probe (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_list_t *probes = itty_bit_string_list_new ();

        itty_bit_string_list_append (traits, create_bit_string (~0UL << (ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2)));
        itty_bit_string_list_append (traits, create_bit_string (~0UL));

        itty_bit_string_list_append (imprints, create_bit_string (0b1010));
        itty_bit_string_list_append (imprints, create_bit_string (0b0101));

        itty_bit_string_list_append (probes, create_bit_string (~0UL));
        itty_bit_string_list_append (probes, create_bit_string (~0UL));

        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        assert (affinity != NULL);

        itty_affinity_probe_options_t options = {
                .total_votes = 5,
                .causal = true
        };
        itty_bit_string_list_t *outputs = itty_affinity_probe_list (affinity, probes, &options);
        assert (outputs != NULL);
        assert (itty_bit_string_list_get_length (outputs) == 2);

        itty_bit_string_t *first_output = itty_bit_string_list_fetch (outputs, 0);
        itty_bit_string_t *second_output = itty_bit_string_list_fetch (outputs, 1);
        assert (first_output != NULL);
        assert (second_output != NULL);
        assert (first_output->words[0] == 0b1010);
        assert (second_output->words[0] == 0b0101);

        itty_bit_string_list_free (outputs);
        itty_affinity_free (affinity);
        itty_bit_string_list_free (probes);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static void
test_itty_affinity_probe_list_with_manager_matches_sync (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_list_t *probes = itty_bit_string_list_new ();

        itty_bit_string_list_append (traits, create_bit_string (~0UL << (ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2)));
        itty_bit_string_list_append (traits, create_bit_string (~0UL));
        itty_bit_string_list_append (traits, create_bit_string (0));

        itty_bit_string_list_append (imprints, create_bit_string (0b1010));
        itty_bit_string_list_append (imprints, create_bit_string (0b0101));
        itty_bit_string_list_append (imprints, create_bit_string (0b1100));

        itty_bit_string_list_append (probes, create_bit_string (~0UL));
        itty_bit_string_list_append (probes, create_bit_string (~0UL));
        itty_bit_string_list_append (probes, create_bit_string (0));

        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        itty_manager_t *manager = itty_manager_new ();
        assert (affinity != NULL);
        assert (manager != NULL);

        itty_affinity_probe_options_t options = {
                .total_votes = 9,
                .locality_window = 3,
                .gray_position_bits = 2,
                .gray_position_weight = 1,
                .causal = true
        };
        itty_bit_string_list_t *sync_outputs = itty_affinity_probe_list (affinity, probes, &options);
        itty_bit_string_list_t *manager_outputs = itty_affinity_probe_list_with_manager (affinity,
                                                                                         probes,
                                                                                         &options,
                                                                                         manager);
        assert (sync_outputs != NULL);
        assert (manager_outputs != NULL);
        assert (itty_bit_string_list_get_length (sync_outputs) == itty_bit_string_list_get_length (manager_outputs));

        for (size_t i = 0; i < itty_bit_string_list_get_length (sync_outputs); i++) {
                itty_bit_string_t *sync_output = itty_bit_string_list_fetch (sync_outputs, i);
                itty_bit_string_t *manager_output = itty_bit_string_list_fetch (manager_outputs, i);
                assert (itty_bit_string_compare (sync_output, manager_output) == 0);
        }

        itty_bit_string_list_free (manager_outputs);
        itty_bit_string_list_free (sync_outputs);
        itty_manager_free (manager);
        itty_affinity_free (affinity);
        itty_bit_string_list_free (probes);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static void
test_itty_affinity_plan_reuses_graph_for_fixed_width_probes (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();
        itty_bit_string_t *first_probe = create_bit_string (~0UL);
        itty_bit_string_t *second_probe = create_bit_string (0);

        itty_bit_string_list_append (traits, create_bit_string (~0UL));
        itty_bit_string_list_append (traits, create_bit_string (0));

        itty_bit_string_list_append (imprints, create_bit_string (0b1010));
        itty_bit_string_list_append (imprints, create_bit_string (0b0101));

        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        assert (affinity != NULL);

        itty_affinity_plan_t *plan = itty_affinity_plan_new (affinity, 1);
        assert (plan != NULL);

        itty_affinity_probe_options_t options = {
                .total_votes = 7
        };
        itty_bit_string_t *first_output = itty_affinity_plan_probe (plan, first_probe, &options);
        itty_bit_string_t *second_output = itty_affinity_plan_probe (plan, second_probe, &options);
        assert (first_output != NULL);
        assert (second_output != NULL);
        assert (first_output->words[0] == 0b1010);
        assert (second_output->words[0] == 0b0101);

        itty_bit_string_t *direct_output = itty_affinity_probe_with_options (affinity, first_probe, &options);
        assert (direct_output != NULL);
        assert (itty_bit_string_compare (first_output, direct_output) == 0);

        itty_bit_string_free (direct_output);
        itty_bit_string_free (second_output);
        itty_bit_string_free (first_output);
        itty_affinity_plan_free (plan);
        itty_affinity_free (affinity);
        itty_bit_string_free (second_probe);
        itty_bit_string_free (first_probe);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

static void
test_itty_affinity_plan_present_shows_staged_graph (void)
{
        itty_bit_string_list_t *traits = itty_bit_string_list_new ();
        itty_bit_string_list_t *imprints = itty_bit_string_list_new ();

        itty_bit_string_list_append (traits, create_bit_string (0b1010));
        itty_bit_string_list_append (traits, create_bit_string (0b0101));

        itty_bit_string_list_append (imprints, create_bit_string (0b1100));
        itty_bit_string_list_append (imprints, create_bit_string (0b0011));

        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        assert (affinity != NULL);

        itty_affinity_plan_t *plan = itty_affinity_plan_new (affinity, 4);
        assert (plan != NULL);

        char *presentation = itty_affinity_plan_present (plan);
        assert (presentation != NULL);
        assert (strstr (presentation, "exec_buffer descriptors=12 stages=4 commands=6") != NULL);
        assert (strstr (presentation, "descriptor 0: name=affinity probe") != NULL);
        assert (strstr (presentation, "name=affinity scores") != NULL);
        assert (strstr (presentation, "name=affinity trait") != NULL);
        assert (strstr (presentation, "name=affinity match") != NULL);
        assert (strstr (presentation, "name=affinity output") != NULL);
        assert (strstr (presentation, "name=affinity imprint") != NULL);
        assert (strstr (presentation, "name=affinity votes") != NULL);
        assert (strstr (presentation, "name=affinity causal clear start") != NULL);
        assert (strstr (presentation, "name=affinity causal clear count") != NULL);
        assert (strstr (presentation, "stage 0: name=affinity match first_command=0 command_count=2") != NULL);
        assert (strstr (presentation, "stage 1: name=affinity score first_command=2 command_count=2") != NULL);
        assert (strstr (presentation, "stage 2: name=affinity causal mask first_command=4 command_count=1") != NULL);
        assert (strstr (presentation, "stage 3: name=affinity output first_command=5 command_count=1") != NULL);
        assert (strstr (presentation, "command 0: XNOR") != NULL);
        assert (strstr (presentation, "command 2: POPCOUNT") != NULL);
        assert (strstr (presentation, "command 4: CLEAR_ARRAY_RANGE") != NULL);
        assert (strstr (presentation, "command 5: WEIGHTED_CONDENSE") != NULL);

        free (presentation);
        itty_affinity_plan_free (plan);
        itty_affinity_free (affinity);
        itty_bit_string_list_free (imprints);
        itty_bit_string_list_free (traits);
}

int
main (void)
{
        test_itty_affinity_probe_exact_match_dominates ();
        test_itty_affinity_rejects_shape_mismatch ();
        test_itty_affinity_locality_breaks_content_tie ();
        test_itty_affinity_options_match_probe_at ();
        test_itty_affinity_causal_masks_future_traits ();
        test_itty_affinity_gray_position_score_breaks_content_tie ();
        test_itty_affinity_scores_over_probe_length ();
        test_itty_affinity_allows_explicit_score_length ();
        test_itty_affinity_plan_rejects_probe_shorter_than_score_length ();
        test_itty_affinity_probe_list_uses_position_per_probe ();
        test_itty_affinity_probe_list_with_manager_matches_sync ();
        test_itty_affinity_plan_reuses_graph_for_fixed_width_probes ();
        test_itty_affinity_plan_present_shows_staged_graph ();
        printf ("All itty-affinity tests passed.\n");
        return 0;
}
