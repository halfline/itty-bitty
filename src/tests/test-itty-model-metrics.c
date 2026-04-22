#include "itty-bit-string.h"
#include "itty-bit-string-list.h"
#include "itty-feed-model.h"
#include "itty-model-metrics.h"
#include "itty-network.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

static itty_bit_string_t *
create_bit_string (size_t word)
{
        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        itty_bit_string_append_word (bit_string, word);
        return bit_string;
}

static size_t
create_half_populated_word (void)
{
        return ((size_t) 1 << (ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2)) - 1;
}

static void
assert_near (double actual,
             double expected)
{
        assert (fabs (actual - expected) < 0.000001);
}

static void
test_itty_model_metrics_entropy_for_counts (void)
{
        assert_near (itty_model_metrics_entropy_for_counts (0, 64), 0.0);
        assert_near (itty_model_metrics_entropy_for_counts (64, 64), 0.0);
        assert_near (itty_model_metrics_entropy_for_counts (32, 64), 1.0);
}

static void
test_itty_model_metrics_measure_bit_string (void)
{
        itty_bit_string_t *bit_string = create_bit_string (create_half_populated_word ());
        itty_model_metrics_bit_summary_t summary;

        assert (itty_model_metrics_measure_bit_string (bit_string, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert (summary.set_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2);
        assert (summary.unset_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2);
        assert_near (summary.set_density, 0.5);
        assert_near (summary.entropy, 1.0);

        itty_bit_string_free (bit_string);
}

static void
test_itty_model_metrics_measure_bit_string_list (void)
{
        itty_bit_string_list_t *list = itty_bit_string_list_new ();
        itty_model_metrics_bit_summary_t summary;

        itty_bit_string_list_append (list, create_bit_string (0));
        itty_bit_string_list_append (list, create_bit_string ((size_t) -1));

        assert (itty_model_metrics_measure_bit_string_list (list, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS * 2);
        assert (summary.set_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert (summary.unset_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert_near (summary.set_density, 0.5);
        assert_near (summary.entropy, 1.0);

        itty_bit_string_list_free (list);
}

static void
test_itty_feed_model_measure_masks (void)
{
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *input = itty_bit_string_list_new ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_model_metrics_bit_summary_t summary;

        itty_bit_string_list_append (input, create_bit_string (0));

        assert (itty_feed_model_measure_masks (model, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert (summary.set_bits == 0);
        assert_near (summary.entropy, 0.0);

        assert (itty_feed_model_train_one (model, input, target));

        assert (itty_feed_model_measure_layer_masks (model, 0, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert (summary.set_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2);
        assert_near (summary.entropy, 1.0);

        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static itty_network_t *
create_two_layer_feed_network (void)
{
        itty_network_t *network = itty_network_new ();

        for (size_t i = 0; i < 2; i++) {
                itty_network_layer_t *layer = itty_network_layer_new ();
                itty_bit_string_list_t *masks = itty_bit_string_list_new ();

                itty_bit_string_list_append (masks, create_bit_string (0));
                itty_network_layer_append (layer, itty_network_feed_node_new (masks));
                itty_network_append (network, layer);
        }

        return network;
}

static void
test_itty_model_metrics_traces_network_activations (void)
{
        itty_network_t *network = create_two_layer_feed_network ();
        itty_bit_string_list_t *input = itty_bit_string_list_new ();
        itty_model_metrics_bit_summary_t summary;

        itty_bit_string_list_append (input, create_bit_string (create_half_populated_word ()));

        itty_model_metrics_activation_trace_t *trace = itty_model_metrics_trace_network_activations (network,
                                                                                                      input,
                                                                                                      NULL);
        assert (trace != NULL);
        assert (itty_model_metrics_activation_trace_get_layer_count (trace) == 2);

        assert (itty_model_metrics_activation_trace_get_layer_summary (trace, 0, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS * 2);
        assert (summary.set_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert_near (summary.set_density, 0.5);
        assert_near (summary.entropy, 1.0);

        assert (itty_model_metrics_activation_trace_get_layer_summary (trace, 1, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS * 4);
        assert (summary.set_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS * 2);
        assert_near (summary.set_density, 0.5);
        assert_near (summary.entropy, 1.0);

        itty_model_metrics_activation_trace_free (trace);
        itty_bit_string_list_free (input);
        itty_network_free (network);
}

int
main (void)
{
        test_itty_model_metrics_entropy_for_counts ();
        test_itty_model_metrics_measure_bit_string ();
        test_itty_model_metrics_measure_bit_string_list ();
        test_itty_feed_model_measure_masks ();
        test_itty_model_metrics_traces_network_activations ();
        printf ("All itty-model-metrics tests passed.\n");
        return 0;
}
