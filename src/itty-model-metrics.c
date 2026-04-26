#include "itty-model-metrics.h"
#include "itty-network.h"

#include <math.h>
#include <stdlib.h>

struct itty_model_metrics_activation_trace_t {
        itty_model_metrics_bit_summary_t *layer_summaries;
        size_t                            layer_count;
};

static double
itty_model_metrics_log2 (double value)
{
        return log (value) / log (2.0);
}

double
itty_model_metrics_entropy_for_counts (size_t set_bits,
                                       size_t bit_count)
{
        if (bit_count == 0 || set_bits == 0 || set_bits == bit_count)
                return 0.0;

        double set_probability = (double) set_bits / (double) bit_count;
        double unset_probability = 1.0 - set_probability;

        return -(set_probability * itty_model_metrics_log2 (set_probability) +
                 unset_probability * itty_model_metrics_log2 (unset_probability));
}

bool
itty_model_metrics_measure_bit_string (itty_bit_string_t                *bit_string,
                                       itty_model_metrics_bit_summary_t *summary)
{
        if (!bit_string || !summary)
                return false;

        size_t bit_count = itty_bit_string_get_length (bit_string);
        size_t set_bits = itty_bit_string_get_pop_count (bit_string);

        *summary = (itty_model_metrics_bit_summary_t) {
                .bit_count = bit_count,
                .set_bits = set_bits,
                .unset_bits = bit_count - set_bits,
                .set_density = bit_count == 0 ? 0.0 : (double) set_bits / (double) bit_count,
                .entropy = itty_model_metrics_entropy_for_counts (set_bits, bit_count)
        };

        return bit_count > 0;
}

bool
itty_model_metrics_measure_bit_string_list (itty_bit_string_list_t           *list,
                                            itty_model_metrics_bit_summary_t *summary)
{
        if (!list || !summary)
                return false;

        *summary = (itty_model_metrics_bit_summary_t) { 0 };

        itty_bit_string_list_iterator_t iterator;
        itty_bit_string_list_iterator_init (list, &iterator);

        itty_bit_string_t *bit_string = NULL;
        while (itty_bit_string_list_iterator_next (&iterator, &bit_string)) {
                summary->bit_count += itty_bit_string_get_length (bit_string);
                summary->set_bits += itty_bit_string_get_pop_count (bit_string);
        }

        summary->unset_bits = summary->bit_count - summary->set_bits;
        summary->set_density = summary->bit_count == 0 ? 0.0 :
                               (double) summary->set_bits / (double) summary->bit_count;
        summary->entropy = itty_model_metrics_entropy_for_counts (summary->set_bits,
                                                                  summary->bit_count);

        return summary->bit_count > 0;
}

itty_model_metrics_activation_trace_t *
itty_model_metrics_trace_network_activations (itty_network_t         *network,
                                              itty_bit_string_list_t *input,
                                              itty_manager_t         *manager)
{
        if (!network || !input)
                return NULL;

        itty_model_metrics_activation_trace_t *trace = malloc (sizeof (itty_model_metrics_activation_trace_t));
        trace->layer_summaries = NULL;
        trace->layer_count = 0;

        itty_bit_string_list_t *current_input = input;
        itty_network_iterator_t iterator;
        itty_network_iterator_init (network, &iterator);

        itty_network_layer_t *layer = NULL;
        while (itty_network_iterator_next (&iterator, &layer)) {
                itty_bit_string_list_t *layer_outputs = itty_network_layer_feed_with_manager (layer,
                                                                                              current_input,
                                                                                              manager);
                if (!layer_outputs) {
                        if (current_input != input)
                                itty_bit_string_list_free (current_input);
                        itty_model_metrics_activation_trace_free (trace);
                        return NULL;
                }

                trace->layer_summaries = realloc (trace->layer_summaries,
                                                  (trace->layer_count + 1) * sizeof (itty_model_metrics_bit_summary_t));
                itty_model_metrics_measure_bit_string_list (layer_outputs,
                                                            &trace->layer_summaries[trace->layer_count]);
                trace->layer_count++;

                if (current_input != input)
                        itty_bit_string_list_free (current_input);
                current_input = layer_outputs;
        }

        if (current_input != input)
                itty_bit_string_list_free (current_input);

        return trace;
}

void
itty_model_metrics_activation_trace_free (itty_model_metrics_activation_trace_t *trace)
{
        if (!trace)
                return;

        free (trace->layer_summaries);
        free (trace);
}

size_t
itty_model_metrics_activation_trace_get_layer_count (itty_model_metrics_activation_trace_t *trace)
{
        if (!trace)
                return 0;

        return trace->layer_count;
}

bool
itty_model_metrics_activation_trace_get_layer_summary (itty_model_metrics_activation_trace_t *trace,
                                                       size_t                                 layer_index,
                                                       itty_model_metrics_bit_summary_t      *summary)
{
        if (!trace || !summary || layer_index >= trace->layer_count)
                return false;

        *summary = trace->layer_summaries[layer_index];
        return true;
}
