#pragma once

#include "itty-bit-string.h"
#include "itty-bit-string-list.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct itty_manager_t itty_manager_t;
typedef struct itty_network_t itty_network_t;
typedef struct itty_model_metrics_activation_trace_t itty_model_metrics_activation_trace_t;

typedef struct {
        size_t bit_count;
        size_t set_bits;
        size_t unset_bits;
        double set_density;
        double entropy;
} itty_model_metrics_bit_summary_t;

double itty_model_metrics_entropy_for_counts (size_t set_bits,
                                              size_t bit_count);
bool itty_model_metrics_measure_bit_string (itty_bit_string_t                  *bit_string,
                                            itty_model_metrics_bit_summary_t   *summary);
bool itty_model_metrics_measure_bit_string_list (itty_bit_string_list_t             *list,
                                                 itty_model_metrics_bit_summary_t   *summary);
itty_model_metrics_activation_trace_t *itty_model_metrics_trace_network_activations (itty_network_t         *network,
                                                                                     itty_bit_string_list_t *input,
                                                                                     itty_manager_t         *manager);
void itty_model_metrics_activation_trace_free (itty_model_metrics_activation_trace_t *trace);
size_t itty_model_metrics_activation_trace_get_layer_count (itty_model_metrics_activation_trace_t *trace);
bool itty_model_metrics_activation_trace_get_layer_summary (itty_model_metrics_activation_trace_t *trace,
                                                            size_t                                 layer_index,
                                                            itty_model_metrics_bit_summary_t      *summary);
