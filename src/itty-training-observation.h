#pragma once

#include "itty-feed-model.h"
#include "itty-manager.h"
#include "itty-model-metrics.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct itty_training_observation_t itty_training_observation_t;
typedef struct itty_training_history_t itty_training_history_t;
typedef struct itty_training_optimizer_t itty_training_optimizer_t;

typedef enum {
        ITTY_TRAINING_OPTIMIZER_FIXED_BUDGET,
        ITTY_TRAINING_OPTIMIZER_DISTANCE_CAPPED_BUDGET,
        ITTY_TRAINING_OPTIMIZER_DISTANCE_FRACTION_BUDGET,
} itty_training_optimizer_kind_t;

typedef struct {
        size_t step;
        size_t before_distance;
        size_t after_distance;
        size_t flips;
        size_t candidate_bits;
        size_t largest_error;
        double before_mask_entropy;
        double after_mask_entropy;
        double before_activation_entropy;
        double after_activation_entropy;
} itty_training_step_summary_t;

typedef struct {
        size_t steps;
        size_t total_flips;
        size_t final_distance;
        double final_mask_entropy;
        double final_activation_entropy;
        bool   reached_target;
} itty_training_history_summary_t;

typedef struct {
        size_t layer_index;
        size_t flips;
        size_t candidate_bits;
        size_t largest_error;
        double before_mask_entropy;
        double after_mask_entropy;
} itty_training_layer_summary_t;

itty_training_observation_t *itty_training_observation_feed_model_train_one (itty_feed_model_t                     *model,
                                                                             itty_bit_string_list_t                *input,
                                                                             itty_bit_string_t                     *target,
                                                                             itty_feed_model_train_options_t const *options,
                                                                             itty_manager_t                        *manager);
itty_training_observation_t *itty_training_observation_feed_model_train_backwards_one (itty_feed_model_t                     *model,
                                                                                       itty_bit_string_list_t                *input,
                                                                                       itty_bit_string_t                     *target,
                                                                                       itty_feed_model_train_options_t const *options,
                                                                                       itty_manager_t                        *manager);
void itty_training_observation_free (itty_training_observation_t *observation);

itty_training_history_t *itty_training_history_feed_model_train_one (itty_feed_model_t                     *model,
                                                                     itty_bit_string_list_t                *input,
                                                                     itty_bit_string_t                     *target,
                                                                     itty_feed_model_train_options_t const *options,
                                                                     itty_manager_t                        *manager,
                                                                     size_t                                 max_steps);
itty_training_history_t *itty_training_history_feed_model_train_backwards_one (itty_feed_model_t                     *model,
                                                                               itty_bit_string_list_t                *input,
                                                                               itty_bit_string_t                     *target,
                                                                               itty_feed_model_train_options_t const *options,
                                                                               itty_manager_t                        *manager,
                                                                               size_t                                 max_steps);
itty_training_history_t *itty_training_history_feed_model_train_one_with_optimizer (itty_feed_model_t        *model,
                                                                                   itty_bit_string_list_t   *input,
                                                                                   itty_bit_string_t        *target,
                                                                                   itty_training_optimizer_t *optimizer,
                                                                                   itty_manager_t           *manager,
                                                                                   size_t                    max_steps);
itty_training_history_t *itty_training_history_feed_model_train_backwards_one_with_optimizer (itty_feed_model_t        *model,
                                                                                             itty_bit_string_list_t   *input,
                                                                                             itty_bit_string_t        *target,
                                                                                             itty_training_optimizer_t *optimizer,
                                                                                             itty_manager_t           *manager,
                                                                                             size_t                    max_steps);
void itty_training_history_free (itty_training_history_t *history);
size_t itty_training_history_get_step_count (itty_training_history_t *history);
bool itty_training_history_get_step_summary (itty_training_history_t       *history,
                                             size_t                         step_index,
                                             itty_training_step_summary_t  *summary);
bool itty_training_history_summarize (itty_training_history_t          *history,
                                      itty_training_history_summary_t  *summary);

itty_training_optimizer_t *itty_training_optimizer_new_fixed_budget (size_t max_flips);
itty_training_optimizer_t *itty_training_optimizer_new_distance_capped_budget (size_t max_flips);
itty_training_optimizer_t *itty_training_optimizer_new_distance_fraction_budget (size_t numerator,
                                                                                 size_t denominator,
                                                                                 size_t max_flips);
void itty_training_optimizer_free (itty_training_optimizer_t *optimizer);
itty_training_optimizer_kind_t itty_training_optimizer_get_kind (itty_training_optimizer_t *optimizer);

bool itty_training_observation_did_train (itty_training_observation_t *observation);
bool itty_training_observation_get_before_mask_summary (itty_training_observation_t       *observation,
                                                        itty_model_metrics_bit_summary_t *summary);
bool itty_training_observation_get_after_mask_summary (itty_training_observation_t       *observation,
                                                       itty_model_metrics_bit_summary_t *summary);
bool itty_training_observation_get_train_stats (itty_training_observation_t    *observation,
                                                itty_feed_model_train_stats_t  *stats);
bool itty_training_observation_get_before_distance (itty_training_observation_t *observation,
                                                    size_t                      *distance);
bool itty_training_observation_get_after_distance (itty_training_observation_t *observation,
                                                   size_t                      *distance);
size_t itty_training_observation_get_layer_summary_count (itty_training_observation_t *observation);
bool itty_training_observation_get_layer_summary (itty_training_observation_t     *observation,
                                                  size_t                           index,
                                                  itty_training_layer_summary_t   *summary);
itty_model_metrics_activation_trace_t *itty_training_observation_get_before_activations (itty_training_observation_t *observation);
itty_model_metrics_activation_trace_t *itty_training_observation_get_after_activations (itty_training_observation_t *observation);
