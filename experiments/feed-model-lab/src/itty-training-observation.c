#include "itty-training-observation.h"
#include "itty-network.h"

#include <stdlib.h>

struct itty_training_observation_t {
        itty_model_metrics_bit_summary_t       before_masks;
        itty_model_metrics_bit_summary_t       after_masks;
        itty_model_metrics_activation_trace_t *before_activations;
        itty_model_metrics_activation_trace_t *after_activations;
        itty_training_layer_summary_t         *layer_summaries;
        itty_feed_model_train_stats_t          train_stats;
        size_t                                 layer_summary_count;
        size_t                                 before_distance;
        size_t                                 after_distance;
        bool                                   has_before_masks;
        bool                                   has_after_masks;
        bool                                   has_before_distance;
        bool                                   has_after_distance;
        bool                                   has_train_stats;
        bool                                   trained;
};

struct itty_training_history_t {
        itty_training_step_summary_t *summaries;
        size_t                        step_count;
};

struct itty_training_optimizer_t {
        itty_training_optimizer_kind_t         kind;
        size_t                                 max_flips;
        size_t                                 numerator;
        size_t                                 denominator;
        itty_feed_model_train_budget_policy_t  budget_policy;
};

static itty_model_metrics_activation_trace_t *
trace_feed_model_activations (itty_feed_model_t      *model,
                              itty_bit_string_list_t *input,
                              itty_manager_t         *manager)
{
        itty_network_t *network = itty_feed_model_build_network (model);
        itty_model_metrics_activation_trace_t *trace = itty_model_metrics_trace_network_activations (network,
                                                                                                      input,
                                                                                                      manager);
        itty_network_free (network);
        return trace;
}

static bool
fold_activation_to_target_width (itty_bit_string_t  *activation,
                                 itty_bit_string_t  *target,
                                 itty_bit_string_t **folded_activation)
{
        size_t target_words = itty_bit_string_get_number_of_words (target);
        size_t activation_words = itty_bit_string_get_number_of_words (activation);

        if (target_words == 0)
                return false;

        if (activation_words == 0) {
                itty_bit_string_t *zero_activation = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
                itty_bit_string_append_zeros (zero_activation, target_words);
                *folded_activation = zero_activation;
                return true;
        }

        itty_bit_string_t *current_activation = activation;
        while (activation_words > target_words) {
                if (activation_words % 2 != 0 || activation_words / 2 < target_words) {
                        if (current_activation != activation)
                                itty_bit_string_free (current_activation);
                        return false;
                }

                itty_bit_string_t *reduced_activation = itty_bit_string_reduce_by_half (current_activation);
                if (current_activation != activation)
                        itty_bit_string_free (current_activation);
                current_activation = reduced_activation;
                activation_words = itty_bit_string_get_number_of_words (current_activation);
        }

        *folded_activation = current_activation;
        return activation_words == target_words;
}

static bool
measure_feed_model_target_distance (itty_feed_model_t      *model,
                                    itty_bit_string_list_t *input,
                                    itty_bit_string_t      *target,
                                    itty_manager_t         *manager,
                                    size_t                 *distance)
{
        itty_network_t *network = itty_feed_model_build_network (model);
        itty_bit_string_list_t *outputs = itty_network_feed_with_manager (network,
                                                                          input,
                                                                          manager);
        itty_network_free (network);
        if (!outputs)
                return false;

        size_t selected_index = 0;
        if (!itty_network_select_output (outputs, &selected_index)) {
                itty_bit_string_list_free (outputs);
                return false;
        }

        itty_bit_string_t *activation = itty_bit_string_list_fetch (outputs,
                                                                    selected_index);
        itty_bit_string_t *folded_activation = NULL;
        if (!fold_activation_to_target_width (activation, target, &folded_activation)) {
                itty_bit_string_list_free (outputs);
                return false;
        }

        itty_bit_string_t *difference = itty_bit_string_exclusive_or (folded_activation,
                                                                      target);
        *distance = itty_bit_string_get_pop_count (difference);
        itty_bit_string_free (difference);

        if (folded_activation != activation)
                itty_bit_string_free (folded_activation);
        itty_bit_string_list_free (outputs);
        return true;
}

static itty_training_observation_t *
observe_feed_model_train (itty_feed_model_t                     *model,
                          itty_bit_string_list_t                *input,
                          itty_bit_string_t                     *target,
                          itty_feed_model_train_options_t const *options,
                          itty_manager_t                        *manager,
                          bool                                   backwards)
{
        itty_training_observation_t *observation = malloc (sizeof (itty_training_observation_t));
        *observation = (itty_training_observation_t) { 0 };

        observation->has_before_masks = itty_feed_model_measure_masks (model,
                                                                       &observation->before_masks);
        observation->before_activations = trace_feed_model_activations (model,
                                                                        input,
                                                                        manager);
        if (backwards && observation->before_activations) {
                observation->layer_summary_count = itty_model_metrics_activation_trace_get_layer_count (observation->before_activations);
                observation->layer_summaries = calloc (observation->layer_summary_count,
                                                       sizeof (itty_training_layer_summary_t));

                for (size_t layer_index = 0; layer_index < observation->layer_summary_count; layer_index++) {
                        itty_model_metrics_bit_summary_t layer_mask_summary;

                        observation->layer_summaries[layer_index].layer_index = layer_index;
                        if (itty_feed_model_measure_layer_masks (model,
                                                                 layer_index,
                                                                 &layer_mask_summary))
                                observation->layer_summaries[layer_index].before_mask_entropy = layer_mask_summary.entropy;
                }
        }
        observation->has_before_distance = measure_feed_model_target_distance (model,
                                                                               input,
                                                                               target,
                                                                               manager,
                                                                               &observation->before_distance);

        if (backwards && observation->layer_summary_count > 0) {
                itty_feed_model_train_stats_t *layer_stats = calloc (observation->layer_summary_count,
                                                                     sizeof (itty_feed_model_train_stats_t));
                observation->trained = itty_feed_model_train_backwards_one_with_layer_stats (model,
                                                                                             input,
                                                                                             target,
                                                                                             options,
                                                                                             &observation->train_stats,
                                                                                             layer_stats,
                                                                                             observation->layer_summary_count);
                if (observation->trained) {
                        for (size_t layer_index = 0; layer_index < observation->layer_summary_count; layer_index++) {
                                observation->layer_summaries[layer_index].flips = layer_stats[layer_index].flips;
                                observation->layer_summaries[layer_index].candidate_bits = layer_stats[layer_index].candidate_bits;
                                observation->layer_summaries[layer_index].largest_error = layer_stats[layer_index].largest_error;
                        }
                }
                free (layer_stats);
        } else {
                observation->trained = backwards ?
                                       itty_feed_model_train_backwards_one_with_stats (model,
                                                                                       input,
                                                                                       target,
                                                                                       options,
                                                                                       &observation->train_stats) :
                                       itty_feed_model_train_one_with_stats (model,
                                                                             input,
                                                                             target,
                                                                             options,
                                                                             &observation->train_stats);
        }
        observation->has_train_stats = observation->trained;

        if (!observation->trained)
                return observation;

        observation->has_after_masks = itty_feed_model_measure_masks (model,
                                                                      &observation->after_masks);
        observation->after_activations = trace_feed_model_activations (model,
                                                                       input,
                                                                       manager);
        for (size_t layer_index = 0; layer_index < observation->layer_summary_count; layer_index++) {
                itty_model_metrics_bit_summary_t layer_mask_summary;

                if (itty_feed_model_measure_layer_masks (model,
                                                         layer_index,
                                                         &layer_mask_summary))
                        observation->layer_summaries[layer_index].after_mask_entropy = layer_mask_summary.entropy;
        }
        observation->has_after_distance = measure_feed_model_target_distance (model,
                                                                              input,
                                                                              target,
                                                                              manager,
                                                                              &observation->after_distance);

        return observation;
}

itty_training_observation_t *
itty_training_observation_feed_model_train_one (itty_feed_model_t                     *model,
                                                itty_bit_string_list_t                *input,
                                                itty_bit_string_t                     *target,
                                                itty_feed_model_train_options_t const *options,
                                                itty_manager_t                        *manager)
{
        return observe_feed_model_train (model,
                                         input,
                                         target,
                                         options,
                                         manager,
                                         false);
}

itty_training_observation_t *
itty_training_observation_feed_model_train_backwards_one (itty_feed_model_t                     *model,
                                                          itty_bit_string_list_t                *input,
                                                          itty_bit_string_t                     *target,
                                                          itty_feed_model_train_options_t const *options,
                                                          itty_manager_t                        *manager)
{
        return observe_feed_model_train (model,
                                         input,
                                         target,
                                         options,
                                         manager,
                                         true);
}

static bool
get_trace_final_entropy (itty_model_metrics_activation_trace_t *trace,
                         double                                *entropy)
{
        if (!trace || !entropy)
                return false;

        size_t layer_count = itty_model_metrics_activation_trace_get_layer_count (trace);
        if (layer_count == 0)
                return false;

        itty_model_metrics_bit_summary_t summary;
        if (!itty_model_metrics_activation_trace_get_layer_summary (trace,
                                                                    layer_count - 1,
                                                                    &summary))
                return false;

        *entropy = summary.entropy;
        return true;
}

static itty_training_step_summary_t
summarize_observation (itty_training_observation_t *observation,
                       size_t                       step)
{
        itty_training_step_summary_t summary = {
                .step = step
        };
        itty_model_metrics_bit_summary_t mask_summary;
        itty_feed_model_train_stats_t stats;

        if (itty_training_observation_get_before_distance (observation, &summary.before_distance) == false)
                summary.before_distance = 0;
        if (itty_training_observation_get_after_distance (observation, &summary.after_distance) == false)
                summary.after_distance = 0;
        if (itty_training_observation_get_train_stats (observation, &stats)) {
                summary.flips = stats.flips;
                summary.candidate_bits = stats.candidate_bits;
                summary.largest_error = stats.largest_error;
        }
        if (itty_training_observation_get_before_mask_summary (observation, &mask_summary))
                summary.before_mask_entropy = mask_summary.entropy;
        if (itty_training_observation_get_after_mask_summary (observation, &mask_summary))
                summary.after_mask_entropy = mask_summary.entropy;
        get_trace_final_entropy (itty_training_observation_get_before_activations (observation),
                                 &summary.before_activation_entropy);
        get_trace_final_entropy (itty_training_observation_get_after_activations (observation),
                                 &summary.after_activation_entropy);

        return summary;
}

static itty_training_history_t *
training_history_new (void)
{
        itty_training_history_t *history = malloc (sizeof (itty_training_history_t));
        history->summaries = NULL;
        history->step_count = 0;
        return history;
}

static void
training_history_append (itty_training_history_t             *history,
                         itty_training_step_summary_t const *summary)
{
        history->summaries = realloc (history->summaries,
                                      (history->step_count + 1) * sizeof (itty_training_step_summary_t));
        history->summaries[history->step_count] = *summary;
        history->step_count++;
}

static itty_training_history_t *
run_feed_model_training_history (itty_feed_model_t                     *model,
                                 itty_bit_string_list_t                *input,
                                 itty_bit_string_t                     *target,
                                 itty_feed_model_train_options_t const *options,
                                 itty_manager_t                        *manager,
                                 size_t                                 max_steps,
                                 bool                                   backwards)
{
        itty_training_history_t *history = training_history_new ();

        for (size_t step = 0; step < max_steps; step++) {
                itty_training_observation_t *observation = backwards ?
                        itty_training_observation_feed_model_train_backwards_one (model,
                                                                                  input,
                                                                                  target,
                                                                                  options,
                                                                                  manager) :
                        itty_training_observation_feed_model_train_one (model,
                                                                        input,
                                                                        target,
                                                                        options,
                                                                        manager);
                if (!observation)
                        break;

                itty_training_step_summary_t summary = summarize_observation (observation,
                                                                              step);
                training_history_append (history, &summary);

                bool trained = itty_training_observation_did_train (observation);
                itty_training_observation_free (observation);

                if (!trained || summary.after_distance == 0)
                        break;
        }

        return history;
}

itty_training_history_t *
itty_training_history_feed_model_train_one (itty_feed_model_t                     *model,
                                            itty_bit_string_list_t                *input,
                                            itty_bit_string_t                     *target,
                                            itty_feed_model_train_options_t const *options,
                                            itty_manager_t                        *manager,
                                            size_t                                 max_steps)
{
        return run_feed_model_training_history (model,
                                                input,
                                                target,
                                                options,
                                                manager,
                                                max_steps,
                                                false);
}

itty_training_history_t *
itty_training_history_feed_model_train_backwards_one (itty_feed_model_t                     *model,
                                                      itty_bit_string_list_t                *input,
                                                      itty_bit_string_t                     *target,
                                                      itty_feed_model_train_options_t const *options,
                                                      itty_manager_t                        *manager,
                                                      size_t                                 max_steps)
{
        return run_feed_model_training_history (model,
                                                input,
                                                target,
                                                options,
                                                manager,
                                                max_steps,
                                                true);
}

static bool
training_optimizer_choose_options (itty_training_optimizer_t       *optimizer,
                                   size_t                           before_distance,
                                   itty_feed_model_train_options_t *options)
{
        if (!optimizer || !options)
                return false;

        size_t max_flips = optimizer->max_flips;
        if (optimizer->kind == ITTY_TRAINING_OPTIMIZER_DISTANCE_CAPPED_BUDGET &&
            before_distance < max_flips)
                max_flips = before_distance;
        if (optimizer->kind == ITTY_TRAINING_OPTIMIZER_DISTANCE_FRACTION_BUDGET) {
                size_t fraction_budget = (before_distance * optimizer->numerator + optimizer->denominator - 1) / optimizer->denominator;
                if (fraction_budget == 0 && before_distance > 0)
                        fraction_budget = 1;
                if (optimizer->max_flips != 0 && fraction_budget > optimizer->max_flips)
                        fraction_budget = optimizer->max_flips;
                max_flips = fraction_budget;
        }

        *options = (itty_feed_model_train_options_t) {
                .max_flips = max_flips,
                .budget_policy = optimizer->budget_policy
        };

        return true;
}

static itty_training_history_t *
run_feed_model_training_history_with_optimizer (itty_feed_model_t        *model,
                                                itty_bit_string_list_t   *input,
                                                itty_bit_string_t        *target,
                                                itty_training_optimizer_t *optimizer,
                                                itty_manager_t           *manager,
                                                size_t                    max_steps,
                                                bool                      backwards)
{
        itty_training_history_t *history = training_history_new ();

        for (size_t step = 0; step < max_steps; step++) {
                size_t before_distance = 0;
                itty_feed_model_train_options_t options;

                if (!measure_feed_model_target_distance (model,
                                                         input,
                                                         target,
                                                         manager,
                                                         &before_distance) ||
                    !training_optimizer_choose_options (optimizer,
                                                        before_distance,
                                                        &options))
                        break;

                itty_training_observation_t *observation = backwards ?
                        itty_training_observation_feed_model_train_backwards_one (model,
                                                                                  input,
                                                                                  target,
                                                                                  &options,
                                                                                  manager) :
                        itty_training_observation_feed_model_train_one (model,
                                                                        input,
                                                                        target,
                                                                        &options,
                                                                        manager);
                if (!observation)
                        break;

                itty_training_step_summary_t summary = summarize_observation (observation,
                                                                              step);
                training_history_append (history, &summary);

                bool trained = itty_training_observation_did_train (observation);
                itty_training_observation_free (observation);

                if (!trained || summary.after_distance == 0)
                        break;
        }

        return history;
}

itty_training_history_t *
itty_training_history_feed_model_train_one_with_optimizer (itty_feed_model_t        *model,
                                                          itty_bit_string_list_t   *input,
                                                          itty_bit_string_t        *target,
                                                          itty_training_optimizer_t *optimizer,
                                                          itty_manager_t           *manager,
                                                          size_t                    max_steps)
{
        return run_feed_model_training_history_with_optimizer (model,
                                                               input,
                                                               target,
                                                               optimizer,
                                                               manager,
                                                               max_steps,
                                                               false);
}

itty_training_history_t *
itty_training_history_feed_model_train_backwards_one_with_optimizer (itty_feed_model_t        *model,
                                                                     itty_bit_string_list_t   *input,
                                                                     itty_bit_string_t        *target,
                                                                     itty_training_optimizer_t *optimizer,
                                                                     itty_manager_t           *manager,
                                                                     size_t                    max_steps)
{
        return run_feed_model_training_history_with_optimizer (model,
                                                               input,
                                                               target,
                                                               optimizer,
                                                               manager,
                                                               max_steps,
                                                               true);
}

static itty_training_optimizer_t *
training_optimizer_new (itty_training_optimizer_kind_t kind,
                        size_t                         max_flips)
{
        itty_training_optimizer_t *optimizer = malloc (sizeof (itty_training_optimizer_t));
        optimizer->kind = kind;
        optimizer->max_flips = max_flips;
        optimizer->numerator = 1;
        optimizer->denominator = 1;
        optimizer->budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST;
        return optimizer;
}

itty_training_optimizer_t *
itty_training_optimizer_new_fixed_budget (size_t max_flips)
{
        return training_optimizer_new (ITTY_TRAINING_OPTIMIZER_FIXED_BUDGET,
                                       max_flips);
}

itty_training_optimizer_t *
itty_training_optimizer_new_distance_capped_budget (size_t max_flips)
{
        return training_optimizer_new (ITTY_TRAINING_OPTIMIZER_DISTANCE_CAPPED_BUDGET,
                                       max_flips);
}

itty_training_optimizer_t *
itty_training_optimizer_new_distance_fraction_budget (size_t numerator,
                                                      size_t denominator,
                                                      size_t max_flips)
{
        itty_training_optimizer_t *optimizer = training_optimizer_new (ITTY_TRAINING_OPTIMIZER_DISTANCE_FRACTION_BUDGET,
                                                                       max_flips);
        optimizer->numerator = numerator;
        optimizer->denominator = denominator == 0 ? 1 : denominator;
        return optimizer;
}

void
itty_training_optimizer_free (itty_training_optimizer_t *optimizer)
{
        free (optimizer);
}

itty_training_optimizer_kind_t
itty_training_optimizer_get_kind (itty_training_optimizer_t *optimizer)
{
        if (!optimizer)
                return ITTY_TRAINING_OPTIMIZER_FIXED_BUDGET;

        return optimizer->kind;
}

void
itty_training_history_free (itty_training_history_t *history)
{
        if (!history)
                return;

        free (history->summaries);
        free (history);
}

size_t
itty_training_history_get_step_count (itty_training_history_t *history)
{
        if (!history)
                return 0;

        return history->step_count;
}

bool
itty_training_history_get_step_summary (itty_training_history_t      *history,
                                        size_t                        step_index,
                                        itty_training_step_summary_t *summary)
{
        if (!history || !summary || step_index >= history->step_count)
                return false;

        *summary = history->summaries[step_index];
        return true;
}

bool
itty_training_history_summarize (itty_training_history_t         *history,
                                 itty_training_history_summary_t *summary)
{
        if (!history || !summary || history->step_count == 0)
                return false;

        *summary = (itty_training_history_summary_t) { 0 };
        summary->steps = history->step_count;

        for (size_t i = 0; i < history->step_count; i++)
                summary->total_flips += history->summaries[i].flips;

        itty_training_step_summary_t *last_step = &history->summaries[history->step_count - 1];
        summary->final_distance = last_step->after_distance;
        summary->final_mask_entropy = last_step->after_mask_entropy;
        summary->final_activation_entropy = last_step->after_activation_entropy;
        summary->reached_target = last_step->after_distance == 0;

        return true;
}

void
itty_training_observation_free (itty_training_observation_t *observation)
{
        if (!observation)
                return;

        itty_model_metrics_activation_trace_free (observation->before_activations);
        itty_model_metrics_activation_trace_free (observation->after_activations);
        free (observation->layer_summaries);
        free (observation);
}

bool
itty_training_observation_did_train (itty_training_observation_t *observation)
{
        return observation && observation->trained;
}

bool
itty_training_observation_get_before_mask_summary (itty_training_observation_t      *observation,
                                                   itty_model_metrics_bit_summary_t *summary)
{
        if (!observation || !summary || !observation->has_before_masks)
                return false;

        *summary = observation->before_masks;
        return true;
}

bool
itty_training_observation_get_after_mask_summary (itty_training_observation_t      *observation,
                                                  itty_model_metrics_bit_summary_t *summary)
{
        if (!observation || !summary || !observation->has_after_masks)
                return false;

        *summary = observation->after_masks;
        return true;
}

bool
itty_training_observation_get_train_stats (itty_training_observation_t   *observation,
                                           itty_feed_model_train_stats_t *stats)
{
        if (!observation || !stats || !observation->has_train_stats)
                return false;

        *stats = observation->train_stats;
        return true;
}

bool
itty_training_observation_get_before_distance (itty_training_observation_t *observation,
                                               size_t                      *distance)
{
        if (!observation || !distance || !observation->has_before_distance)
                return false;

        *distance = observation->before_distance;
        return true;
}

bool
itty_training_observation_get_after_distance (itty_training_observation_t *observation,
                                              size_t                      *distance)
{
        if (!observation || !distance || !observation->has_after_distance)
                return false;

        *distance = observation->after_distance;
        return true;
}

size_t
itty_training_observation_get_layer_summary_count (itty_training_observation_t *observation)
{
        if (!observation)
                return 0;

        return observation->layer_summary_count;
}

bool
itty_training_observation_get_layer_summary (itty_training_observation_t   *observation,
                                             size_t                         index,
                                             itty_training_layer_summary_t *summary)
{
        if (!observation || !summary || index >= observation->layer_summary_count)
                return false;

        *summary = observation->layer_summaries[index];
        return true;
}

itty_model_metrics_activation_trace_t *
itty_training_observation_get_before_activations (itty_training_observation_t *observation)
{
        if (!observation)
                return NULL;

        return observation->before_activations;
}

itty_model_metrics_activation_trace_t *
itty_training_observation_get_after_activations (itty_training_observation_t *observation)
{
        if (!observation)
                return NULL;

        return observation->after_activations;
}
