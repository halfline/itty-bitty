#include "itty-feed-model.h"
#include "itty-bit-string-private.h"
#include "itty-bit-string-list-private.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
        size_t bit_index;
        size_t error_weight;
        bool   target_bit;
} itty_feed_model_training_candidate_t;

typedef struct {
        size_t state;
} itty_feed_model_random_t;

typedef struct itty_feed_model_layer_state_snapshot_t itty_feed_model_layer_state_snapshot_t;

typedef struct {
        size_t final_node;
        size_t condensed_bit;
        size_t output_bit;
        size_t decoded_bit;
        size_t quota_size;
        bool   value;
} itty_feed_model_final_repair_t;

typedef struct {
        itty_feed_model_final_repair_t *items;
        size_t                          count;
        size_t                          capacity;
} itty_feed_model_final_repair_list_t;

typedef struct {
        size_t layer_node;
        size_t bit_index;
        bool   value;
} itty_feed_model_layer_assignment_t;

typedef struct {
        itty_feed_model_layer_assignment_t *items;
        size_t                              count;
        size_t                              capacity;
} itty_feed_model_layer_assignment_list_t;

typedef struct {
        size_t layer_node;
        size_t input_index;
        size_t bit_index;
} itty_feed_model_mask_flip_t;

typedef struct {
        itty_feed_model_mask_flip_t *items;
        size_t                       count;
        size_t                       capacity;
} itty_feed_model_mask_flip_list_t;

typedef struct {
        itty_feed_model_mask_flip_t flip;
        size_t                      cost;
        bool                        unsafe;
} itty_feed_model_scored_mask_flip_t;

typedef struct {
        size_t layer_index;
        size_t layer_node;
        size_t input_index;
        size_t bit_index;
        bool   new_mask_bit;
        size_t bad_count;
        size_t harmless_count;
        size_t damaged_bits;
        size_t helped_decoded_bit;
} itty_feed_model_bad_flip_frequency_t;

typedef struct {
        itty_feed_model_bad_flip_frequency_t *items;
        size_t                                count;
        size_t                                capacity;
} itty_feed_model_bad_flip_frequency_list_t;

typedef struct {
        size_t layer_index;
        size_t layer_node;
        size_t input_index;
        size_t bit_index;
        bool   desired_value;
        bool   zero_protection_taboo;
        bool   one_protection_taboo;
} itty_feed_model_replay_taboo_entry_t;

typedef struct {
        itty_feed_model_replay_taboo_entry_t *items;
        size_t                                count;
        size_t                                capacity;
} itty_feed_model_replay_taboo_map_t;

static size_t itty_feed_model_count_selected_segment_votes (itty_bit_string_list_t *outputs,
                                                            size_t                  selected_node,
                                                            size_t                  decoded_bit,
                                                            size_t                  target_bit_capacity);
static size_t itty_feed_model_count_selected_condensed_votes (itty_feed_model_t      *model,
                                                              itty_bit_string_list_t *outputs,
                                                              size_t                  layer_index,
                                                              size_t                  selected_node,
                                                              size_t                  decoded_bit,
                                                              size_t                  target_bit_capacity);
static size_t itty_feed_model_trace_expanded_bit_to_layer (itty_feed_model_t *model,
                                                           size_t             layer_index,
                                                           size_t             bit_index);
static bool itty_feed_model_collect_final_repairs (itty_feed_model_t                  *model,
                                                   itty_bit_string_list_t             *final_layer_input,
                                                   itty_bit_string_t                  *target,
                                                   itty_feed_model_final_repair_list_t *repairs);
static bool itty_feed_model_apply_contender_restore_current_state (itty_feed_model_t                           *model,
                                                                   itty_bit_string_list_t                      *first_input,
                                                                   itty_bit_string_t                           *first_target,
                                                                   itty_bit_string_list_t                      *second_input,
                                                                   itty_bit_string_t                           *second_target,
                                                                   itty_feed_model_train_options_t const      *options,
                                                                   bool                                         preserve_second,
                                                                   itty_feed_model_contender_restore_summary_t *summary);
static bool itty_feed_model_apply_finish_damage_set_restore_current_state (itty_feed_model_t                      *model,
                                                                           itty_bit_string_list_t                 *first_input,
                                                                           itty_bit_string_t                      *first_target,
                                                                           itty_bit_string_list_t                 *second_input,
                                                                           itty_bit_string_t                      *second_target,
                                                                           itty_feed_model_train_options_t const *options,
                                                                           itty_feed_model_decoder_objective_t const *a_before,
                                                                           itty_feed_model_decoder_objective_t const *b_before,
                                                                           itty_feed_model_finish_candidate_trace_t *trace,
                                                                           itty_feed_model_mask_flip_trace_t const *finish_mask_flips,
                                                                           size_t                                   finish_mask_flip_count,
                                                                           size_t                                *damaged_bit_count,
                                                                           size_t                                *flip_count,
                                                                           itty_feed_model_decoder_objective_t   *a_after,
                                                                           itty_feed_model_decoder_objective_t   *b_after,
                                                                           bool                                  *distance_preserved,
                                                                           bool                                  *progress_preserved);
static size_t itty_feed_model_collect_layer_mask_flip_traces (itty_feed_model_t                            *model,
                                                              size_t                                        layer_index,
                                                              itty_feed_model_layer_state_snapshot_t const *before_snapshot,
                                                              itty_feed_model_mask_flip_trace_t            *traces,
                                                              size_t                                        trace_limit);
static size_t itty_feed_model_count_mask_flip_trace_overlap (itty_feed_model_mask_flip_trace_t const *a_traces,
                                                             size_t                                   a_count,
                                                             itty_feed_model_mask_flip_trace_t const *b_traces,
                                                             size_t                                   b_count,
                                                             bool                                     require_same_direction);
static void itty_feed_model_set_mask_trace_value (itty_feed_model_t const                 *model,
                                                  itty_bit_string_list_t                  *masks,
                                                  itty_feed_model_mask_flip_trace_t const *trace);

typedef enum {
        ITTY_FEED_MODEL_PROJECTED_REPAIR_STRICT_DISTANCE,
        ITTY_FEED_MODEL_PROJECTED_REPAIR_BLOCKER,
        ITTY_FEED_MODEL_PROJECTED_REPAIR_OBJECTIVE,
} itty_feed_model_projected_repair_rank_t;

typedef struct {
        itty_feed_model_layer_assignment_list_t  output_assignments;
        itty_feed_model_layer_assignment_list_t  condensed_assignments;
        itty_feed_model_decoder_objective_t      objective;
        size_t                                   final_node;
        size_t                                   final_output_bit;
        size_t                                   original_index;
        size_t                                   decoded_bit;
        size_t                                   quota_size;
        size_t                                   distance_delta;
        size_t                                   false_negative_count_delta;
        size_t                                   blocker_delta;
        size_t                                   vote_deficit_delta;
        size_t                                   target_one_margin_delta;
        size_t                                   estimated_flips;
        size_t                                   residual_enable_flips;
        size_t                                   residual_mask_flips;
        size_t                                   already_satisfied_bits;
        size_t                                   bits_needing_flips;
        size_t                                   available_flippable_votes;
        size_t                                   replay_collateral_cost;
        bool                                     use_residual;
        bool                                     prefer_blocker_efficiency;
        itty_feed_model_projected_repair_rank_t  rank;
} itty_feed_model_projected_repair_candidate_t;

typedef struct {
        size_t condensed_bit;
        size_t output_bit;
        size_t estimated_flips;
        size_t replay_protection_penalty;
        size_t replay_taboo_penalty;
        size_t original_index;
} itty_feed_model_quota_vote_candidate_t;

typedef enum {
        ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_EQUAL,
        ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_WEIGHTED,
        ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_GATED,
} itty_feed_model_downstream_request_mode_t;

typedef struct {
        size_t             distance;
        size_t             selected_index;
        itty_bit_string_t *folded_activation;
} itty_feed_model_output_evaluation_t;

struct itty_feed_model_t {
        itty_bit_string_list_t ***masks_by_layer_node;
        itty_bit_string_list_t  **input_route_adapters;
        itty_bit_string_t      ***residual_enable_by_layer_node;
        itty_bit_string_t      ***residual_mask_by_layer_node;
        size_t                    number_of_layers;
        size_t                    nodes_per_layer;
        size_t                    inputs_per_node;
        size_t                    vocabulary_words;
        size_t                    within_node_condense_threshold_override;
        bool                      residual_merge_enabled;
        bool                      input_route_adapter_enabled;
        bool                     *gray_offset_override_enabled_by_route;
        size_t                   *gray_offset_override_payload_start_by_route;
        size_t                   *rotations_by_layer;
        itty_feed_model_decoder_t decoder;
};

static itty_bit_string_t *itty_feed_model_run_node_condensed (itty_bit_string_list_t *input,
                                                              itty_bit_string_list_t *masks);
static itty_bit_string_t *itty_feed_model_run_node_condensed_with_threshold (itty_bit_string_list_t *input,
                                                                             itty_bit_string_list_t *masks,
                                                                             size_t                  threshold_override);
static itty_bit_string_t *itty_feed_model_expand_condensed_output (itty_bit_string_t *condensed_output,
                                                                   size_t             rotation);
static bool itty_feed_model_count_segment_votes_for_layer (itty_feed_model_t  *model,
                                                           itty_bit_string_t  *desired_output,
                                                           size_t              layer_index,
                                                           size_t            **set_votes,
                                                           size_t             *layer_words,
                                                           size_t             *layer_bit_capacity,
                                                           size_t             *votes_per_bit);
static itty_bit_string_t *itty_feed_model_make_majority_target_from_votes (size_t const *set_votes,
                                                                           size_t        layer_words,
                                                                           size_t        layer_bit_capacity,
                                                                           size_t        votes_per_bit);
static bool itty_feed_model_count_segment_partition_votes_for_layer (itty_feed_model_t  *model,
                                                                     itty_bit_string_t  *desired_output,
                                                                     size_t              layer_index,
                                                                     size_t              partition_count,
                                                                     size_t            **set_votes_by_partition,
                                                                     size_t            **vote_counts_by_partition,
                                                                     size_t             *layer_words,
                                                                     size_t             *layer_bit_capacity);
static itty_bit_string_t *itty_feed_model_make_node_target_for_layer (itty_feed_model_t                              *model,
                                                                      itty_bit_string_list_t                         *layer_input,
                                                                      itty_bit_string_t                              *target,
                                                                      itty_bit_string_t                              *desired_output,
                                                                      itty_feed_model_suffix_oracle_options_t const   *options);
static bool itty_feed_model_evaluate_decoder_objective (itty_feed_model_t                   *model,
                                                        itty_bit_string_list_t              *outputs,
                                                        itty_bit_string_t                   *target,
                                                        itty_feed_model_decoder_objective_t *objective);
static itty_bit_string_list_t *itty_feed_model_run_outputs (itty_feed_model_t      *model,
                                                            itty_bit_string_list_t *input);
static bool itty_feed_model_decoder_objective_is_better (itty_feed_model_decoder_objective_t const *candidate,
                                                         itty_feed_model_decoder_objective_t const *baseline);
static bool itty_feed_model_evaluate_suffix_decoder_objective (itty_feed_model_t                   *model,
                                                               itty_bit_string_list_t              *layer_outputs,
                                                               size_t                               layer_index,
                                                               itty_bit_string_t                   *target,
                                                               itty_feed_model_decoder_objective_t *objective);
static itty_feed_model_layer_state_snapshot_t *itty_feed_model_snapshot_layer_state (itty_feed_model_t *model,
                                                                                    size_t             layer_index);
static void itty_feed_model_free_layer_state_snapshot (itty_feed_model_t                      *model,
                                                       itty_feed_model_layer_state_snapshot_t *snapshot);
static void itty_feed_model_restore_layer_state_snapshot (itty_feed_model_t                      *model,
                                                          size_t                                  layer_index,
                                                          itty_feed_model_layer_state_snapshot_t *snapshot);
static bool itty_feed_model_measure_replay_examples (itty_feed_model_t                         *model,
                                                     itty_feed_model_replay_example_t const    *replay_examples,
                                                     size_t                                     replay_example_count,
                                                     itty_feed_model_decoder_objective_t       *objectives,
                                                     itty_bit_string_t                        **folded_outputs);
static bool itty_feed_model_score_replay_after_batch (itty_feed_model_t                                      *model,
                                                      itty_feed_model_replay_example_t const                 *replay_examples,
                                                      size_t                                                  replay_example_count,
                                                      bool                                                    strict_replay_guard,
                                                      itty_feed_model_decoder_objective_t const              *before_objectives,
                                                      itty_bit_string_t                                     **before_folded_outputs,
                                                      itty_feed_model_refreshed_projected_repair_round_t     *round_stats);
static bool itty_feed_model_evaluate_replay_example (itty_feed_model_t                      *model,
                                                     itty_feed_model_replay_example_t const *example,
                                                     itty_feed_model_decoder_objective_t    *objective,
                                                     itty_bit_string_t                     **folded_activation);
static void itty_feed_model_accumulate_replay_transition (itty_feed_model_replay_transition_matrix_t *matrix,
                                                         itty_bit_string_t                          *target,
                                                         itty_bit_string_t                          *before_folded,
                                                         itty_bit_string_t                          *after_folded);
static bool itty_feed_model_output_transform_swaps_bit (itty_feed_model_output_transform_t transform,
                                                        size_t                             bit_index,
                                                        size_t                             bit_capacity);
static void itty_feed_model_apply_output_transform (itty_bit_string_list_t            *outputs,
                                                    itty_feed_model_output_transform_t transform);
static bool itty_feed_model_decoder_uses_segment_style (itty_feed_model_decoder_t decoder);
static bool itty_feed_model_decoder_uses_direct_duplicated_target (itty_feed_model_decoder_t decoder);
static size_t itty_feed_model_margin_preserving_vote_for_bit_string (itty_bit_string_t *bit_string);
static bool itty_feed_model_fold_activation_with_segment_weighted_condense (itty_bit_string_t  *activation,
                                                                            itty_bit_string_t  *target,
                                                                            itty_bit_string_t **folded_activation);
static itty_bit_string_list_t *itty_feed_model_apply_input_route_adapter (itty_feed_model_t      *model,
                                                                          itty_bit_string_list_t *input,
                                                                          size_t                  route_index);
static bool itty_feed_model_decoder_uses_segment_style (itty_feed_model_decoder_t decoder);
static size_t itty_feed_model_margin_preserving_vote_for_bit_string (itty_bit_string_t *bit_string);
static bool itty_feed_model_fold_activation_with_segment_weighted_condense (itty_bit_string_t  *activation,
                                                                            itty_bit_string_t  *target,
                                                                            itty_bit_string_t **folded_activation);

static int
compare_training_candidates_descending (const void *a,
                                        const void *b)
{
        itty_feed_model_training_candidate_t const *candidate_a = a;
        itty_feed_model_training_candidate_t const *candidate_b = b;

        if (candidate_a->error_weight > candidate_b->error_weight)
                return -1;
        if (candidate_a->error_weight < candidate_b->error_weight)
                return 1;
        if (candidate_a->bit_index < candidate_b->bit_index)
                return -1;
        if (candidate_a->bit_index > candidate_b->bit_index)
                return 1;
        return 0;
}

static bool
itty_feed_model_decoder_uses_segment_style (itty_feed_model_decoder_t decoder)
{
        return decoder == ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE ||
               decoder == ITTY_FEED_MODEL_DECODER_SEGMENT_WEIGHTED_CONDENSE;
}

static bool
itty_feed_model_decoder_uses_direct_duplicated_target (itty_feed_model_decoder_t decoder)
{
        return decoder == ITTY_FEED_MODEL_DECODER_DIRECT_DUPLICATED_TARGET;
}

static bool
itty_feed_model_decoder_uses_direct_padded_target (itty_feed_model_decoder_t decoder)
{
        return decoder == ITTY_FEED_MODEL_DECODER_DIRECT_PADDED_TARGET;
}

static bool
itty_feed_model_decoder_uses_direct_gray_offset_target (itty_feed_model_decoder_t decoder)
{
        return decoder == ITTY_FEED_MODEL_DECODER_DIRECT_GRAY_OFFSET_TARGET;
}

static bool
itty_feed_model_decoder_uses_direct_activation_target (itty_feed_model_decoder_t decoder)
{
        return itty_feed_model_decoder_uses_direct_duplicated_target (decoder) ||
               itty_feed_model_decoder_uses_direct_padded_target (decoder) ||
               itty_feed_model_decoder_uses_direct_gray_offset_target (decoder);
}

static size_t
itty_feed_model_gray_code (size_t value)
{
        return value ^ (value >> 1);
}

static size_t
itty_feed_model_gray_offset_payload_start (size_t activation_bit_capacity,
                                           size_t payload_bit_capacity,
                                           size_t selected_index)
{
        size_t prefix_bits = 8;
        size_t min_start = prefix_bits;
        size_t max_start;

        if (activation_bit_capacity <= payload_bit_capacity ||
            activation_bit_capacity <= prefix_bits)
                return 0;

        max_start = activation_bit_capacity - payload_bit_capacity;
        if (max_start <= min_start)
                return min_start;

        return min_start + ((selected_index * (max_start - min_start)) / 7);
}

static size_t
itty_feed_model_gray_offset_distance_for_start (itty_bit_string_t *activation,
                                                itty_bit_string_t *target,
                                                size_t             payload_start)
{
        size_t activation_bit_capacity = itty_bit_string_get_length (activation);
        size_t payload_bit_capacity = itty_bit_string_get_length (target);
        size_t gray_offset = itty_feed_model_gray_code (payload_start & 0xff);
        size_t distance = 0;

        for (size_t bit_index = 0; bit_index < activation_bit_capacity; bit_index++) {
                bool target_bit = false;

                if (bit_index < 8) {
                        target_bit = ((gray_offset >> bit_index) & 1U) != 0;
                } else if (bit_index >= payload_start &&
                           bit_index < payload_start + payload_bit_capacity) {
                        target_bit = itty_bit_string_get_bit (target,
                                                              bit_index - payload_start);
                }

                if (itty_bit_string_get_bit (activation, bit_index) != target_bit)
                        distance++;
        }

        return distance;
}

static size_t
itty_feed_model_find_best_gray_offset_payload_start (itty_bit_string_t *activation,
                                                     itty_bit_string_t *target,
                                                     size_t             selected_index)
{
        size_t activation_bit_capacity = itty_bit_string_get_length (activation);
        size_t payload_bit_capacity = itty_bit_string_get_length (target);
        size_t min_start = 8;
        size_t max_start;
        size_t best_start;
        size_t best_distance;

        if (activation_bit_capacity <= payload_bit_capacity ||
            activation_bit_capacity <= min_start)
                return 0;

        max_start = activation_bit_capacity - payload_bit_capacity;
        if (max_start <= min_start)
                return min_start;

        best_start = itty_feed_model_gray_offset_payload_start (activation_bit_capacity,
                                                                payload_bit_capacity,
                                                                selected_index);
        best_distance = itty_feed_model_gray_offset_distance_for_start (activation,
                                                                        target,
                                                                        best_start);

        for (size_t payload_start = min_start; payload_start <= max_start; payload_start++) {
                size_t distance =
                        itty_feed_model_gray_offset_distance_for_start (activation,
                                                                        target,
                                                                        payload_start);

                if (distance < best_distance ||
                    (distance == best_distance &&
                     payload_start < best_start)) {
                        best_start = payload_start;
                        best_distance = distance;
                }
        }

        return best_start;
}

static itty_bit_string_t *
itty_feed_model_build_gray_offset_target (itty_bit_string_t *activation,
                                          itty_bit_string_t *target,
                                          size_t             selected_index,
                                          bool               override_enabled,
                                          size_t             override_payload_start)
{
        size_t activation_words = itty_bit_string_get_number_of_words (activation);
        size_t activation_bit_capacity = itty_bit_string_get_length (activation);
        size_t payload_bit_capacity = itty_bit_string_get_length (target);
        size_t payload_start = override_enabled ?
                               override_payload_start :
                               itty_feed_model_find_best_gray_offset_payload_start (activation,
                                                                                    target,
                                                                                    selected_index);
        size_t gray_offset = itty_feed_model_gray_code (payload_start & 0xff);
        itty_bit_string_t *decoder_target =
                itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);

        if (!decoder_target)
                return NULL;

        for (size_t word_index = 0; word_index < activation_words; word_index++)
                itty_bit_string_append_word (decoder_target, 0);

        if (itty_bit_string_get_number_of_words (decoder_target) != activation_words) {
                itty_bit_string_free (decoder_target);
                return NULL;
        }

        for (size_t bit_index = 0; bit_index < activation_bit_capacity; bit_index++)
                itty_bit_string_set_bit (decoder_target, bit_index, (bit_index % 2) != 0);

        for (size_t bit_index = 0; bit_index < 8 && bit_index < activation_bit_capacity; bit_index++)
                itty_bit_string_set_bit (decoder_target,
                                         bit_index,
                                         ((gray_offset >> bit_index) & 1U) != 0);

        for (size_t bit_index = 0;
             bit_index < payload_bit_capacity &&
             payload_start + bit_index < activation_bit_capacity;
             bit_index++) {
                itty_bit_string_set_bit (decoder_target,
                                         payload_start + bit_index,
                                         itty_bit_string_get_bit (target, bit_index));
        }

        return decoder_target;
}

static size_t
itty_feed_model_gray_offset_payload_start_for_activation (itty_feed_model_t *model,
                                                          itty_bit_string_t *activation,
                                                          itty_bit_string_t *target,
                                                          size_t             selected_index)
{
        if (selected_index < model->nodes_per_layer &&
            model->gray_offset_override_enabled_by_route &&
            model->gray_offset_override_enabled_by_route[selected_index])
                return model->gray_offset_override_payload_start_by_route[selected_index];

        return itty_feed_model_find_best_gray_offset_payload_start (activation,
                                                                    target,
                                                                    selected_index);
}

static itty_bit_string_t *
itty_feed_model_extract_gray_payload_window (itty_feed_model_t *model,
                                             itty_bit_string_t *activation,
                                             itty_bit_string_t *target,
                                             size_t             selected_index)
{
        size_t payload_bit_capacity = itty_bit_string_get_length (target);
        size_t payload_words = itty_bit_string_get_number_of_words (target);
        size_t payload_start =
                itty_feed_model_gray_offset_payload_start_for_activation (model,
                                                                          activation,
                                                                          target,
                                                                          selected_index);
        itty_bit_string_t *window =
                itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);

        if (!window)
                return NULL;

        itty_bit_string_append_zeros (window, payload_words);

        for (size_t bit_index = 0; bit_index < payload_bit_capacity; bit_index++) {
                bool value = itty_bit_string_get_bit (activation, payload_start + bit_index);
                itty_bit_string_set_bit (window, bit_index, value);
        }

        return window;
}

static size_t
itty_feed_model_margin_preserving_vote_for_bit_string (itty_bit_string_t *bit_string)
{
        return itty_bit_string_get_pop_count (bit_string) + 1;
}

static itty_bit_string_t *
itty_feed_model_condense_with_threshold (itty_bit_string_list_t *list,
                                         size_t                  threshold)
{
        if (!list || list->count == 0)
                return NULL;

        size_t bit_length = itty_bit_string_list_get_bit_length (list);
        size_t number_of_words = (bit_length + ITTY_BIT_STRING_WORD_SIZE_IN_BITS - 1) /
                                 ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        itty_bit_string_t *condensed_bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);

        if (!condensed_bit_string)
                return NULL;

        condensed_bit_string->words = calloc (number_of_words, sizeof (size_t));
        if (number_of_words > 0 && !condensed_bit_string->words) {
                itty_bit_string_free (condensed_bit_string);
                return NULL;
        }
        condensed_bit_string->number_of_words = number_of_words;

        if (threshold == 0)
                threshold = list->count / 2 + 1;
        if (threshold > list->count)
                threshold = list->count;

        for (size_t bit_index = 0; bit_index < bit_length; bit_index++) {
                size_t one_votes = 0;

                for (size_t string_index = 0; string_index < list->count; string_index++) {
                        if (itty_bit_string_get_bit (list->bit_strings[string_index], bit_index))
                                one_votes++;
                }

                if (one_votes >= threshold)
                        itty_bit_string_set_bit (condensed_bit_string, bit_index, true);
        }

        return condensed_bit_string;
}

static void
itty_feed_model_final_repair_list_clear (itty_feed_model_final_repair_list_t *repairs)
{
        free (repairs->items);
        *repairs = (itty_feed_model_final_repair_list_t) { 0 };
}

static void
itty_feed_model_final_repair_list_append (itty_feed_model_final_repair_list_t *repairs,
                                          size_t                               final_node,
                                          size_t                               condensed_bit,
                                          size_t                               output_bit,
                                          size_t                               decoded_bit,
                                          size_t                               quota_size,
                                          bool                                 value)
{
        if (repairs->count == repairs->capacity) {
                repairs->capacity = repairs->capacity == 0 ? 64 : repairs->capacity * 2;
                repairs->items = realloc (repairs->items,
                                          repairs->capacity * sizeof (itty_feed_model_final_repair_t));
        }

        repairs->items[repairs->count++] = (itty_feed_model_final_repair_t) {
                .final_node = final_node,
                .condensed_bit = condensed_bit,
                .output_bit = output_bit,
                .decoded_bit = decoded_bit,
                .quota_size = quota_size,
                .value = value
        };
}

static void
itty_feed_model_layer_assignment_list_clear (itty_feed_model_layer_assignment_list_t *assignments)
{
        free (assignments->items);
        *assignments = (itty_feed_model_layer_assignment_list_t) { 0 };
}

static void
itty_feed_model_layer_assignment_list_append (itty_feed_model_layer_assignment_list_t *assignments,
                                             size_t                                   layer_node,
                                             size_t                                   bit_index,
                                             bool                                     value)
{
        if (assignments->count == assignments->capacity) {
                assignments->capacity = assignments->capacity == 0 ? 64 : assignments->capacity * 2;
                assignments->items = realloc (assignments->items,
                                              assignments->capacity * sizeof (itty_feed_model_layer_assignment_t));
        }

        assignments->items[assignments->count++] = (itty_feed_model_layer_assignment_t) {
                .layer_node = layer_node,
                .bit_index = bit_index,
                .value = value
        };
}

static bool
itty_feed_model_layer_assignment_list_append_unique (itty_feed_model_layer_assignment_list_t *assignments,
                                                     size_t                                   layer_node,
                                                     size_t                                   bit_index,
                                                     bool                                     value)
{
        for (size_t assignment_index = 0; assignment_index < assignments->count; assignment_index++) {
                itty_feed_model_layer_assignment_t *assignment = &assignments->items[assignment_index];

                if (assignment->layer_node != layer_node ||
                    assignment->bit_index != bit_index)
                        continue;

                return assignment->value == value;
        }

        itty_feed_model_layer_assignment_list_append (assignments,
                                                     layer_node,
                                                     bit_index,
                                                     value);
        return true;
}

static bool
itty_feed_model_layer_assignment_list_copy (itty_feed_model_layer_assignment_list_t       *destination,
                                            itty_feed_model_layer_assignment_list_t const *source)
{
        for (size_t assignment_index = 0; assignment_index < source->count; assignment_index++) {
                itty_feed_model_layer_assignment_t const *assignment = &source->items[assignment_index];

                if (!itty_feed_model_layer_assignment_list_append_unique (destination,
                                                                         assignment->layer_node,
                                                                         assignment->bit_index,
                                                                         assignment->value))
                        return false;
        }

        return true;
}

static void
itty_feed_model_mask_flip_list_append (itty_feed_model_mask_flip_list_t *list,
                                       size_t                            layer_node,
                                       size_t                            input_index,
                                       size_t                            bit_index)
{
        if (list->count == list->capacity) {
                list->capacity = list->capacity == 0 ? 8 : list->capacity * 2;
                list->items = realloc (list->items,
                                       list->capacity * sizeof (itty_feed_model_mask_flip_t));
        }

        list->items[list->count++] = (itty_feed_model_mask_flip_t) {
                .layer_node = layer_node,
                .input_index = input_index,
                .bit_index = bit_index
        };
}

static void
itty_feed_model_mask_flip_list_clear (itty_feed_model_mask_flip_list_t *list)
{
        free (list->items);
        *list = (itty_feed_model_mask_flip_list_t) { 0 };
}

static void
itty_feed_model_bad_flip_frequency_list_clear (itty_feed_model_bad_flip_frequency_list_t *list)
{
        free (list->items);
        *list = (itty_feed_model_bad_flip_frequency_list_t) { 0 };
}

static void
itty_feed_model_bad_flip_frequency_list_record (itty_feed_model_bad_flip_frequency_list_t *list,
                                                size_t                                    layer_index,
                                                itty_feed_model_mask_flip_t const        *flip,
                                                bool                                      new_mask_bit,
                                                bool                                      bad,
                                                size_t                                    damaged_bits,
                                                size_t                                    helped_decoded_bit)
{
        for (size_t item_index = 0; item_index < list->count; item_index++) {
                itty_feed_model_bad_flip_frequency_t *item = &list->items[item_index];

                if (item->layer_index != layer_index ||
                    item->layer_node != flip->layer_node ||
                    item->input_index != flip->input_index ||
                    item->bit_index != flip->bit_index ||
                    item->new_mask_bit != new_mask_bit)
                        continue;

                if (bad) {
                        item->bad_count++;
                        item->damaged_bits += damaged_bits;
                } else {
                        item->harmless_count++;
                }
                return;
        }

        if (list->count == list->capacity) {
                list->capacity = list->capacity == 0 ? 16 : list->capacity * 2;
                list->items = realloc (list->items,
                                       list->capacity * sizeof (itty_feed_model_bad_flip_frequency_t));
        }

        list->items[list->count++] = (itty_feed_model_bad_flip_frequency_t) {
                .layer_index = layer_index,
                .layer_node = flip->layer_node,
                .input_index = flip->input_index,
                .bit_index = flip->bit_index,
                .new_mask_bit = new_mask_bit,
                .bad_count = bad ? 1 : 0,
                .harmless_count = bad ? 0 : 1,
                .damaged_bits = bad ? damaged_bits : 0,
                .helped_decoded_bit = helped_decoded_bit
        };
}

static void
itty_feed_model_bad_flip_frequency_list_finish (itty_feed_model_bad_flip_frequency_list_t const *list,
                                                itty_feed_model_projected_repair_stats_t        *stats)
{
        if (!stats)
                return;

        for (size_t item_index = 0; item_index < list->count; item_index++) {
                itty_feed_model_bad_flip_frequency_t const *item = &list->items[item_index];

                if (item->bad_count == 0)
                        continue;

                stats->replay_bad_flip_unique++;
                if (item->bad_count < stats->replay_bad_flip_top_frequency)
                        continue;
                if (item->bad_count == stats->replay_bad_flip_top_frequency &&
                    item->damaged_bits <= stats->replay_bad_flip_top_damaged_bits)
                        continue;

                stats->replay_bad_flip_top_frequency = item->bad_count;
                stats->replay_bad_flip_top_harmless_uses = item->harmless_count;
                stats->replay_bad_flip_top_damaged_bits = item->damaged_bits;
                stats->replay_bad_flip_top_helped_decoded_bit = item->helped_decoded_bit;
                stats->replay_bad_flip_top_layer = item->layer_index;
                stats->replay_bad_flip_top_node = item->layer_node;
                stats->replay_bad_flip_top_input = item->input_index;
                stats->replay_bad_flip_top_bit = item->bit_index;
                stats->replay_bad_flip_top_value = item->new_mask_bit;
        }
}

static void
itty_feed_model_replay_taboo_map_clear (itty_feed_model_replay_taboo_map_t *map)
{
        free (map->items);
        *map = (itty_feed_model_replay_taboo_map_t) { 0 };
}

static itty_feed_model_replay_taboo_entry_t *
itty_feed_model_replay_taboo_map_find (itty_feed_model_replay_taboo_map_t *map,
                                       size_t                              layer_index,
                                       size_t                              layer_node,
                                       size_t                              input_index,
                                       size_t                              bit_index,
                                       bool                                desired_value)
{
        for (size_t item_index = 0; item_index < map->count; item_index++) {
                itty_feed_model_replay_taboo_entry_t *item = &map->items[item_index];

                if (item->layer_index != layer_index ||
                    item->layer_node != layer_node ||
                    item->input_index != input_index ||
                    item->bit_index != bit_index ||
                    item->desired_value != desired_value)
                        continue;

                return item;
        }

        return NULL;
}

static bool
itty_feed_model_replay_taboo_map_record (itty_feed_model_replay_taboo_map_t *map,
                                         size_t                               layer_index,
                                         size_t                               layer_node,
                                         size_t                               input_index,
                                         size_t                               bit_index,
                                         bool                                 desired_value,
                                         bool                                 zero_protection_taboo,
                                         bool                                 one_protection_taboo)
{
        itty_feed_model_replay_taboo_entry_t *existing =
                itty_feed_model_replay_taboo_map_find (map,
                                                       layer_index,
                                                       layer_node,
                                                       input_index,
                                                       bit_index,
                                                       desired_value);
        if (existing) {
                existing->zero_protection_taboo |= zero_protection_taboo;
                existing->one_protection_taboo |= one_protection_taboo;
                return true;
        }

        if (map->count == map->capacity) {
                map->capacity = map->capacity == 0 ? 32 : map->capacity * 2;
                map->items = realloc (map->items,
                                      map->capacity * sizeof (itty_feed_model_replay_taboo_entry_t));
        }

        map->items[map->count++] = (itty_feed_model_replay_taboo_entry_t) {
                .layer_index = layer_index,
                .layer_node = layer_node,
                .input_index = input_index,
                .bit_index = bit_index,
                .desired_value = desired_value,
                .zero_protection_taboo = zero_protection_taboo,
                .one_protection_taboo = one_protection_taboo,
        };

        return true;
}

static void
itty_feed_model_accumulate_bad_flip_top (itty_feed_model_projected_repair_stats_t       *stats,
                                         itty_feed_model_projected_repair_stats_t const *batch_stats)
{
        stats->replay_bad_flip_unique += batch_stats->replay_bad_flip_unique;
        stats->replay_taboo_vote_candidates += batch_stats->replay_taboo_vote_candidates;
        stats->replay_taboo_mask_flips += batch_stats->replay_taboo_mask_flips;
        stats->replay_taboo_penalty_total += batch_stats->replay_taboo_penalty_total;
        stats->replay_taboo_rejected_vote_candidates += batch_stats->replay_taboo_rejected_vote_candidates;
        stats->replay_minus_one_bad_candidates += batch_stats->replay_minus_one_bad_candidates;
        stats->replay_minus_one_bad_safe_candidates += batch_stats->replay_minus_one_bad_safe_candidates;
        stats->replay_minus_one_bad_deficit_candidates += batch_stats->replay_minus_one_bad_deficit_candidates;
        stats->replay_minus_one_bad_strict_candidates += batch_stats->replay_minus_one_bad_strict_candidates;
        if (batch_stats->replay_bad_flip_top_frequency < stats->replay_bad_flip_top_frequency)
                return;
        if (batch_stats->replay_bad_flip_top_frequency == stats->replay_bad_flip_top_frequency &&
            batch_stats->replay_bad_flip_top_damaged_bits <= stats->replay_bad_flip_top_damaged_bits)
                return;

        stats->replay_bad_flip_top_frequency = batch_stats->replay_bad_flip_top_frequency;
        stats->replay_bad_flip_top_harmless_uses = batch_stats->replay_bad_flip_top_harmless_uses;
        stats->replay_bad_flip_top_damaged_bits = batch_stats->replay_bad_flip_top_damaged_bits;
        stats->replay_bad_flip_top_helped_decoded_bit = batch_stats->replay_bad_flip_top_helped_decoded_bit;
        stats->replay_bad_flip_top_layer = batch_stats->replay_bad_flip_top_layer;
        stats->replay_bad_flip_top_node = batch_stats->replay_bad_flip_top_node;
        stats->replay_bad_flip_top_input = batch_stats->replay_bad_flip_top_input;
        stats->replay_bad_flip_top_bit = batch_stats->replay_bad_flip_top_bit;
        stats->replay_bad_flip_top_value = batch_stats->replay_bad_flip_top_value;
}

static void
itty_feed_model_projected_repair_candidate_clear (itty_feed_model_projected_repair_candidate_t *candidate)
{
        itty_feed_model_layer_assignment_list_clear (&candidate->output_assignments);
        itty_feed_model_layer_assignment_list_clear (&candidate->condensed_assignments);
        *candidate = (itty_feed_model_projected_repair_candidate_t) { 0 };
}

static int
compare_projected_repair_candidates (const void *a,
                                     const void *b)
{
        itty_feed_model_projected_repair_candidate_t const *candidate_a = a;
        itty_feed_model_projected_repair_candidate_t const *candidate_b = b;

        if (candidate_a->rank < candidate_b->rank)
                return -1;
        if (candidate_a->rank > candidate_b->rank)
                return 1;
        if (candidate_a->distance_delta > candidate_b->distance_delta)
                return -1;
        if (candidate_a->distance_delta < candidate_b->distance_delta)
                return 1;
        if (candidate_a->false_negative_count_delta > candidate_b->false_negative_count_delta)
                return -1;
        if (candidate_a->false_negative_count_delta < candidate_b->false_negative_count_delta)
                return 1;
        if (candidate_a->prefer_blocker_efficiency ||
            candidate_b->prefer_blocker_efficiency) {
                size_t cost_a = candidate_a->estimated_flips == 0 ? 1 : candidate_a->estimated_flips;
                size_t cost_b = candidate_b->estimated_flips == 0 ? 1 : candidate_b->estimated_flips;
                size_t efficiency_a = candidate_a->blocker_delta * cost_b;
                size_t efficiency_b = candidate_b->blocker_delta * cost_a;

                if (efficiency_a > efficiency_b)
                        return -1;
                if (efficiency_a < efficiency_b)
                        return 1;
        }
        if (candidate_a->blocker_delta > candidate_b->blocker_delta)
                return -1;
        if (candidate_a->blocker_delta < candidate_b->blocker_delta)
                return 1;
        if (candidate_a->vote_deficit_delta > candidate_b->vote_deficit_delta)
                return -1;
        if (candidate_a->vote_deficit_delta < candidate_b->vote_deficit_delta)
                return 1;
        if (candidate_a->target_one_margin_delta > candidate_b->target_one_margin_delta)
                return -1;
        if (candidate_a->target_one_margin_delta < candidate_b->target_one_margin_delta)
                return 1;
        size_t effective_cost_a = candidate_a->estimated_flips + candidate_a->replay_collateral_cost;
        size_t effective_cost_b = candidate_b->estimated_flips + candidate_b->replay_collateral_cost;
        if (effective_cost_a < effective_cost_b)
                return -1;
        if (effective_cost_a > effective_cost_b)
                return 1;
        if (candidate_a->replay_collateral_cost < candidate_b->replay_collateral_cost)
                return -1;
        if (candidate_a->replay_collateral_cost > candidate_b->replay_collateral_cost)
                return 1;
        if (candidate_a->estimated_flips < candidate_b->estimated_flips)
                return -1;
        if (candidate_a->estimated_flips > candidate_b->estimated_flips)
                return 1;
        if (candidate_a->decoded_bit < candidate_b->decoded_bit)
                return -1;
        if (candidate_a->decoded_bit > candidate_b->decoded_bit)
                return 1;
        if (candidate_a->final_node < candidate_b->final_node)
                return -1;
        if (candidate_a->final_node > candidate_b->final_node)
                return 1;
        if (candidate_a->condensed_assignments.count < candidate_b->condensed_assignments.count)
                return -1;
        if (candidate_a->condensed_assignments.count > candidate_b->condensed_assignments.count)
                return 1;
        if (candidate_a->original_index < candidate_b->original_index)
                return -1;
        if (candidate_a->original_index > candidate_b->original_index)
                return 1;
        return 0;
}

static int
compare_quota_vote_candidates_by_cost (const void *a,
                                       const void *b)
{
        itty_feed_model_quota_vote_candidate_t const *candidate_a = a;
        itty_feed_model_quota_vote_candidate_t const *candidate_b = b;
        size_t cost_a = candidate_a->estimated_flips +
                        candidate_a->replay_protection_penalty +
                        candidate_a->replay_taboo_penalty;
        size_t cost_b = candidate_b->estimated_flips +
                        candidate_b->replay_protection_penalty +
                        candidate_b->replay_taboo_penalty;

        if (cost_a < cost_b)
                return -1;
        if (cost_a > cost_b)
                return 1;
        if (candidate_a->replay_protection_penalty < candidate_b->replay_protection_penalty)
                return -1;
        if (candidate_a->replay_protection_penalty > candidate_b->replay_protection_penalty)
                return 1;
        if (candidate_a->replay_taboo_penalty < candidate_b->replay_taboo_penalty)
                return -1;
        if (candidate_a->replay_taboo_penalty > candidate_b->replay_taboo_penalty)
                return 1;
        if (candidate_a->output_bit < candidate_b->output_bit)
                return -1;
        if (candidate_a->output_bit > candidate_b->output_bit)
                return 1;
        if (candidate_a->condensed_bit < candidate_b->condensed_bit)
                return -1;
        if (candidate_a->condensed_bit > candidate_b->condensed_bit)
                return 1;
        if (candidate_a->original_index < candidate_b->original_index)
                return -1;
        if (candidate_a->original_index > candidate_b->original_index)
                return 1;
        return 0;
}

static int
compare_scored_mask_flips_by_cost (const void *a,
                                   const void *b)
{
        itty_feed_model_scored_mask_flip_t const *flip_a = a;
        itty_feed_model_scored_mask_flip_t const *flip_b = b;

        if (flip_a->cost < flip_b->cost)
                return -1;
        if (flip_a->cost > flip_b->cost)
                return 1;
        if (flip_a->unsafe != flip_b->unsafe)
                return flip_a->unsafe ? 1 : -1;
        if (flip_a->flip.layer_node < flip_b->flip.layer_node)
                return -1;
        if (flip_a->flip.layer_node > flip_b->flip.layer_node)
                return 1;
        if (flip_a->flip.input_index < flip_b->flip.input_index)
                return -1;
        if (flip_a->flip.input_index > flip_b->flip.input_index)
                return 1;
        if (flip_a->flip.bit_index < flip_b->flip.bit_index)
                return -1;
        if (flip_a->flip.bit_index > flip_b->flip.bit_index)
                return 1;
        return 0;
}

static size_t
itty_feed_model_positive_delta (size_t before,
                                size_t after)
{
        return before > after ? before - after : 0;
}

static size_t
itty_feed_model_positive_increase (size_t before,
                                   size_t after)
{
        return after > before ? after - before : 0;
}

static ptrdiff_t
itty_feed_model_signed_delta (size_t before,
                              size_t after)
{
        return (ptrdiff_t) before - (ptrdiff_t) after;
}

static ptrdiff_t
itty_feed_model_signed_increase (size_t before,
                                 size_t after)
{
        return (ptrdiff_t) after - (ptrdiff_t) before;
}

static size_t
itty_feed_model_weighted_deficit_histogram_cost (itty_feed_model_decoder_objective_t const *objective)
{
        size_t cost = 0;

        for (size_t bucket = 0; bucket < ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS; bucket++)
                cost += bucket * objective->false_negative_vote_deficit_histogram[bucket];

        return cost;
}

static size_t
itty_feed_model_random_next (itty_feed_model_random_t *random)
{
        random->state ^= random->state >> 12;
        random->state ^= random->state << 25;
        random->state ^= random->state >> 27;
        return random->state * 2685821657736338717ULL;
}

static itty_bit_string_t *
itty_feed_model_bit_string_clone (itty_bit_string_t *bit_string)
{
        itty_bit_string_t *copy = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        copy->number_of_words = bit_string->number_of_words;
        if (copy->number_of_words > 0) {
                copy->words = malloc (copy->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
                memcpy (copy->words, bit_string->words, copy->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
        }
        copy->pop_count = bit_string->pop_count;
        copy->pop_count_computed = bit_string->pop_count_computed;
        return copy;
}

static itty_bit_string_t *
itty_feed_model_bit_string_clone_to_words (itty_bit_string_t *bit_string,
                                           size_t             number_of_words)
{
        itty_bit_string_t *copy = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        copy->number_of_words = number_of_words;
        if (copy->number_of_words > 0) {
                copy->words = calloc (copy->number_of_words, ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
                size_t copied_words = bit_string->number_of_words < copy->number_of_words ?
                                      bit_string->number_of_words :
                                      copy->number_of_words;
                memcpy (copy->words,
                        bit_string->words,
                        copied_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
        }
        return copy;
}

static itty_bit_string_t *
itty_feed_model_bit_string_complement_clone (itty_bit_string_t *bit_string)
{
        itty_bit_string_t *copy = itty_feed_model_bit_string_clone (bit_string);

        for (size_t word_index = 0; word_index < copy->number_of_words; word_index++)
                copy->words[word_index] = ~copy->words[word_index];

        copy->pop_count_computed = false;
        return copy;
}

static itty_bit_string_list_t *
itty_feed_model_bit_string_list_clone (itty_bit_string_list_t *list)
{
        itty_bit_string_list_t *copy = itty_bit_string_list_new ();

        for (size_t i = 0; i < list->count; i++)
                itty_bit_string_list_append (copy,
                                             itty_feed_model_bit_string_clone (list->bit_strings[i]));

        return copy;
}

static itty_bit_string_t *
itty_feed_model_zero_mask_new (size_t number_of_words)
{
        itty_bit_string_t *mask = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        itty_bit_string_append_zeros (mask, number_of_words);
        return mask;
}

static size_t
itty_feed_model_get_layer_input_count (itty_feed_model_t *model,
                                       size_t             layer_index)
{
        return layer_index == 0 ? model->inputs_per_node : model->nodes_per_layer;
}

static bool
itty_feed_model_masked_input_bit (itty_bit_string_t *input,
                                  itty_bit_string_t *mask,
                                  size_t             bit_index)
{
        return itty_bit_string_get_bit (input, bit_index) != itty_bit_string_get_bit (mask, bit_index);
}

static void
itty_feed_model_flip_mask_bit (itty_bit_string_t *mask,
                               size_t             bit_index)
{
        itty_bit_string_set_bit (mask, bit_index, !itty_bit_string_get_bit (mask, bit_index));
        mask->pop_count_computed = false;
}

static bool
itty_feed_model_output_transform_swaps_bit (itty_feed_model_output_transform_t transform,
                                            size_t                             bit_index,
                                            size_t                             bit_capacity)
{
        switch (transform) {
        case ITTY_FEED_MODEL_OUTPUT_TRANSFORM_IDENTITY:
                return false;
        case ITTY_FEED_MODEL_OUTPUT_TRANSFORM_ODD_SWAP:
                return (bit_index % 2) == 1;
        case ITTY_FEED_MODEL_OUTPUT_TRANSFORM_EVEN_SWAP:
                return (bit_index % 2) == 0;
        case ITTY_FEED_MODEL_OUTPUT_TRANSFORM_HALF_SWAP:
                return bit_index >= bit_capacity / 2;
        default:
                return false;
        }
}

static void
itty_feed_model_apply_output_transform (itty_bit_string_list_t            *outputs,
                                        itty_feed_model_output_transform_t transform)
{
        if (!outputs ||
            outputs->count < 2 ||
            transform == ITTY_FEED_MODEL_OUTPUT_TRANSFORM_IDENTITY)
                return;

        size_t bit_capacity = itty_bit_string_get_length (outputs->bit_strings[0]);

        for (size_t node_index = 0; node_index + 1 < outputs->count; node_index += 2) {
                itty_bit_string_t *left = outputs->bit_strings[node_index];
                itty_bit_string_t *right = outputs->bit_strings[node_index + 1];

                for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                        if (!itty_feed_model_output_transform_swaps_bit (transform,
                                                                         bit_index,
                                                                         bit_capacity))
                                continue;

                        bool left_bit = itty_bit_string_get_bit (left, bit_index);
                        bool right_bit = itty_bit_string_get_bit (right, bit_index);

                        if (left_bit == right_bit)
                                continue;

                        itty_bit_string_set_bit (left, bit_index, right_bit);
                        itty_bit_string_set_bit (right, bit_index, left_bit);
                }

                left->pop_count_computed = false;
                right->pop_count_computed = false;
        }
}

static void
itty_feed_model_set_mutable_bit (itty_bit_string_t *bit_string,
                                 size_t             bit_index,
                                 bool               value)
{
        itty_bit_string_set_bit (bit_string,
                                 bit_index,
                                 value);
        bit_string->pop_count_computed = false;
}

static size_t
itty_feed_model_count_votes (itty_bit_string_list_t *input,
                             itty_bit_string_list_t *masks,
                             size_t                  bit_index)
{
        size_t vote_count = 0;

        for (size_t input_index = 0; input_index < input->count; input_index++) {
                if (itty_feed_model_masked_input_bit (input->bit_strings[input_index],
                                                      masks->bit_strings[input_index],
                                                      bit_index))
                        vote_count++;
        }

        return vote_count;
}

static itty_bit_string_t *
itty_feed_model_expand_target_for_layer (itty_bit_string_t *target,
                                         size_t             layer_index)
{
        itty_bit_string_t *expanded_target = itty_feed_model_bit_string_clone (target);

        for (size_t i = 0; i < layer_index; i++) {
                itty_bit_string_t *next = itty_bit_string_double (expanded_target);
                itty_bit_string_free (expanded_target);
                expanded_target = next;
        }

        return expanded_target;
}

static itty_bit_string_t *
itty_feed_model_prepare_decoder_target_for_activation (itty_feed_model_t *model,
                                                       itty_bit_string_t *activation,
                                                       itty_bit_string_t *target,
                                                       size_t             selected_index)
{
        if (!model || !activation || !target)
                return NULL;

        if (!itty_feed_model_decoder_uses_direct_activation_target (model->decoder))
                return itty_feed_model_bit_string_clone (target);

        if (itty_feed_model_decoder_uses_direct_gray_offset_target (model->decoder))
                return itty_feed_model_bit_string_clone (target);

        itty_bit_string_t *decoder_target = itty_feed_model_bit_string_clone (target);
        if (!decoder_target)
                return NULL;

        if (itty_feed_model_decoder_uses_direct_duplicated_target (model->decoder)) {
                while (itty_bit_string_get_number_of_words (decoder_target) <
                       itty_bit_string_get_number_of_words (activation)) {
                        itty_bit_string_t *next = itty_bit_string_double (decoder_target);
                        itty_bit_string_free (decoder_target);
                        decoder_target = next;
                        if (!decoder_target)
                                return NULL;
                }
        } else {
                size_t target_words = itty_bit_string_get_number_of_words (decoder_target);
                size_t activation_words = itty_bit_string_get_number_of_words (activation);

                if (target_words < activation_words)
                        itty_bit_string_append_zeros (decoder_target, activation_words - target_words);
        }

        if (itty_bit_string_get_number_of_words (decoder_target) !=
            itty_bit_string_get_number_of_words (activation)) {
                itty_bit_string_free (decoder_target);
                return NULL;
        }

        return decoder_target;
}

static itty_bit_string_t *
itty_feed_model_run_node (itty_bit_string_list_t *input,
                          itty_bit_string_list_t *masks,
                          size_t                  node_index,
                          itty_bit_string_t      *residual_enable,
                          itty_bit_string_t      *residual_mask,
                          size_t                  rotation,
                          itty_feed_model_decoder_t decoder,
                          size_t                  within_node_threshold_override,
                          bool                    residual_merge_enabled)
{
        itty_bit_string_list_t *modulated_inputs = itty_bit_string_list_exclusive_or (input,
                                                                                      masks);
        itty_bit_string_t *condensed_output;
        if (decoder == ITTY_FEED_MODEL_DECODER_SEGMENT_WEIGHTED_CONDENSE) {
                size_t votes[64] = { 0 };
                size_t vote_count = modulated_inputs ? modulated_inputs->count : 0;
                if (vote_count > sizeof votes / sizeof votes[0])
                        vote_count = sizeof votes / sizeof votes[0];
                for (size_t i = 0; i < vote_count; i++)
                        votes[i] = itty_feed_model_margin_preserving_vote_for_bit_string (modulated_inputs->bit_strings[i]);
                condensed_output = itty_bit_string_list_weighted_condense (modulated_inputs,
                                                                           votes,
                                                                           vote_count);
        } else {
                condensed_output = itty_feed_model_condense_with_threshold (modulated_inputs,
                                                                            within_node_threshold_override);
        }
        itty_bit_string_list_free (modulated_inputs);

        if (!condensed_output)
                condensed_output = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);

        if (residual_merge_enabled &&
            residual_enable &&
            residual_mask &&
            itty_bit_string_get_pop_count (residual_enable) > 0 &&
            input->count > 0) {
                itty_bit_string_t *skip_input = itty_bit_string_list_fetch (input,
                                                                            node_index % input->count);
                itty_bit_string_t *skip = itty_bit_string_exclusive_or (skip_input,
                                                                        residual_mask);
                itty_bit_string_t *enabled_skip = itty_bit_string_mask (residual_enable,
                                                                        skip);
                itty_bit_string_t *merged_output = itty_bit_string_combine (condensed_output,
                                                                            enabled_skip);
                itty_bit_string_free (skip);
                itty_bit_string_free (enabled_skip);
                itty_bit_string_free (condensed_output);
                condensed_output = merged_output;
        }

        itty_bit_string_t *doubled_output = itty_feed_model_expand_condensed_output (condensed_output,
                                                                                     rotation);
        itty_bit_string_free (condensed_output);

        return doubled_output;
}

static itty_bit_string_t *
itty_feed_model_expand_condensed_output (itty_bit_string_t *condensed_output,
                                         size_t             rotation)
{
        return rotation == 0 ?
               itty_bit_string_double (condensed_output) :
               itty_bit_string_double_with_rotated_half (condensed_output,
                                                         rotation);
}

static itty_bit_string_list_t *
itty_feed_model_apply_input_route_adapter (itty_feed_model_t      *model,
                                           itty_bit_string_list_t *input,
                                           size_t                  route_index)
{
        if (!model ||
            !input ||
            !model->input_route_adapter_enabled ||
            route_index >= model->nodes_per_layer)
                return input;

        itty_bit_string_list_t *adapter = model->input_route_adapters[route_index];
        if (!adapter)
                return input;

        itty_bit_string_list_t *adapted = itty_bit_string_list_exclusive_or (input, adapter);
        return adapted ? adapted : input;
}

static itty_bit_string_list_t *
itty_feed_model_run_layer (itty_feed_model_t      *model,
                           size_t                  layer_index,
                           itty_bit_string_list_t *input)
{
        itty_bit_string_list_t *outputs = itty_bit_string_list_new ();

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_list_t *node_input = input;
                if (layer_index == 0)
                        node_input = itty_feed_model_apply_input_route_adapter (model, input, node_index);

                itty_bit_string_t *output = itty_feed_model_run_node (node_input,
                                                                      model->masks_by_layer_node[layer_index][node_index],
                                                                      node_index,
                                                                      model->residual_enable_by_layer_node[layer_index][node_index],
                                                                      model->residual_mask_by_layer_node[layer_index][node_index],
                                                                      model->rotations_by_layer[layer_index],
                                                                      model->decoder,
                                                                      model->within_node_condense_threshold_override,
                                                                      model->residual_merge_enabled);
                itty_bit_string_list_append (outputs, output);
                if (node_input != input)
                        itty_bit_string_list_free (node_input);
        }

        return outputs;
}

static itty_bit_string_list_t *
itty_feed_model_run_to_layer_input (itty_feed_model_t      *model,
                                    itty_bit_string_list_t *input,
                                    size_t                  target_layer)
{
        itty_bit_string_list_t *current_input = input;

        for (size_t layer_index = 0; layer_index < target_layer; layer_index++) {
                itty_bit_string_list_t *layer_output = itty_feed_model_run_layer (model,
                                                                                  layer_index,
                                                                                  current_input);
                if (current_input != input)
                        itty_bit_string_list_free (current_input);
                current_input = layer_output;
        }

        return current_input;
}

static itty_bit_string_list_t *
itty_feed_model_run_outputs (itty_feed_model_t      *model,
                             itty_bit_string_list_t *input)
{
        return itty_feed_model_run_to_layer_input (model,
                                                   input,
                                                   model->number_of_layers);
}

static bool
itty_feed_model_train_layer_one_node (itty_bit_string_list_t                *masks,
                                      itty_bit_string_list_t                *input,
                                      itty_bit_string_t                     *target,
                                      itty_feed_model_train_options_t const *options,
                                      itty_feed_model_train_stats_t         *stats)
{
        size_t bit_capacity = itty_bit_string_get_number_of_words (target) * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t threshold = itty_bit_string_list_get_length (input) / 2 + 1;
        itty_feed_model_training_candidate_t *candidates = calloc (bit_capacity, sizeof (itty_feed_model_training_candidate_t));
        size_t candidate_count = 0;

        for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                bool target_bit = itty_bit_string_get_bit (target, bit_index);
                size_t vote_count = itty_feed_model_count_votes (input, masks, bit_index);

                if (target_bit) {
                        if (vote_count < threshold) {
                                candidates[candidate_count++] = (itty_feed_model_training_candidate_t) {
                                        .bit_index = bit_index,
                                        .error_weight = threshold - vote_count,
                                        .target_bit = target_bit
                                };
                        }
                } else {
                        if (vote_count >= threshold) {
                                candidates[candidate_count++] = (itty_feed_model_training_candidate_t) {
                                        .bit_index = bit_index,
                                        .error_weight = vote_count - (threshold - 1),
                                        .target_bit = target_bit
                                };
                        }
                }
        }

        qsort (candidates, candidate_count, sizeof (itty_feed_model_training_candidate_t), compare_training_candidates_descending);

        if (stats) {
                stats->candidate_bits += candidate_count;
                if (candidate_count > 0 && candidates[0].error_weight > stats->largest_error)
                        stats->largest_error = candidates[0].error_weight;
        }

        size_t max_flips = options ? options->max_flips : 0;
        size_t flips = 0;

        for (size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
                size_t bit_index = candidates[candidate_index].bit_index;
                bool target_bit = candidates[candidate_index].target_bit;
                size_t vote_count = itty_feed_model_count_votes (input, masks, bit_index);

                for (size_t input_index = 0; input_index < input->count; input_index++) {
                        if (max_flips != 0 && flips >= max_flips) {
                                free (candidates);
                                return true;
                        }

                        if (target_bit && vote_count >= threshold)
                                break;
                        if (!target_bit && vote_count < threshold)
                                break;

                        itty_bit_string_t *input_bit_string = input->bit_strings[input_index];
                        itty_bit_string_t *mask = masks->bit_strings[input_index];
                        bool masked_bit = itty_feed_model_masked_input_bit (input_bit_string,
                                                                            mask,
                                                                            bit_index);
                        if (masked_bit == target_bit)
                                continue;

                        itty_feed_model_flip_mask_bit (mask, bit_index);
                        flips++;
                        if (stats)
                                stats->flips++;
                        if (target_bit)
                                vote_count++;
                        else
                                vote_count--;
                }
        }

        free (candidates);

        return true;
}

static bool
itty_feed_model_train_layer_one_node_with_care (itty_bit_string_list_t        *masks,
                                                itty_bit_string_list_t        *input,
                                                itty_bit_string_t             *target,
                                                itty_bit_string_t             *care,
                                                size_t                         max_flips,
                                                itty_feed_model_train_stats_t *stats)
{
        size_t bit_capacity = itty_bit_string_get_number_of_words (target) * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t threshold = itty_bit_string_list_get_length (input) / 2 + 1;
        itty_feed_model_training_candidate_t *candidates = calloc (bit_capacity, sizeof (itty_feed_model_training_candidate_t));
        size_t candidate_count = 0;

        for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                if (!itty_bit_string_get_bit (care, bit_index))
                        continue;

                bool target_bit = itty_bit_string_get_bit (target, bit_index);
                size_t vote_count = itty_feed_model_count_votes (input, masks, bit_index);

                if (target_bit) {
                        if (vote_count < threshold) {
                                candidates[candidate_count++] = (itty_feed_model_training_candidate_t) {
                                        .bit_index = bit_index,
                                        .error_weight = threshold - vote_count,
                                        .target_bit = target_bit
                                };
                        }
                } else {
                        if (vote_count >= threshold) {
                                candidates[candidate_count++] = (itty_feed_model_training_candidate_t) {
                                        .bit_index = bit_index,
                                        .error_weight = vote_count - (threshold - 1),
                                        .target_bit = target_bit
                                };
                        }
                }
        }

        qsort (candidates, candidate_count, sizeof (itty_feed_model_training_candidate_t), compare_training_candidates_descending);

        if (stats) {
                stats->candidate_bits += candidate_count;
                if (candidate_count > 0 && candidates[0].error_weight > stats->largest_error)
                        stats->largest_error = candidates[0].error_weight;
        }

        size_t flips = 0;

        for (size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
                size_t bit_index = candidates[candidate_index].bit_index;
                bool target_bit = candidates[candidate_index].target_bit;
                size_t vote_count = itty_feed_model_count_votes (input, masks, bit_index);

                for (size_t input_index = 0; input_index < input->count; input_index++) {
                        if (max_flips != 0 && flips >= max_flips) {
                                free (candidates);
                                return true;
                        }

                        if (target_bit && vote_count >= threshold)
                                break;
                        if (!target_bit && vote_count < threshold)
                                break;

                        itty_bit_string_t *input_bit_string = input->bit_strings[input_index];
                        itty_bit_string_t *mask = masks->bit_strings[input_index];
                        bool masked_bit = itty_feed_model_masked_input_bit (input_bit_string,
                                                                            mask,
                                                                            bit_index);
                        if (masked_bit == target_bit)
                                continue;

                        itty_feed_model_flip_mask_bit (mask, bit_index);
                        flips++;
                        if (stats)
                                stats->flips++;
                        if (target_bit)
                                vote_count++;
                        else
                                vote_count--;
                }
        }

        free (candidates);

        return true;
}

static itty_bit_string_t *
itty_feed_model_reduce_desired_output_for_layer (itty_bit_string_t *desired_output,
                                                 size_t             rotation)
{
        return rotation == 0 ?
               itty_bit_string_reduce_by_half (desired_output) :
               itty_bit_string_reduce_rotated_by_half (desired_output, rotation);
}

static bool
itty_feed_model_fold_activation_with_repeated_and (itty_bit_string_t  *activation,
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
itty_feed_model_fold_activation_with_segment_condense (itty_bit_string_t  *activation,
                                                       itty_bit_string_t  *target,
                                                       itty_bit_string_t **folded_activation)
{
        size_t target_words = itty_bit_string_get_number_of_words (target);
        size_t activation_words = itty_bit_string_get_number_of_words (activation);
        size_t target_bit_capacity = target_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t activation_bit_capacity = activation_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        if (target_words == 0)
                return false;

        if (activation_words == 0) {
                itty_bit_string_t *zero_activation = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
                itty_bit_string_append_zeros (zero_activation, target_words);
                *folded_activation = zero_activation;
                return true;
        }

        if (activation_words == target_words) {
                *folded_activation = activation;
                return true;
        }

        if (activation_words < target_words ||
            activation_words % target_words != 0)
                return false;

        size_t segment_count = activation_words / target_words;
        size_t threshold = segment_count / 2 + 1;
        itty_bit_string_t *condensed_activation = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        itty_bit_string_append_zeros (condensed_activation,
                                      target_words);

        for (size_t target_bit_index = 0; target_bit_index < target_bit_capacity; target_bit_index++) {
                size_t ones = 0;

                for (size_t segment_index = 0; segment_index < segment_count; segment_index++) {
                        size_t activation_bit_index = segment_index * target_bit_capacity + target_bit_index;

                        if (activation_bit_index < activation_bit_capacity &&
                            itty_bit_string_get_bit (activation,
                                                     activation_bit_index))
                                ones++;
                }

                if (ones >= threshold)
                        itty_bit_string_set_bit (condensed_activation,
                                                 target_bit_index,
                                                 true);
        }

        condensed_activation->pop_count_computed = false;
        *folded_activation = condensed_activation;
        return true;
}

static bool
itty_feed_model_fold_activation_with_segment_weighted_condense (itty_bit_string_t  *activation,
                                                                itty_bit_string_t  *target,
                                                                itty_bit_string_t **folded_activation)
{
        size_t target_words = itty_bit_string_get_number_of_words (target);
        size_t activation_words = itty_bit_string_get_number_of_words (activation);
        size_t target_bit_capacity = target_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t activation_bit_capacity = activation_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        if (target_words == 0)
                return false;

        if (activation_words == 0) {
                itty_bit_string_t *zero_activation = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
                itty_bit_string_append_zeros (zero_activation, target_words);
                *folded_activation = zero_activation;
                return true;
        }

        if (activation_words == target_words) {
                *folded_activation = activation;
                return true;
        }

        if (activation_words < target_words ||
            activation_words % target_words != 0)
                return false;

        size_t segment_count = activation_words / target_words;
        size_t *votes = calloc (segment_count, sizeof (size_t));
        itty_bit_string_t *condensed_activation = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        if (!votes || !condensed_activation) {
                free (votes);
                if (condensed_activation)
                        itty_bit_string_free (condensed_activation);
                return false;
        }

        itty_bit_string_append_zeros (condensed_activation, target_words);

        __uint128_t total_votes = 0;
        for (size_t segment_index = 0; segment_index < segment_count; segment_index++) {
                size_t vote = 1;
                size_t segment_base = segment_index * target_bit_capacity;

                for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                        size_t activation_bit_index = segment_base + bit_index;
                        if (activation_bit_index < activation_bit_capacity &&
                            itty_bit_string_get_bit (activation, activation_bit_index))
                                vote++;
                }

                votes[segment_index] = vote;
                total_votes += vote;
        }

        __uint128_t threshold = total_votes / 2 + 1;
        for (size_t target_bit_index = 0; target_bit_index < target_bit_capacity; target_bit_index++) {
                __uint128_t one_votes = 0;

                for (size_t segment_index = 0; segment_index < segment_count; segment_index++) {
                        size_t activation_bit_index = segment_index * target_bit_capacity + target_bit_index;
                        if (activation_bit_index < activation_bit_capacity &&
                            itty_bit_string_get_bit (activation, activation_bit_index))
                                one_votes += votes[segment_index];
                }

                if (one_votes >= threshold)
                        itty_bit_string_set_bit (condensed_activation, target_bit_index, true);
        }

        free (votes);
        condensed_activation->pop_count_computed = false;
        *folded_activation = condensed_activation;
        return true;
}

static bool
itty_feed_model_fold_activation_to_target_width (itty_feed_model_t   *model,
                                                 itty_bit_string_t   *activation,
                                                 itty_bit_string_t   *target,
                                                 size_t               selected_index,
                                                 itty_bit_string_t  **folded_activation)
{
        if (itty_feed_model_decoder_uses_direct_gray_offset_target (model->decoder)) {
                *folded_activation = itty_feed_model_extract_gray_payload_window (model,
                                                                                  activation,
                                                                                  target,
                                                                                  selected_index);
                return *folded_activation != NULL;
        }

        if (itty_feed_model_decoder_uses_direct_activation_target (model->decoder)) {
                *folded_activation = activation;
                return true;
        }

        if (model->decoder == ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE)
                return itty_feed_model_fold_activation_with_segment_condense (activation,
                                                                              target,
                                                                              folded_activation);
        if (model->decoder == ITTY_FEED_MODEL_DECODER_SEGMENT_WEIGHTED_CONDENSE)
                return itty_feed_model_fold_activation_with_segment_weighted_condense (activation,
                                                                                       target,
                                                                                       folded_activation);

        return itty_feed_model_fold_activation_with_repeated_and (activation,
                                                                  target,
                                                                  folded_activation);
}

static bool
itty_feed_model_evaluate_output (itty_feed_model_t                  *model,
                                 itty_bit_string_list_t             *outputs,
                                 itty_bit_string_t                  *target,
                                 itty_feed_model_output_evaluation_t *evaluation)
{
        size_t selected_index = 0;

        if (!itty_network_select_output (outputs, &selected_index))
                return false;

        itty_bit_string_t *activation = itty_bit_string_list_fetch (outputs,
                                                                    selected_index);
        itty_bit_string_t *decoder_target =
                itty_feed_model_prepare_decoder_target_for_activation (model,
                                                                       activation,
                                                                       target,
                                                                       selected_index);
        itty_bit_string_t *folded_activation = NULL;
        if (!decoder_target ||
            !itty_feed_model_fold_activation_to_target_width (model,
                                                              activation,
                                                              decoder_target,
                                                              selected_index,
                                                              &folded_activation)) {
                if (decoder_target)
                        itty_bit_string_free (decoder_target);
                return false;
        }

        itty_bit_string_t *difference = itty_bit_string_exclusive_or (folded_activation,
                                                                      decoder_target);
        evaluation->distance = itty_bit_string_get_pop_count (difference);
        evaluation->selected_index = selected_index;
        evaluation->folded_activation = folded_activation == activation ?
                                        itty_feed_model_bit_string_clone (folded_activation) :
                                        folded_activation;
        itty_bit_string_free (difference);
        itty_bit_string_free (decoder_target);

        return true;
}

static void
itty_feed_model_summarize_residual_decode (itty_feed_model_output_evaluation_t     *evaluation,
                                           itty_bit_string_t                       *target,
                                           itty_feed_model_residual_decode_summary_t *summary)
{
        size_t target_bit_capacity = itty_bit_string_get_number_of_words (target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        *summary = (itty_feed_model_residual_decode_summary_t) {
                .distance = evaluation->distance,
                .selected_index = evaluation->selected_index
        };

        for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                bool folded_bit = itty_bit_string_get_bit (evaluation->folded_activation,
                                                           bit_index);
                bool target_bit = itty_bit_string_get_bit (target,
                                                           bit_index);

                if (folded_bit && !target_bit)
                        summary->false_positive_bits++;
                else if (!folded_bit && target_bit)
                        summary->false_negative_bits++;
        }
}

static void
itty_feed_model_count_decoder_ancestor_state (itty_feed_model_t *model,
                                              itty_bit_string_t *activation,
                                              itty_bit_string_t *target,
                                              size_t             target_bit_index,
                                              bool               target_bit,
                                              bool               folded_bit,
                                              size_t            *blocker_bits,
                                              size_t            *safety_bits,
                                              size_t            *positive_margin,
                                              size_t            *negative_excess)
{
        size_t activation_bit_capacity = itty_bit_string_get_length (activation);
        size_t target_bit_capacity = itty_bit_string_get_number_of_words (target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t zero_ancestors = 0;
        size_t ancestor_count = 0;

        *blocker_bits = 0;
        *safety_bits = 0;
        *positive_margin = 0;
        *negative_excess = 0;

        if (target_bit_capacity == 0)
                return;

        for (size_t output_bit_index = target_bit_index;
             output_bit_index < activation_bit_capacity;
             output_bit_index += target_bit_capacity) {
                ancestor_count++;
                if (!itty_bit_string_get_bit (activation, output_bit_index))
                        zero_ancestors++;
        }

        if (itty_feed_model_decoder_uses_segment_style (model->decoder)) {
                size_t ones = ancestor_count - zero_ancestors;
                size_t threshold = ancestor_count / 2 + 1;
                size_t max_ones_for_zero = threshold == 0 ? 0 : threshold - 1;

                if (model->decoder == ITTY_FEED_MODEL_DECODER_SEGMENT_WEIGHTED_CONDENSE) {
                        __uint128_t weighted_ones = 0;
                        __uint128_t total_votes = 0;

                        for (size_t output_bit_index = target_bit_index, segment_index = 0;
                             output_bit_index < activation_bit_capacity;
                             output_bit_index += target_bit_capacity, segment_index++) {
                                size_t vote = 1;
                                size_t segment_base = segment_index * target_bit_capacity;
                                for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                                        size_t weighted_bit_index = segment_base + bit_index;
                                        if (weighted_bit_index < activation_bit_capacity &&
                                            itty_bit_string_get_bit (activation, weighted_bit_index))
                                                vote++;
                                }
                                total_votes += vote;
                                if (itty_bit_string_get_bit (activation, output_bit_index))
                                        weighted_ones += vote;
                        }

                        ones = (size_t) weighted_ones;
                        threshold = (size_t) (total_votes / 2 + 1);
                        max_ones_for_zero = threshold == 0 ? 0 : threshold - 1;
                }

                if (target_bit && !folded_bit && ones < threshold)
                        *blocker_bits = threshold - ones;
                else if (target_bit && folded_bit)
                        *positive_margin = ones - threshold;
                else if (!target_bit && folded_bit && ones > max_ones_for_zero)
                        *negative_excess = ones - max_ones_for_zero;
                else if (!target_bit && !folded_bit && ones <= max_ones_for_zero)
                        *safety_bits = max_ones_for_zero - ones;
                return;
        }

        if (target_bit && !folded_bit)
                *blocker_bits = zero_ancestors;
        else if (!target_bit)
                *safety_bits = zero_ancestors;
}

static void
itty_feed_model_decoder_histogram_increment (size_t *histogram,
                                             size_t  value)
{
        size_t bucket = value;

        if (bucket >= ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS)
                bucket = ITTY_FEED_MODEL_DECODER_HISTOGRAM_OVERFLOW_BUCKET;

        histogram[bucket]++;
}

static void
itty_feed_model_decoder_objective_record_margins (itty_feed_model_decoder_objective_t *objective,
                                                  bool                                 target_bit,
                                                  bool                                 folded_bit,
                                                  size_t                               blocker_bits,
                                                  size_t                               safety_bits,
                                                  size_t                               positive_margin,
                                                  size_t                               negative_excess)
{
        if (target_bit && !folded_bit) {
                if (objective->false_negative_count == 0 ||
                    blocker_bits < objective->false_negative_vote_deficit_min)
                        objective->false_negative_vote_deficit_min = blocker_bits;
                if (blocker_bits > objective->false_negative_vote_deficit_max)
                        objective->false_negative_vote_deficit_max = blocker_bits;
                objective->false_negative_count++;
                objective->false_negative_blocker_bits += blocker_bits;
                objective->false_negative_vote_deficit += blocker_bits;
                itty_feed_model_decoder_histogram_increment (objective->false_negative_vote_deficit_histogram,
                                                             blocker_bits);
        } else if (target_bit) {
                objective->target_one_margin += positive_margin;
                itty_feed_model_decoder_histogram_increment (objective->target_one_margin_histogram,
                                                             positive_margin);
        } else if (folded_bit) {
                objective->false_positive_vote_excess += negative_excess;
                itty_feed_model_decoder_histogram_increment (objective->false_positive_vote_excess_histogram,
                                                             negative_excess);
        } else {
                if (objective->target_zero_safety_histogram[0] == 0 &&
                    objective->target_zero_safety == 0)
                        objective->target_zero_safety_min = safety_bits;
                else if (safety_bits < objective->target_zero_safety_min)
                        objective->target_zero_safety_min = safety_bits;
                objective->zero_veto_safety_bits += safety_bits;
                objective->target_zero_safety += safety_bits;
                itty_feed_model_decoder_histogram_increment (objective->target_zero_safety_histogram,
                                                             safety_bits);
        }
}

static void
itty_feed_model_decoder_objective_measure_selected_margins (itty_feed_model_t                  *model,
                                                            itty_bit_string_t                  *selected_activation,
                                                            itty_bit_string_t                  *folded_activation,
                                                            itty_bit_string_t                  *target,
                                                            itty_feed_model_decoder_objective_t *objective)
{
        size_t target_bit_capacity = itty_bit_string_get_number_of_words (target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        for (size_t target_bit_index = 0; target_bit_index < target_bit_capacity; target_bit_index++) {
                bool folded_bit = itty_bit_string_get_bit (folded_activation,
                                                           target_bit_index);
                bool target_bit = itty_bit_string_get_bit (target,
                                                           target_bit_index);
                size_t blocker_bits = 0;
                size_t safety_bits = 0;
                size_t positive_margin = 0;
                size_t negative_excess = 0;

                itty_feed_model_count_decoder_ancestor_state (model,
                                                              selected_activation,
                                                              target,
                                                              target_bit_index,
                                                              target_bit,
                                                              folded_bit,
                                                              &blocker_bits,
                                                              &safety_bits,
                                                              &positive_margin,
                                                              &negative_excess);
                itty_feed_model_decoder_objective_record_margins (objective,
                                                                  target_bit,
                                                                  folded_bit,
                                                                  blocker_bits,
                                                                  safety_bits,
                                                                  positive_margin,
                                                                  negative_excess);
        }
}

static bool
itty_feed_model_evaluate_decoder_objective (itty_feed_model_t                   *model,
                                            itty_bit_string_list_t              *outputs,
                                            itty_bit_string_t                   *target,
                                            itty_feed_model_decoder_objective_t *objective)
{
        itty_feed_model_output_evaluation_t evaluation = { 0 };

        if (!itty_feed_model_evaluate_output (model, outputs, target, &evaluation))
                return false;

        itty_bit_string_t *selected_activation = itty_bit_string_list_fetch (outputs,
                                                                             evaluation.selected_index);
        itty_bit_string_t *decoder_target =
                itty_feed_model_prepare_decoder_target_for_activation (model,
                                                                       selected_activation,
                                                                       target,
                                                                       evaluation.selected_index);
        if (!decoder_target) {
                itty_bit_string_free (evaluation.folded_activation);
                return false;
        }

        *objective = (itty_feed_model_decoder_objective_t) {
                .selected_distance = evaluation.distance,
                .selected_node = evaluation.selected_index,
                .selected_popcount = itty_bit_string_get_pop_count (selected_activation),
                .best_decoded_node = evaluation.selected_index,
                .best_decoded_distance = evaluation.distance
        };

        itty_feed_model_decoder_objective_measure_selected_margins (model,
                                                                    selected_activation,
                                                                    evaluation.folded_activation,
                                                                    decoder_target,
                                                                    objective);

        size_t nearest_wrong_distance = (size_t) -1;
        for (size_t node_index = 0; node_index < outputs->count; node_index++) {
                itty_bit_string_t *activation = itty_bit_string_list_fetch (outputs,
                                                                            node_index);
                itty_bit_string_t *folded_activation = NULL;

                if (!itty_feed_model_fold_activation_to_target_width (model,
                                                                      activation,
                                                                      decoder_target,
                                                                      node_index,
                                                                      &folded_activation))
                        continue;

                itty_bit_string_t *difference = itty_bit_string_exclusive_or (folded_activation,
                                                                              decoder_target);
                size_t distance = itty_bit_string_get_pop_count (difference);
                itty_bit_string_free (difference);
                if (folded_activation != activation)
                        itty_bit_string_free (folded_activation);

                if (distance < objective->best_decoded_distance) {
                        objective->best_decoded_node = node_index;
                        objective->best_decoded_distance = distance;
                }

                if (distance != 0 && distance < nearest_wrong_distance)
                        nearest_wrong_distance = distance;
        }

        itty_bit_string_free (evaluation.folded_activation);
        itty_bit_string_free (decoder_target);
        (void) model;

        if (nearest_wrong_distance != (size_t) -1 &&
            nearest_wrong_distance > objective->selected_distance)
                objective->nearest_wrong_margin = nearest_wrong_distance - objective->selected_distance;

        return true;
}

static bool
itty_feed_model_evaluate_decoder_objective_for_node (itty_feed_model_t                   *model,
                                                     itty_bit_string_list_t              *outputs,
                                                     itty_bit_string_t                   *target,
                                                     size_t                               selected_index,
                                                     itty_feed_model_decoder_objective_t *objective)
{
        if (selected_index >= outputs->count)
                return false;

        itty_bit_string_t *selected_activation = itty_bit_string_list_fetch (outputs,
                                                                             selected_index);
        itty_bit_string_t *decoder_target =
                itty_feed_model_prepare_decoder_target_for_activation (model,
                                                                       selected_activation,
                                                                       target,
                                                                       selected_index);
        itty_bit_string_t *folded_activation = NULL;
        if (!decoder_target ||
            !itty_feed_model_fold_activation_to_target_width (model,
                                                              selected_activation,
                                                              decoder_target,
                                                              selected_index,
                                                              &folded_activation)) {
                if (decoder_target)
                        itty_bit_string_free (decoder_target);
                return false;
        }

        itty_bit_string_t *difference = itty_bit_string_exclusive_or (folded_activation,
                                                                      decoder_target);
        size_t distance = itty_bit_string_get_pop_count (difference);
        itty_bit_string_free (difference);

        size_t target_bit_capacity = itty_bit_string_get_number_of_words (decoder_target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        *objective = (itty_feed_model_decoder_objective_t) {
                .selected_distance = distance,
                .selected_node = selected_index,
                .selected_popcount = itty_bit_string_get_pop_count (selected_activation),
                .best_decoded_node = selected_index,
                .best_decoded_distance = distance
        };

        for (size_t target_bit_index = 0; target_bit_index < target_bit_capacity; target_bit_index++) {
                bool folded_bit = itty_bit_string_get_bit (folded_activation,
                                                           target_bit_index);
                bool target_bit = itty_bit_string_get_bit (decoder_target,
                                                           target_bit_index);
                size_t blocker_bits = 0;
                size_t safety_bits = 0;
                size_t positive_margin = 0;
                size_t negative_excess = 0;

                itty_feed_model_count_decoder_ancestor_state (model,
                                                              selected_activation,
                                                              decoder_target,
                                                              target_bit_index,
                                                              target_bit,
                                                              folded_bit,
                                                              &blocker_bits,
                                                              &safety_bits,
                                                              &positive_margin,
                                                              &negative_excess);

                if (target_bit && !folded_bit) {
                        objective->false_negative_count++;
                        objective->false_negative_blocker_bits += blocker_bits;
                        objective->false_negative_vote_deficit += blocker_bits;
                } else if (target_bit) {
                        objective->target_one_margin += positive_margin;
                } else if (folded_bit) {
                        objective->false_positive_vote_excess += negative_excess;
                } else if (!target_bit) {
                        objective->zero_veto_safety_bits += safety_bits;
                        objective->target_zero_safety += safety_bits;
                }
        }

        size_t nearest_wrong_distance = (size_t) -1;
        for (size_t node_index = 0; node_index < outputs->count; node_index++) {
                itty_bit_string_t *activation = itty_bit_string_list_fetch (outputs,
                                                                            node_index);
                itty_bit_string_t *node_folded_activation = NULL;

                if (!itty_feed_model_fold_activation_to_target_width (model,
                                                                      activation,
                                                                      decoder_target,
                                                                      node_index,
                                                                      &node_folded_activation))
                        continue;

                itty_bit_string_t *node_difference = itty_bit_string_exclusive_or (node_folded_activation,
                                                                                   decoder_target);
                size_t node_distance = itty_bit_string_get_pop_count (node_difference);
                itty_bit_string_free (node_difference);
                if (node_folded_activation != activation)
                        itty_bit_string_free (node_folded_activation);

                if (node_distance < objective->best_decoded_distance) {
                        objective->best_decoded_node = node_index;
                        objective->best_decoded_distance = node_distance;
                }

                if (node_index != selected_index &&
                    node_distance != 0 &&
                    node_distance < nearest_wrong_distance)
                        nearest_wrong_distance = node_distance;
        }

        if (folded_activation != selected_activation)
                itty_bit_string_free (folded_activation);
        itty_bit_string_free (decoder_target);
        (void) model;

        if (nearest_wrong_distance != (size_t) -1 &&
            nearest_wrong_distance > objective->selected_distance)
                objective->nearest_wrong_margin = nearest_wrong_distance - objective->selected_distance;

        return true;
}

static bool
itty_feed_model_evaluate_decoder_objective_for_activation (itty_feed_model_t                   *model,
                                                           itty_bit_string_t                   *activation,
                                                           itty_bit_string_t                   *target,
                                                           size_t                               node_index,
                                                           itty_feed_model_decoder_objective_t *objective)
{
        itty_bit_string_t *decoder_target =
                itty_feed_model_prepare_decoder_target_for_activation (model,
                                                                       activation,
                                                                       target,
                                                                       node_index);
        itty_bit_string_t *folded_activation = NULL;

        if (!decoder_target ||
            !itty_feed_model_fold_activation_to_target_width (model,
                                                              activation,
                                                              decoder_target,
                                                              node_index,
                                                              &folded_activation)) {
                if (decoder_target)
                        itty_bit_string_free (decoder_target);
                return false;
        }

        itty_bit_string_t *difference = itty_bit_string_exclusive_or (folded_activation,
                                                                      decoder_target);
        size_t distance = itty_bit_string_get_pop_count (difference);
        itty_bit_string_free (difference);

        *objective = (itty_feed_model_decoder_objective_t) {
                .selected_distance = distance,
                .selected_node = node_index,
                .selected_popcount = itty_bit_string_get_pop_count (activation),
                .best_decoded_node = node_index,
                .best_decoded_distance = distance
        };

        itty_feed_model_decoder_objective_measure_selected_margins (model,
                                                                    activation,
                                                                    folded_activation,
                                                                    decoder_target,
                                                                    objective);
        if (folded_activation != activation)
                itty_bit_string_free (folded_activation);
        itty_bit_string_free (decoder_target);

        return true;
}

static bool
itty_feed_model_measure_route_margin_state (itty_feed_model_t                   *model,
                                            itty_bit_string_list_t              *input,
                                            itty_bit_string_t                   *target,
                                            size_t                               owner_route,
                                            itty_feed_model_decoder_objective_t *owner_objective,
                                            itty_feed_model_decoder_objective_t *global_objective,
                                            ptrdiff_t                           *margin,
                                            size_t                              *competitor_route)
{
        if (!itty_feed_model_measure_decoder_objective_for_node (model,
                                                                 input,
                                                                 target,
                                                                 owner_route,
                                                                 owner_objective) ||
            !itty_feed_model_measure_decoder_objective (model,
                                                       input,
                                                       target,
                                                       global_objective))
                return false;

        size_t route_popcount = owner_objective->selected_popcount;
        size_t max_other_popcount = 0;
        size_t max_other_route = owner_route;

        for (size_t route = 0; route < model->nodes_per_layer; route++) {
                if (route == owner_route)
                        continue;

                itty_feed_model_decoder_objective_t route_objective = { 0 };
                if (!itty_feed_model_measure_decoder_objective_for_node (model,
                                                                         input,
                                                                         target,
                                                                         route,
                                                                         &route_objective))
                        continue;
                if (route_objective.selected_popcount > max_other_popcount) {
                        max_other_popcount = route_objective.selected_popcount;
                        max_other_route = route;
                }
        }

        if (margin)
                *margin = (ptrdiff_t) route_popcount - (ptrdiff_t) max_other_popcount;
        if (competitor_route)
                *competitor_route = max_other_route;
        return true;
}

static void
itty_feed_model_lane_range_resolve (size_t  bit_capacity,
                                    size_t  bit_offset,
                                    size_t  bit_count,
                                    size_t *range_start,
                                    size_t *range_end)
{
        size_t start = bit_offset;
        size_t end = bit_capacity;

        if (start > bit_capacity)
                start = bit_capacity;
        if (bit_count > 0) {
                end = start + bit_count;
                if (end > bit_capacity)
                        end = bit_capacity;
        }

        if (range_start)
                *range_start = start;
        if (range_end)
                *range_end = end;
}

static bool
itty_feed_model_train_options_has_lane_split (itty_feed_model_train_options_t const *options)
{
        return options &&
               (options->selector_lane_bit_count > 0 ||
                options->decoder_lane_bit_count > 0);
}

static bool
itty_feed_model_train_options_bit_allowed (itty_feed_model_train_options_t const *options,
                                           size_t                                 bit_index,
                                           size_t                                 bit_capacity,
                                           bool                                   selector_lane)
{
        size_t range_start = 0;
        size_t range_end = bit_capacity;

        if (!itty_feed_model_train_options_has_lane_split (options))
                return true;

        itty_feed_model_lane_range_resolve (bit_capacity,
                                            selector_lane ? options->selector_lane_bit_offset :
                                                            options->decoder_lane_bit_offset,
                                            selector_lane ? options->selector_lane_bit_count :
                                                            options->decoder_lane_bit_count,
                                            &range_start,
                                            &range_end);
        return bit_index >= range_start && bit_index < range_end;
}

static itty_bit_string_t *
itty_feed_model_clone_bit_string_lane_range (itty_bit_string_t *source,
                                             size_t             bit_offset,
                                             size_t             bit_count)
{
        size_t bit_capacity = itty_bit_string_get_length (source);
        size_t range_start = 0;
        size_t range_end = bit_capacity;
        itty_bit_string_t *clone = itty_feed_model_bit_string_clone_to_words (source,
                                                                              itty_bit_string_get_number_of_words (source));

        if (!clone)
                return NULL;

        itty_feed_model_lane_range_resolve (bit_capacity,
                                            bit_offset,
                                            bit_count,
                                            &range_start,
                                            &range_end);
        for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                if (bit_index >= range_start &&
                    bit_index < range_end)
                        continue;
                itty_bit_string_set_bit (clone, bit_index, false);
        }
        clone->pop_count_computed = false;
        return clone;
}

static itty_bit_string_list_t *
itty_feed_model_clone_outputs_lane_range (itty_bit_string_list_t *outputs,
                                          size_t                  bit_offset,
                                          size_t                  bit_count)
{
        itty_bit_string_list_t *cloned = itty_bit_string_list_new ();

        if (!cloned)
                return NULL;

        for (size_t index = 0; index < outputs->count; index++) {
                itty_bit_string_t *activation = itty_bit_string_list_fetch (outputs, index);
                itty_bit_string_t *masked = itty_feed_model_clone_bit_string_lane_range (activation,
                                                                                         bit_offset,
                                                                                         bit_count);
                if (!masked) {
                        itty_bit_string_list_free (cloned);
                        return NULL;
                }
                itty_bit_string_list_append (cloned, masked);
        }

        return cloned;
}

static bool
itty_feed_model_measure_decoder_objective_with_lane_split_from_outputs (itty_feed_model_t                   *model,
                                                                        itty_bit_string_list_t              *outputs,
                                                                        itty_bit_string_t                   *target,
                                                                        size_t                               selector_lane_bit_offset,
                                                                        size_t                               selector_lane_bit_count,
                                                                        size_t                               decoder_lane_bit_offset,
                                                                        size_t                               decoder_lane_bit_count,
                                                                        itty_feed_model_decoder_objective_t *objective)
{
        itty_bit_string_list_t *selector_outputs = NULL;
        itty_bit_string_list_t *decoder_outputs = NULL;
        size_t selected_index = 0;
        size_t selected_popcount = 0;

        selector_outputs = itty_feed_model_clone_outputs_lane_range (outputs,
                                                                     selector_lane_bit_offset,
                                                                     selector_lane_bit_count);
        decoder_outputs = itty_feed_model_clone_outputs_lane_range (outputs,
                                                                    decoder_lane_bit_offset,
                                                                    decoder_lane_bit_count);
        if (!selector_outputs || !decoder_outputs) {
                if (selector_outputs)
                        itty_bit_string_list_free (selector_outputs);
                if (decoder_outputs)
                        itty_bit_string_list_free (decoder_outputs);
                return false;
        }

        for (size_t node_index = 0; node_index < selector_outputs->count; node_index++) {
                itty_bit_string_t *activation = itty_bit_string_list_fetch (selector_outputs, node_index);
                size_t popcount = itty_bit_string_get_pop_count (activation);

                if (node_index == 0 || popcount > selected_popcount) {
                        selected_index = node_index;
                        selected_popcount = popcount;
                }
        }

        bool measured = itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                             decoder_outputs,
                                                                             target,
                                                                             selected_index,
                                                                             objective);
        if (measured)
                objective->selected_popcount = selected_popcount;

        itty_bit_string_list_free (selector_outputs);
        itty_bit_string_list_free (decoder_outputs);
        return measured;
}

static bool
itty_feed_model_measure_decoder_objective_for_node_with_lane_split_from_outputs (itty_feed_model_t                   *model,
                                                                                 itty_bit_string_list_t              *outputs,
                                                                                 itty_bit_string_t                   *target,
                                                                                 size_t                               selected_node,
                                                                                 size_t                               selector_lane_bit_offset,
                                                                                 size_t                               selector_lane_bit_count,
                                                                                 size_t                               decoder_lane_bit_offset,
                                                                                 size_t                               decoder_lane_bit_count,
                                                                                 itty_feed_model_decoder_objective_t *objective)
{
        itty_bit_string_list_t *selector_outputs = NULL;
        itty_bit_string_list_t *decoder_outputs = NULL;
        bool measured;

        selector_outputs = itty_feed_model_clone_outputs_lane_range (outputs,
                                                                     selector_lane_bit_offset,
                                                                     selector_lane_bit_count);
        decoder_outputs = itty_feed_model_clone_outputs_lane_range (outputs,
                                                                    decoder_lane_bit_offset,
                                                                    decoder_lane_bit_count);
        if (!selector_outputs || !decoder_outputs) {
                if (selector_outputs)
                        itty_bit_string_list_free (selector_outputs);
                if (decoder_outputs)
                        itty_bit_string_list_free (decoder_outputs);
                return false;
        }

        measured = itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                        decoder_outputs,
                                                                        target,
                                                                        selected_node,
                                                                        objective);
        if (measured) {
                itty_bit_string_t *selector_activation = itty_bit_string_list_fetch (selector_outputs,
                                                                                     selected_node);
                objective->selected_popcount = itty_bit_string_get_pop_count (selector_activation);
        }

        itty_bit_string_list_free (selector_outputs);
        itty_bit_string_list_free (decoder_outputs);
        return measured;
}

static bool
itty_feed_model_measure_route_margin_state_with_options (itty_feed_model_t                     *model,
                                                         itty_bit_string_list_t                *input,
                                                         itty_bit_string_t                     *target,
                                                         size_t                                 owner_route,
                                                         itty_feed_model_train_options_t const *options,
                                                         itty_feed_model_decoder_objective_t   *owner_objective,
                                                         itty_feed_model_decoder_objective_t   *global_objective,
                                                         ptrdiff_t                             *margin,
                                                         size_t                                *competitor_route)
{
        if (itty_feed_model_train_options_has_lane_split (options)) {
                if (!itty_feed_model_measure_decoder_objective_for_node_with_lane_split (model,
                                                                                         input,
                                                                                         target,
                                                                                         owner_route,
                                                                                         options->selector_lane_bit_offset,
                                                                                         options->selector_lane_bit_count,
                                                                                         options->decoder_lane_bit_offset,
                                                                                         options->decoder_lane_bit_count,
                                                                                         owner_objective) ||
                    !itty_feed_model_measure_decoder_objective_with_lane_split (model,
                                                                                input,
                                                                                target,
                                                                                options->selector_lane_bit_offset,
                                                                                options->selector_lane_bit_count,
                                                                                options->decoder_lane_bit_offset,
                                                                                options->decoder_lane_bit_count,
                                                                                global_objective))
                        return false;

                size_t route_popcount = owner_objective->selected_popcount;
                size_t max_other_popcount = 0;
                size_t max_other_route = owner_route;

                for (size_t route = 0; route < model->nodes_per_layer; route++) {
                        itty_feed_model_decoder_objective_t route_objective = { 0 };

                        if (route == owner_route)
                                continue;
                        if (!itty_feed_model_measure_decoder_objective_for_node_with_lane_split (model,
                                                                                                 input,
                                                                                                 target,
                                                                                                 route,
                                                                                                 options->selector_lane_bit_offset,
                                                                                                 options->selector_lane_bit_count,
                                                                                                 options->decoder_lane_bit_offset,
                                                                                                 options->decoder_lane_bit_count,
                                                                                                 &route_objective))
                                continue;
                        if (route_objective.selected_popcount > max_other_popcount) {
                                max_other_popcount = route_objective.selected_popcount;
                                max_other_route = route;
                        }
                }

                if (margin)
                        *margin = (ptrdiff_t) route_popcount - (ptrdiff_t) max_other_popcount;
                if (competitor_route)
                        *competitor_route = max_other_route;
                return true;
        }

        return itty_feed_model_measure_route_margin_state (model,
                                                           input,
                                                           target,
                                                           owner_route,
                                                           owner_objective,
                                                           global_objective,
                                                           margin,
                                                           competitor_route);
}

static bool
itty_feed_model_decoder_objective_is_better (itty_feed_model_decoder_objective_t const *candidate,
                                             itty_feed_model_decoder_objective_t const *current)
{
        if (candidate->selected_distance < current->selected_distance)
                return true;
        if (candidate->selected_distance > current->selected_distance)
                return false;
        if (candidate->false_negative_vote_deficit < current->false_negative_vote_deficit)
                return true;
        if (candidate->false_negative_vote_deficit > current->false_negative_vote_deficit)
                return false;
        if (candidate->false_positive_vote_excess < current->false_positive_vote_excess)
                return true;
        if (candidate->false_positive_vote_excess > current->false_positive_vote_excess)
                return false;
        if (candidate->target_one_margin > current->target_one_margin)
                return true;
        if (candidate->target_one_margin < current->target_one_margin)
                return false;
        return candidate->target_zero_safety > current->target_zero_safety;
}

static bool
itty_feed_model_route_local_balanced_objective_is_better (itty_feed_model_decoder_objective_t const *candidate,
                                                          itty_feed_model_decoder_objective_t const *current)
{
        if (candidate->selected_distance < current->selected_distance)
                return true;
        if (candidate->selected_distance > current->selected_distance)
                return false;
        if (candidate->false_positive_vote_excess < current->false_positive_vote_excess)
                return true;
        if (candidate->false_positive_vote_excess > current->false_positive_vote_excess)
                return false;
        if (candidate->false_negative_vote_deficit < current->false_negative_vote_deficit)
                return true;
        if (candidate->false_negative_vote_deficit > current->false_negative_vote_deficit)
                return false;
        if (candidate->target_one_margin > current->target_one_margin)
                return true;
        if (candidate->target_one_margin < current->target_one_margin)
                return false;
        return candidate->target_zero_safety > current->target_zero_safety;
}

static bool
itty_feed_model_decoder_objective_accepts (itty_feed_model_decoder_objective_t *before,
                                           itty_feed_model_decoder_objective_t *candidate)
{
        if (candidate->selected_distance > before->selected_distance)
                return false;
        if (candidate->selected_distance < before->selected_distance)
                return true;

        if (candidate->false_negative_blocker_bits < before->false_negative_blocker_bits)
                return true;
        if (candidate->false_negative_blocker_bits > before->false_negative_blocker_bits)
                return false;

        if (candidate->false_positive_vote_excess < before->false_positive_vote_excess)
                return true;
        if (candidate->false_positive_vote_excess > before->false_positive_vote_excess)
                return false;

        if (candidate->target_one_margin > before->target_one_margin)
                return true;
        if (candidate->target_one_margin < before->target_one_margin)
                return false;

        if (candidate->nearest_wrong_margin > before->nearest_wrong_margin)
                return true;
        if (candidate->nearest_wrong_margin < before->nearest_wrong_margin)
                return false;

        return candidate->zero_veto_safety_bits > before->zero_veto_safety_bits;
}

static bool
itty_feed_model_run_suffix_outputs (itty_feed_model_t      *model,
                                    itty_bit_string_list_t *layer_outputs,
                                    size_t                  layer_index,
                                    itty_bit_string_list_t **final_outputs)
{
        itty_bit_string_list_t *current_input = layer_outputs;

        for (size_t next_layer_index = layer_index + 1; next_layer_index < model->number_of_layers; next_layer_index++) {
                itty_bit_string_list_t *next_output = itty_feed_model_run_layer (model,
                                                                                 next_layer_index,
                                                                                 current_input);
                if (current_input != layer_outputs)
                        itty_bit_string_list_free (current_input);
                current_input = next_output;
        }

        *final_outputs = current_input == layer_outputs ?
                         itty_feed_model_bit_string_list_clone (current_input) :
                         current_input;
        return *final_outputs != NULL;
}

static bool
itty_feed_model_evaluate_suffix_output (itty_feed_model_t                 *model,
                                        itty_bit_string_list_t            *layer_outputs,
                                        size_t                             layer_index,
                                        itty_bit_string_t                 *target,
                                        itty_feed_model_output_evaluation_t *evaluation)
{
        itty_bit_string_list_t *current_input = layer_outputs;

        for (size_t next_layer_index = layer_index + 1; next_layer_index < model->number_of_layers; next_layer_index++) {
                itty_bit_string_list_t *next_output = itty_feed_model_run_layer (model,
                                                                                 next_layer_index,
                                                                                 current_input);
                if (current_input != layer_outputs)
                        itty_bit_string_list_free (current_input);
                current_input = next_output;
        }

        bool evaluated = itty_feed_model_evaluate_output (model,
                                                          current_input,
                                                          target,
                                                          evaluation);
        if (current_input != layer_outputs)
                itty_bit_string_list_free (current_input);

        return evaluated;
}

static bool
itty_feed_model_fold_suffix_selected_output (itty_feed_model_t      *model,
                                             itty_bit_string_list_t *layer_outputs,
                                             size_t                  layer_index,
                                             itty_bit_string_t      *target,
                                             itty_bit_string_t     **folded_activation)
{
        itty_feed_model_output_evaluation_t evaluation = { 0 };

        if (!itty_feed_model_evaluate_suffix_output (model,
                                                     layer_outputs,
                                                     layer_index,
                                                     target,
                                                     &evaluation))
                return false;

        *folded_activation = evaluation.folded_activation;
        return true;
}

static void
itty_feed_model_measure_decoded_bit_transitions (itty_bit_string_t                         *before_folded,
                                                 itty_bit_string_t                         *after_folded,
                                                 itty_bit_string_t                         *target,
                                                 itty_feed_model_projected_repair_stats_t  *stats)
{
        size_t target_bit_capacity = itty_bit_string_get_number_of_words (target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                bool before_bit = itty_bit_string_get_bit (before_folded,
                                                           bit_index);
                bool after_bit = itty_bit_string_get_bit (after_folded,
                                                          bit_index);
                bool target_bit = itty_bit_string_get_bit (target,
                                                           bit_index);
                bool before_correct = before_bit == target_bit;
                bool after_correct = after_bit == target_bit;

                if (before_correct && after_correct) {
                        stats->previous_layer_harm_unchanged_correct_bits++;
                } else if (!before_correct && !after_correct) {
                        if (!target_bit && before_bit && !after_bit)
                                stats->previous_layer_harm_false_positive_to_false_negative_bits++;
                        else if (target_bit && !before_bit && after_bit)
                                stats->previous_layer_harm_false_negative_to_false_positive_bits++;
                        else
                                stats->previous_layer_harm_unchanged_wrong_bits++;
                } else if (before_correct) {
                        if (!target_bit)
                                stats->previous_layer_harm_correct_zero_to_false_positive_bits++;
                        else
                                stats->previous_layer_harm_correct_one_to_false_negative_bits++;
                } else {
                        if (!target_bit)
                                stats->previous_layer_harm_false_positive_to_correct_zero_bits++;
                        else
                                stats->previous_layer_harm_false_negative_to_correct_one_bits++;
                }
        }
}

static bool
itty_feed_model_preserves_correct_decoded_bits (itty_bit_string_t *before_folded,
                                                itty_bit_string_t *after_folded,
                                                itty_bit_string_t *target)
{
        size_t target_bit_capacity = itty_bit_string_get_number_of_words (target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                bool before_bit = itty_bit_string_get_bit (before_folded,
                                                           bit_index);
                bool after_bit = itty_bit_string_get_bit (after_folded,
                                                          bit_index);
                bool target_bit = itty_bit_string_get_bit (target,
                                                           bit_index);

                if (before_bit == target_bit &&
                    after_bit != target_bit)
                        return false;
        }

        return true;
}

static bool
itty_feed_model_evaluate_suffix_decoder_objective (itty_feed_model_t                   *model,
                                                   itty_bit_string_list_t              *layer_outputs,
                                                   size_t                               layer_index,
                                                   itty_bit_string_t                   *target,
                                                   itty_feed_model_decoder_objective_t *objective)
{
        itty_bit_string_list_t *current_input = layer_outputs;

        for (size_t next_layer_index = layer_index + 1; next_layer_index < model->number_of_layers; next_layer_index++) {
                itty_bit_string_list_t *next_output = itty_feed_model_run_layer (model,
                                                                                 next_layer_index,
                                                                                 current_input);
                if (current_input != layer_outputs)
                        itty_bit_string_list_free (current_input);
                current_input = next_output;
        }

        bool evaluated = itty_feed_model_evaluate_decoder_objective (model,
                                                                     current_input,
                                                                     target,
                                                                     objective);
        if (current_input != layer_outputs)
                itty_bit_string_list_free (current_input);

        return evaluated;
}

static bool
itty_feed_model_evaluate_suffix_decoder_objective_for_node (itty_feed_model_t                   *model,
                                                            itty_bit_string_list_t              *layer_outputs,
                                                            size_t                               layer_index,
                                                            itty_bit_string_t                   *target,
                                                            size_t                               selected_node,
                                                            itty_feed_model_decoder_objective_t *objective)
{
        itty_bit_string_list_t *current_input = layer_outputs;

        for (size_t next_layer_index = layer_index + 1; next_layer_index < model->number_of_layers; next_layer_index++) {
                itty_bit_string_list_t *next_output = itty_feed_model_run_layer (model,
                                                                                 next_layer_index,
                                                                                 current_input);
                if (current_input != layer_outputs)
                        itty_bit_string_list_free (current_input);
                current_input = next_output;
        }

        bool evaluated = itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                              current_input,
                                                                              target,
                                                                              selected_node,
                                                                              objective);
        if (current_input != layer_outputs)
                itty_bit_string_list_free (current_input);

        return evaluated;
}

static size_t
itty_feed_model_trace_expanded_bit_to_layer (itty_feed_model_t *model,
                                             size_t             layer_index,
                                             size_t             bit_index)
{
        size_t source_bit_index = bit_index;

        for (size_t layer_count = model->number_of_layers; layer_count > layer_index; layer_count--) {
                size_t expansion_layer = layer_count - 1;
                size_t previous_bit_capacity = (model->vocabulary_words << expansion_layer) *
                                               ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

                if (source_bit_index >= previous_bit_capacity) {
                        size_t rotation = model->rotations_by_layer[expansion_layer] % previous_bit_capacity;

                        source_bit_index -= previous_bit_capacity;
                        source_bit_index = (source_bit_index + previous_bit_capacity - rotation) %
                                           previous_bit_capacity;
                }
        }

        return source_bit_index;
}

static size_t
itty_feed_model_trace_output_bit_to_condensed (itty_feed_model_t *model,
                                               size_t             layer_index,
                                               size_t             output_bit_index)
{
        size_t condensed_bit_capacity = (model->vocabulary_words << layer_index) *
                                        ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        if (output_bit_index < condensed_bit_capacity)
                return output_bit_index;

        size_t rotation = model->rotations_by_layer[layer_index] % condensed_bit_capacity;
        size_t second_half_bit_index = output_bit_index - condensed_bit_capacity;

        return (second_half_bit_index + condensed_bit_capacity - rotation) %
               condensed_bit_capacity;
}

static bool
itty_feed_model_final_layer_apply_repeated_and_false_negative_block (itty_feed_model_t *model,
                                                                     size_t             target_bit_index,
                                                                     itty_bit_string_t *candidate_condensed)
{
        size_t final_layer = model->number_of_layers - 1;
        size_t condensed_bit_capacity = itty_bit_string_get_length (candidate_condensed);
        size_t output_bit_capacity = condensed_bit_capacity * 2;
        size_t target_bit_capacity = model->vocabulary_words *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        bool changed = false;

        for (size_t output_bit_index = target_bit_index;
             output_bit_index < output_bit_capacity;
             output_bit_index += target_bit_capacity) {
                size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                            final_layer,
                                                                                            output_bit_index);
                if (!itty_bit_string_get_bit (candidate_condensed, condensed_bit_index)) {
                        itty_bit_string_set_bit (candidate_condensed,
                                                 condensed_bit_index,
                                                 true);
                        changed = true;
                }
        }

        return changed;
}

static bool
itty_feed_model_final_layer_apply_repeated_and_false_positive_block (itty_feed_model_t *model,
                                                                     size_t             target_bit_index,
                                                                     itty_bit_string_t *candidate_condensed)
{
        size_t final_layer = model->number_of_layers - 1;
        size_t condensed_bit_capacity = itty_bit_string_get_length (candidate_condensed);
        size_t output_bit_capacity = condensed_bit_capacity * 2;
        size_t target_bit_capacity = model->vocabulary_words *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        for (size_t output_bit_index = target_bit_index;
             output_bit_index < output_bit_capacity;
             output_bit_index += target_bit_capacity) {
                size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                            final_layer,
                                                                                            output_bit_index);
                if (itty_bit_string_get_bit (candidate_condensed, condensed_bit_index)) {
                        itty_bit_string_set_bit (candidate_condensed,
                                                 condensed_bit_index,
                                                 false);
                        return true;
                }
        }

        return false;
}

static bool
itty_feed_model_final_layer_apply_segment_condense_block (itty_feed_model_t *model,
                                                          size_t             target_bit_index,
                                                          bool               target_bit,
                                                          itty_bit_string_t *candidate_condensed)
{
        size_t final_layer = model->number_of_layers - 1;
        size_t condensed_bit_capacity = itty_bit_string_get_length (candidate_condensed);
        size_t output_bit_capacity = condensed_bit_capacity * 2;
        size_t target_bit_capacity = model->vocabulary_words *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t segment_count = output_bit_capacity / target_bit_capacity;
        size_t threshold = segment_count / 2 + 1;
        size_t max_ones_for_zero = threshold - 1;
        size_t ones = 0;
        bool changed = false;

        for (size_t output_bit_index = target_bit_index;
             output_bit_index < output_bit_capacity;
             output_bit_index += target_bit_capacity) {
                size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                            final_layer,
                                                                                            output_bit_index);
                if (itty_bit_string_get_bit (candidate_condensed,
                                             condensed_bit_index))
                        ones++;
        }

        if (target_bit) {
                if (ones >= threshold)
                        return false;

                size_t flips_needed = threshold - ones;
                for (size_t output_bit_index = target_bit_index;
                     output_bit_index < output_bit_capacity && flips_needed > 0;
                     output_bit_index += target_bit_capacity) {
                        size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                                    final_layer,
                                                                                                    output_bit_index);
                        if (!itty_bit_string_get_bit (candidate_condensed,
                                                      condensed_bit_index)) {
                                itty_bit_string_set_bit (candidate_condensed,
                                                         condensed_bit_index,
                                                         true);
                                flips_needed--;
                                changed = true;
                        }
                }
        } else {
                if (ones <= max_ones_for_zero)
                        return false;

                size_t flips_needed = ones - max_ones_for_zero;
                for (size_t output_bit_index = target_bit_index;
                     output_bit_index < output_bit_capacity && flips_needed > 0;
                     output_bit_index += target_bit_capacity) {
                        size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                                    final_layer,
                                                                                                    output_bit_index);
                        if (itty_bit_string_get_bit (candidate_condensed,
                                                     condensed_bit_index)) {
                                itty_bit_string_set_bit (candidate_condensed,
                                                         condensed_bit_index,
                                                         false);
                                flips_needed--;
                                changed = true;
                        }
                }
        }

        return changed;
}

static bool
itty_feed_model_final_layer_apply_decoder_block (itty_feed_model_t *model,
                                                 size_t             target_bit_index,
                                                 bool               target_bit,
                                                 itty_bit_string_t *candidate_condensed)
{
        if (model->decoder == ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE)
                return itty_feed_model_final_layer_apply_segment_condense_block (model,
                                                                                 target_bit_index,
                                                                                 target_bit,
                                                                                 candidate_condensed);

        return target_bit ?
               itty_feed_model_final_layer_apply_repeated_and_false_negative_block (model,
                                                                                    target_bit_index,
                                                                                    candidate_condensed) :
               itty_feed_model_final_layer_apply_repeated_and_false_positive_block (model,
                                                                                    target_bit_index,
                                                                                   candidate_condensed);
}

static bool
itty_feed_model_get_segment_condense_vote_need (itty_feed_model_t *model,
                                                size_t             target_bit_index,
                                                bool               target_bit,
                                                itty_bit_string_t *candidate_condensed,
                                                size_t            *votes_needed)
{
        size_t final_layer = model->number_of_layers - 1;
        size_t condensed_bit_capacity = itty_bit_string_get_length (candidate_condensed);
        size_t output_bit_capacity = condensed_bit_capacity * 2;
        size_t target_bit_capacity = model->vocabulary_words *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t segment_count = output_bit_capacity / target_bit_capacity;
        size_t threshold = segment_count / 2 + 1;
        size_t max_ones_for_zero = threshold - 1;
        size_t ones = 0;

        for (size_t output_bit_index = target_bit_index;
             output_bit_index < output_bit_capacity;
             output_bit_index += target_bit_capacity) {
                size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                            final_layer,
                                                                                            output_bit_index);
                if (itty_bit_string_get_bit (candidate_condensed,
                                             condensed_bit_index))
                        ones++;
        }

        if (target_bit) {
                if (ones >= threshold)
                        return false;
                *votes_needed = threshold - ones;
        } else {
                if (ones <= max_ones_for_zero)
                        return false;
                *votes_needed = ones - max_ones_for_zero;
        }

        return true;
}

static bool
itty_feed_model_apply_final_output_clear_set (itty_feed_model_t *model,
                                              size_t             layer_index,
                                              itty_bit_string_t *candidate_condensed,
                                              size_t const      *output_bits,
                                              size_t             output_bit_count)
{
        bool changed = false;

        for (size_t output_index = 0; output_index < output_bit_count; output_index++) {
                size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                            layer_index,
                                                                                            output_bits[output_index]);
                if (itty_bit_string_get_bit (candidate_condensed,
                                             condensed_bit_index)) {
                        itty_bit_string_set_bit (candidate_condensed,
                                                 condensed_bit_index,
                                                 false);
                        changed = true;
                }
        }

        return changed;
}

static bool
itty_feed_model_collect_selected_node_clear_set (itty_feed_model_t       *model,
                                                 size_t                   layer_index,
                                                 itty_bit_string_t       *target,
                                                 itty_bit_string_t       *selected_condensed,
                                                 size_t const            *decoded_bits,
                                                 size_t                   decoded_bit_count,
                                                 size_t                  *output_bits,
                                                 size_t                  *output_bit_count)
{
        size_t target_bit_capacity;
        size_t condensed_bit_capacity;
        size_t output_bit_capacity;
        size_t clear_count = 0;

        if (!model || !target || !selected_condensed || !output_bits || !output_bit_count)
                return false;

        target_bit_capacity = itty_bit_string_get_number_of_words (target) *
                              ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        condensed_bit_capacity = itty_bit_string_get_length (selected_condensed);
        output_bit_capacity = condensed_bit_capacity * 2;

        for (size_t decoded_index = 0; decoded_index < decoded_bit_count; decoded_index++) {
                size_t decoded_bit = decoded_bits[decoded_index];
                size_t votes_needed = 0;
                size_t available_votes = 0;
                size_t available_output_bits[ITTY_FEED_MODEL_RESTORE_CLEAR_TRACE_LIMIT] = { 0 };

                if (decoded_bit >= target_bit_capacity)
                        return false;
                if (itty_bit_string_get_bit (target, decoded_bit))
                        return false;
                if (!itty_feed_model_get_segment_condense_vote_need (model,
                                                                    decoded_bit,
                                                                    false,
                                                                    selected_condensed,
                                                                    &votes_needed))
                        return false;

                for (size_t output_bit_index = decoded_bit;
                     output_bit_index < output_bit_capacity;
                     output_bit_index += target_bit_capacity) {
                        size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                                    layer_index,
                                                                                                    output_bit_index);
                        if (!itty_bit_string_get_bit (selected_condensed, condensed_bit_index))
                                continue;
                        if (available_votes < ITTY_FEED_MODEL_RESTORE_CLEAR_TRACE_LIMIT)
                                available_output_bits[available_votes++] = output_bit_index;
                }

                if (available_votes < votes_needed)
                        return false;

                for (size_t vote_index = 0; vote_index < votes_needed; vote_index++)
                        output_bits[clear_count++] = available_output_bits[vote_index];
        }

        *output_bit_count = clear_count;
        return clear_count > 0;
}

static void
itty_feed_model_measure_output_selection (itty_bit_string_list_t *outputs,
                                          size_t                 *selected_index,
                                          size_t                 *selection_gap)
{
        size_t best_index = 0;
        size_t best_popcount = 0;
        size_t runner_up = 0;

        for (size_t node_index = 0; node_index < outputs->count; node_index++) {
                size_t popcount = itty_bit_string_get_pop_count (itty_bit_string_list_fetch (outputs, node_index));

                if (node_index == 0 || popcount > best_popcount) {
                        runner_up = best_popcount;
                        best_popcount = popcount;
                        best_index = node_index;
                } else if (popcount > runner_up) {
                        runner_up = popcount;
                }
        }

        if (selected_index)
                *selected_index = best_index;
        if (selection_gap)
                *selection_gap = best_popcount - runner_up;
}

static bool
itty_feed_model_measure_or_train_selected_node_direct_clear_set (itty_feed_model_t                             *model,
                                                                 itty_bit_string_list_t                        *input,
                                                                 itty_bit_string_t                             *target,
                                                                 size_t const                                  *decoded_bits,
                                                                 size_t                                         decoded_bit_count,
                                                                 size_t                                         compare_node_a,
                                                                 size_t                                         compare_node_b,
                                                                 itty_feed_model_train_options_t const         *options,
                                                                 bool                                           train_selected_node,
                                                                 itty_feed_model_selected_clear_set_summary_t *summary)
{
        size_t final_layer;
        itty_bit_string_list_t *final_layer_input = NULL;
        itty_bit_string_list_t *final_outputs = NULL;
        itty_bit_string_list_t *candidate_outputs = NULL;
        itty_bit_string_t *selected_condensed = NULL;
        itty_bit_string_t *candidate_condensed = NULL;
        itty_feed_model_output_evaluation_t evaluation = { 0 };
        size_t output_bits[ITTY_FEED_MODEL_RESTORE_CLEAR_TRACE_LIMIT] = { 0 };
        size_t output_bit_count = 0;
        bool ok = false;

        if (summary)
                *summary = (itty_feed_model_selected_clear_set_summary_t) { 0 };
        if (!model ||
            !input ||
            !target ||
            !decoded_bits ||
            decoded_bit_count == 0 ||
            !summary ||
            model->number_of_layers == 0 ||
            compare_node_a >= model->nodes_per_layer ||
            compare_node_b >= model->nodes_per_layer ||
            (train_selected_node && !options))
                return false;

        final_layer = model->number_of_layers - 1;
        final_layer_input = itty_feed_model_run_to_layer_input (model, input, final_layer);
        final_outputs = itty_feed_model_run_layer (model, final_layer, final_layer_input);
        ok = final_outputs &&
             itty_feed_model_evaluate_output (model, final_outputs, target, &evaluation);
        if (!ok)
                goto done;

        summary->selected_node_before = evaluation.selected_index;
        summary->compare_node_a = compare_node_a;
        summary->compare_node_a_popcount_before =
                itty_bit_string_get_pop_count (itty_bit_string_list_fetch (final_outputs, compare_node_a));
        summary->compare_node_b = compare_node_b;
        summary->compare_node_b_popcount_before =
                itty_bit_string_get_pop_count (itty_bit_string_list_fetch (final_outputs, compare_node_b));
        selected_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                 model->masks_by_layer_node[final_layer][evaluation.selected_index]);
        if (!selected_condensed)
                goto done;
        if (!itty_feed_model_collect_selected_node_clear_set (model,
                                                              final_layer,
                                                              target,
                                                              selected_condensed,
                                                              decoded_bits,
                                                              decoded_bit_count,
                                                              output_bits,
                                                              &output_bit_count))
                goto done;

        summary->clear_vote_count = output_bit_count;
        candidate_outputs = itty_feed_model_bit_string_list_clone (final_outputs);
        if (!candidate_outputs)
                goto done;

        for (size_t output_index = 0; output_index < output_bit_count; output_index++) {
                itty_bit_string_set_bit (itty_bit_string_list_fetch (candidate_outputs, evaluation.selected_index),
                                         output_bits[output_index],
                                         false);
        }

        if (train_selected_node) {
                itty_feed_model_train_stats_t stats = { 0 };

                candidate_condensed = itty_feed_model_bit_string_clone_to_words (selected_condensed,
                                                                                 itty_bit_string_get_number_of_words (selected_condensed));
                if (!candidate_condensed)
                        goto done;
                summary->changed = itty_feed_model_apply_final_output_clear_set (model,
                                                                                 final_layer,
                                                                                 candidate_condensed,
                                                                                 output_bits,
                                                                                 output_bit_count);
                if (!summary->changed)
                        goto done;
                if (!itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][evaluation.selected_index],
                                                           final_layer_input,
                                                           candidate_condensed,
                                                           options,
                                                           &stats))
                        goto done;
                summary->total_flips = stats.flips;
                itty_bit_string_list_free (candidate_outputs);
                candidate_outputs = itty_feed_model_run_layer (model, final_layer, final_layer_input);
                if (!candidate_outputs)
                        goto done;
        } else {
                summary->changed = true;
        }

        itty_feed_model_measure_output_selection (candidate_outputs,
                                                  &summary->selected_node_after,
                                                  &summary->selection_margin_after);
        summary->compare_node_a_popcount_after =
                itty_bit_string_get_pop_count (itty_bit_string_list_fetch (candidate_outputs, summary->compare_node_a));
        summary->compare_node_b_popcount_after =
                itty_bit_string_get_pop_count (itty_bit_string_list_fetch (candidate_outputs, summary->compare_node_b));
        if (!itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                  candidate_outputs,
                                                                  target,
                                                                  summary->selected_node_after,
                                                                  &summary->selected_objective_after))
                goto done;

        ok = true;

done:
        if (candidate_condensed)
                itty_bit_string_free (candidate_condensed);
        if (selected_condensed)
                itty_bit_string_free (selected_condensed);
        if (candidate_outputs)
                itty_bit_string_list_free (candidate_outputs);
        if (final_outputs)
                itty_bit_string_list_free (final_outputs);
        if (final_layer_input != input)
                itty_bit_string_list_free (final_layer_input);
        itty_bit_string_free (evaluation.folded_activation);
        return ok;
}

bool
itty_feed_model_measure_selected_node_direct_clear_set (itty_feed_model_t                             *model,
                                                        itty_bit_string_list_t                        *input,
                                                        itty_bit_string_t                             *target,
                                                        size_t const                                  *decoded_bits,
                                                        size_t                                         decoded_bit_count,
                                                        size_t                                         compare_node_a,
                                                        size_t                                         compare_node_b,
                                                        itty_feed_model_selected_clear_set_summary_t *summary)
{
        return itty_feed_model_measure_or_train_selected_node_direct_clear_set (model,
                                                                                input,
                                                                                target,
                                                                                decoded_bits,
                                                                                decoded_bit_count,
                                                                                compare_node_a,
                                                                                compare_node_b,
                                                                                NULL,
                                                                                false,
                                                                                summary);
}

bool
itty_feed_model_measure_selected_node_direct_condensed_clear_set (itty_feed_model_t                             *model,
                                                                  itty_bit_string_list_t                        *input,
                                                                  itty_bit_string_t                             *target,
                                                                  size_t const                                  *decoded_bits,
                                                                  size_t                                         decoded_bit_count,
                                                                  size_t                                         compare_node_a,
                                                                  size_t                                         compare_node_b,
                                                                  itty_feed_model_selected_clear_set_summary_t *summary)
{
        size_t final_layer;
        itty_bit_string_list_t *final_layer_input = NULL;
        itty_bit_string_list_t *final_outputs = NULL;
        itty_bit_string_list_t *candidate_outputs = NULL;
        itty_bit_string_t *selected_condensed = NULL;
        itty_bit_string_t *candidate_condensed = NULL;
        itty_bit_string_t *candidate_activation = NULL;
        itty_feed_model_output_evaluation_t evaluation = { 0 };
        size_t output_bits[ITTY_FEED_MODEL_RESTORE_CLEAR_TRACE_LIMIT] = { 0 };
        size_t output_bit_count = 0;
        bool ok = false;

        if (summary)
                *summary = (itty_feed_model_selected_clear_set_summary_t) { 0 };
        if (!model ||
            !input ||
            !target ||
            !decoded_bits ||
            decoded_bit_count == 0 ||
            !summary ||
            model->number_of_layers == 0 ||
            compare_node_a >= model->nodes_per_layer ||
            compare_node_b >= model->nodes_per_layer)
                return false;

        final_layer = model->number_of_layers - 1;
        final_layer_input = itty_feed_model_run_to_layer_input (model, input, final_layer);
        final_outputs = itty_feed_model_run_layer (model, final_layer, final_layer_input);
        ok = final_outputs &&
             itty_feed_model_evaluate_output (model, final_outputs, target, &evaluation);
        if (!ok)
                goto done;

        summary->selected_node_before = evaluation.selected_index;
        summary->compare_node_a = compare_node_a;
        summary->compare_node_b = compare_node_b;
        summary->compare_node_a_popcount_before =
                itty_bit_string_get_pop_count (itty_bit_string_list_fetch (final_outputs, compare_node_a));
        summary->compare_node_b_popcount_before =
                itty_bit_string_get_pop_count (itty_bit_string_list_fetch (final_outputs, compare_node_b));

        selected_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                 model->masks_by_layer_node[final_layer][evaluation.selected_index]);
        if (!selected_condensed)
                goto done;
        if (!itty_feed_model_collect_selected_node_clear_set (model,
                                                              final_layer,
                                                              target,
                                                              selected_condensed,
                                                              decoded_bits,
                                                              decoded_bit_count,
                                                              output_bits,
                                                              &output_bit_count))
                goto done;

        summary->clear_vote_count = output_bit_count;
        candidate_condensed = itty_feed_model_bit_string_clone_to_words (selected_condensed,
                                                                         itty_bit_string_get_number_of_words (selected_condensed));
        if (!candidate_condensed)
                goto done;
        summary->changed = itty_feed_model_apply_final_output_clear_set (model,
                                                                         final_layer,
                                                                         candidate_condensed,
                                                                         output_bits,
                                                                         output_bit_count);
        if (!summary->changed)
                goto done;

        candidate_outputs = itty_feed_model_bit_string_list_clone (final_outputs);
        candidate_activation = itty_bit_string_clone (itty_bit_string_list_fetch (candidate_outputs,
                                                                                  evaluation.selected_index));
        if (!candidate_outputs || !candidate_activation)
                goto done;

        for (size_t output_bit_index = 0;
             output_bit_index < itty_bit_string_get_length (candidate_activation);
             output_bit_index++) {
                size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                            final_layer,
                                                                                            output_bit_index);
                bool condensed_bit = itty_bit_string_get_bit (candidate_condensed, condensed_bit_index);
                itty_bit_string_set_bit (candidate_activation, output_bit_index, condensed_bit);
        }
        itty_bit_string_free (candidate_outputs->bit_strings[evaluation.selected_index]);
        candidate_outputs->bit_strings[evaluation.selected_index] = candidate_activation;
        candidate_activation = NULL;

        itty_feed_model_measure_output_selection (candidate_outputs,
                                                  &summary->selected_node_after,
                                                  &summary->selection_margin_after);
        summary->compare_node_a_popcount_after =
                itty_bit_string_get_pop_count (itty_bit_string_list_fetch (candidate_outputs, compare_node_a));
        summary->compare_node_b_popcount_after =
                itty_bit_string_get_pop_count (itty_bit_string_list_fetch (candidate_outputs, compare_node_b));
        if (!itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                  candidate_outputs,
                                                                  target,
                                                                  summary->selected_node_after,
                                                                  &summary->selected_objective_after))
                goto done;

        ok = true;

done:
        if (candidate_activation)
                itty_bit_string_free (candidate_activation);
        if (candidate_condensed)
                itty_bit_string_free (candidate_condensed);
        if (selected_condensed)
                itty_bit_string_free (selected_condensed);
        if (candidate_outputs)
                itty_bit_string_list_free (candidate_outputs);
        if (final_outputs)
                itty_bit_string_list_free (final_outputs);
        if (final_layer_input != input)
                itty_bit_string_list_free (final_layer_input);
        itty_bit_string_free (evaluation.folded_activation);
        return ok;
}

bool
itty_feed_model_train_selected_node_direct_clear_set (itty_feed_model_t                             *model,
                                                      itty_bit_string_list_t                        *input,
                                                      itty_bit_string_t                             *target,
                                                      size_t const                                  *decoded_bits,
                                                      size_t                                         decoded_bit_count,
                                                      size_t                                         compare_node_a,
                                                      size_t                                         compare_node_b,
                                                      itty_feed_model_train_options_t const         *options,
                                                      itty_feed_model_selected_clear_set_summary_t *summary)
{
        return itty_feed_model_measure_or_train_selected_node_direct_clear_set (model,
                                                                                input,
                                                                                target,
                                                                                decoded_bits,
                                                                                decoded_bit_count,
                                                                                compare_node_a,
                                                                                compare_node_b,
                                                                                options,
                                                                                true,
                                                                                summary);
}

static bool
itty_feed_model_contender_clear_set_trace_is_better (itty_feed_model_contender_clear_set_trace_t const *candidate,
                                                     itty_feed_model_contender_clear_set_trace_t const *best)
{
        if (!best)
                return true;
        if (candidate->a_restored != best->a_restored)
                return candidate->a_restored;
        if (candidate->b_distance_after_restore != best->b_distance_after_restore)
                return candidate->b_distance_after_restore < best->b_distance_after_restore;
        if (candidate->b_false_negative_deficit_after_restore != best->b_false_negative_deficit_after_restore)
                return candidate->b_false_negative_deficit_after_restore < best->b_false_negative_deficit_after_restore;
        if (candidate->b_false_positive_excess_after_restore != best->b_false_positive_excess_after_restore)
                return candidate->b_false_positive_excess_after_restore < best->b_false_positive_excess_after_restore;
        if (candidate->total_flips != best->total_flips)
                return candidate->total_flips < best->total_flips;
        if (candidate->selection_margin_after_restore != best->selection_margin_after_restore)
                return candidate->selection_margin_after_restore > best->selection_margin_after_restore;
        if (candidate->clear_vote_count != best->clear_vote_count)
                return candidate->clear_vote_count < best->clear_vote_count;

        for (size_t index = 0; index < candidate->clear_vote_count; index++) {
                if (candidate->segment_indices[index] != best->segment_indices[index])
                        return candidate->segment_indices[index] < best->segment_indices[index];
        }

        return false;
}

typedef struct {
        size_t min_deficit;
        size_t deficit_one_bits;
        size_t deficit_two_bits;
        size_t cheapest_completion_cost;
        size_t top_k_completion_cost;
} itty_feed_model_completion_frontier_t;

static void
itty_feed_model_measure_completion_frontier (itty_feed_model_decoder_objective_t const *objective,
                                             size_t                                     top_k,
                                             itty_feed_model_completion_frontier_t      *frontier)
{
        *frontier = (itty_feed_model_completion_frontier_t) { 0 };

        if (objective->false_negative_count == 0)
                return;

        frontier->min_deficit = objective->false_negative_vote_deficit_min;
        frontier->deficit_one_bits = objective->false_negative_vote_deficit_histogram[1];
        frontier->deficit_two_bits = objective->false_negative_vote_deficit_histogram[2];
        frontier->cheapest_completion_cost = frontier->min_deficit;

        size_t remaining = top_k;
        for (size_t bucket = 1;
             bucket < ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS && remaining > 0;
             bucket++) {
                size_t bucket_count = objective->false_negative_vote_deficit_histogram[bucket];
                size_t used = bucket_count < remaining ? bucket_count : remaining;
                frontier->top_k_completion_cost += used * bucket;
                remaining -= used;
        }
}

static bool
itty_feed_model_completion_frontier_is_better (itty_feed_model_completion_frontier_t const *candidate,
                                               itty_feed_model_completion_frontier_t const *baseline)
{
        if (candidate->cheapest_completion_cost != baseline->cheapest_completion_cost)
                return candidate->cheapest_completion_cost < baseline->cheapest_completion_cost;
        if (candidate->deficit_one_bits != baseline->deficit_one_bits)
                return candidate->deficit_one_bits > baseline->deficit_one_bits;
        if (candidate->deficit_two_bits != baseline->deficit_two_bits)
                return candidate->deficit_two_bits > baseline->deficit_two_bits;
        if (candidate->top_k_completion_cost != baseline->top_k_completion_cost)
                return candidate->top_k_completion_cost < baseline->top_k_completion_cost;
        if (candidate->min_deficit != baseline->min_deficit)
                return candidate->min_deficit < baseline->min_deficit;
        return false;
}

static bool
itty_feed_model_try_best_transaction_completion_candidate (itty_feed_model_t                               *model,
                                                           itty_bit_string_list_t                          *first_input,
                                                           itty_bit_string_t                               *first_target,
                                                           itty_bit_string_list_t                          *second_input,
                                                           itty_bit_string_t                               *second_target,
                                                           itty_feed_model_train_options_t const          *options,
                                                           itty_feed_model_decoder_objective_t const      *a_before,
                                                           itty_feed_model_decoder_objective_t const      *b_before,
                                                           itty_feed_model_completion_frontier_t const    *b_frontier_before,
                                                           itty_feed_model_transaction_scaffold_round_t   *round_trace,
                                                           itty_feed_model_transaction_scaffold_summary_t *summary,
                                                           itty_feed_model_finish_candidate_trace_t       *best_candidate_trace)
{
        size_t final_layer = model->number_of_layers - 1;
        size_t frontier_top_k = 4;
        itty_feed_model_replay_example_t first_example = {
                .input = first_input,
                .target = first_target,
        };
        itty_bit_string_list_t *final_layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                        second_input,
                                                                                        final_layer);
        itty_feed_model_final_repair_list_t repairs = { 0 };
        itty_bit_string_t *a_before_folded = NULL;
        itty_feed_model_segment_node_selection_summary_t a_selection_before = { 0 };
        if (!final_layer_input ||
            !itty_feed_model_evaluate_replay_example (model,
                                                      &first_example,
                                                      &(itty_feed_model_decoder_objective_t) { 0 },
                                                      &a_before_folded) ||
            !itty_feed_model_measure_segment_node_selection (model,
                                                             first_input,
                                                             first_target,
                                                             &a_selection_before) ||
            !itty_feed_model_collect_final_repairs (model,
                                                    final_layer_input,
                                                    second_target,
                                                    &repairs)) {
                if (a_before_folded)
                        itty_bit_string_free (a_before_folded);
                if (final_layer_input != second_input)
                        itty_bit_string_list_free (final_layer_input);
                return false;
        }

        bool found = false;
        itty_feed_model_layer_state_snapshot_t *best_snapshot = NULL;
        itty_feed_model_transaction_scaffold_round_t best_trace = { 0 };
        itty_feed_model_segment_node_selection_summary_t best_selection = { 0 };

        for (size_t repair_index = 0; repair_index < repairs.count; repair_index++) {
                itty_feed_model_final_repair_t const *repair = &repairs.items[repair_index];
                if (repair->quota_size != 1)
                        continue;

                itty_feed_model_layer_state_snapshot_t *candidate_snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                                   final_layer);
                itty_bit_string_t *current_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                           model->masks_by_layer_node[final_layer][repair->final_node]);
                itty_bit_string_t *oracle_target = current_condensed ?
                                                   itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                                              itty_bit_string_get_number_of_words (current_condensed)) :
                                                   NULL;
                itty_feed_model_train_stats_t b_step = { 0 };
                itty_feed_model_contender_restore_summary_t contender = { 0 };
                itty_feed_model_mask_flip_trace_t finish_mask_traces[ITTY_FEED_MODEL_FINISH_CLEAR_SET_TRACE_LIMIT] = { 0 };
                size_t finish_mask_flip_count = 0;
                itty_feed_model_finish_candidate_trace_t candidate_trace = {
                        .decoded_bit = repair->decoded_bit,
                        .final_node = repair->final_node,
                        .final_output_bit = repair->output_bit,
                        .finish_margin_required = options->finish_margin,
                        .finish_condensed_bit = repair->condensed_bit,
                        .b_distance_before = b_before->selected_distance,
                        .b_deficit_before = b_before->false_negative_vote_deficit,
                        .b_target_one_margin_before = b_before->target_one_margin,
                        .a_selected_node_before = a_selection_before.selected_by_popcount,
                };
                bool accepted = false;
                bool used_restore = false;
                size_t restore_flips = 0;
                bool strict = false;
                bool dist = false;
                bool progress = false;
                bool no_reg = false;
                bool frontier_improved = false;
                itty_feed_model_decoder_objective_t a_after = { 0 };
                itty_feed_model_decoder_objective_t b_after = { 0 };
                itty_feed_model_completion_frontier_t frontier_after = { 0 };
                itty_feed_model_segment_node_selection_summary_t selection = { 0 };
                itty_feed_model_segment_node_selection_summary_t a_selection_after_finish = { 0 };
                itty_bit_string_t *a_after_finish_folded = NULL;

                if (summary)
                        summary->finish_candidates++;
                if (oracle_target) {
                        itty_bit_string_set_bit (oracle_target,
                                                 repair->condensed_bit,
                                                 repair->value);
                        oracle_target->pop_count_computed = false;
                }

                if (oracle_target &&
                    itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][repair->final_node],
                                                          final_layer_input,
                                                          oracle_target,
                                                          options,
                                                          &b_step) &&
                    itty_feed_model_measure_decoder_objective (model, first_input, first_target, &a_after) &&
                    itty_feed_model_measure_decoder_objective (model, second_input, second_target, &b_after) &&
                    itty_feed_model_evaluate_replay_example (model,
                                                             &first_example,
                                                             &(itty_feed_model_decoder_objective_t) { 0 },
                                                             &a_after_finish_folded) &&
                    itty_feed_model_measure_segment_node_selection (model,
                                                                    first_input,
                                                                    first_target,
                                                                    &a_selection_after_finish)) {
                        itty_feed_model_replay_transition_matrix_t a_finish_transitions = { 0 };
                        itty_feed_model_accumulate_replay_transition (&a_finish_transitions,
                                                                     first_target,
                                                                     a_before_folded,
                                                                     a_after_finish_folded);
                        candidate_trace.b_distance_after_finish = b_after.selected_distance;
                        candidate_trace.b_deficit_after_finish = b_after.false_negative_vote_deficit;
                        candidate_trace.b_target_one_margin_after_finish = b_after.target_one_margin;
                        candidate_trace.a_distance_after_finish = a_after.selected_distance;
                        candidate_trace.a_excess_after_finish = a_after.false_positive_vote_excess;
                        candidate_trace.a_deficit_after_finish = a_after.false_negative_vote_deficit;
                        candidate_trace.a_selected_node_after_finish = a_selection_after_finish.selected_by_popcount;
                        candidate_trace.a_damaged_correct_zero_to_false_positive_bits =
                                a_finish_transitions.correct_zero_to_false_positive_bits;
                        candidate_trace.a_damaged_correct_one_to_false_negative_bits =
                                a_finish_transitions.correct_one_to_false_negative_bits;
                        candidate_trace.family_one_finish =
                                b_after.selected_distance < b_before->selected_distance &&
                                a_finish_transitions.correct_zero_to_false_positive_bits > 0 &&
                                a_finish_transitions.correct_one_to_false_negative_bits == 0 &&
                                a_selection_after_finish.selected_by_popcount == a_selection_before.selected_by_popcount;
                        candidate_trace.finish_margin_met_before_restore =
                                b_after.selected_distance < b_before->selected_distance &&
                                b_after.target_one_margin >= b_before->target_one_margin + options->finish_margin;
                        finish_mask_flip_count =
                                itty_feed_model_collect_layer_mask_flip_traces (model,
                                                                                final_layer,
                                                                                candidate_snapshot,
                                                                                finish_mask_traces,
                                                                                ITTY_FEED_MODEL_FINISH_CLEAR_SET_TRACE_LIMIT);
                        candidate_trace.finish_mask_flip_count = finish_mask_flip_count;
                        if (b_after.selected_distance < b_before->selected_distance && summary)
                                summary->finish_pre_restore_distance_helpful++;
                        if (b_after.false_negative_vote_deficit < b_before->false_negative_vote_deficit && summary)
                                summary->finish_pre_restore_deficit_helpful++;
                        if (candidate_trace.finish_margin_met_before_restore && summary)
                                summary->finish_pre_restore_margin_met++;
                        if (b_after.selected_distance < b_before->selected_distance && summary)
                                summary->finish_complete_b_before_restore++;
                        if (a_finish_transitions.correct_zero_to_false_positive_bits > 0 && summary)
                                summary->finish_create_a_correct_zero_to_false_positive++;
                        if (a_finish_transitions.correct_one_to_false_negative_bits > 0 && summary)
                                summary->finish_create_a_correct_one_to_false_negative++;
                        if (a_selection_after_finish.selected_by_popcount != a_selection_before.selected_by_popcount &&
                            summary)
                                summary->finish_switch_a_selected_node++;

                        if (a_after.selected_distance <= a_before->selected_distance &&
                            a_after.false_positive_vote_excess <= a_before->false_positive_vote_excess) {
                                used_restore = false;
                                strict = true;
                                dist = true;
                                progress = true;
                                no_reg = true;
                                itty_feed_model_measure_completion_frontier (&b_after,
                                                                             frontier_top_k,
                                                                             &frontier_after);
                                frontier_improved = itty_feed_model_completion_frontier_is_better (&frontier_after,
                                                                                                  b_frontier_before);
                                accepted = candidate_trace.finish_margin_met_before_restore &&
                                           b_after.selected_distance <= b_before->selected_distance &&
                                           (b_after.selected_distance < b_before->selected_distance ||
                                            b_after.false_negative_vote_deficit < b_before->false_negative_vote_deficit ||
                                            frontier_improved);
                        } else {
                                used_restore = true;
                                candidate_trace.restore_needed = true;
                                if (summary)
                                        summary->finish_clobbers_a++;
                                if (candidate_trace.family_one_finish) {
                                        itty_feed_model_restore_failure_summary_t restore_audit = { 0 };
                                        if (itty_feed_model_measure_final_layer_restore_failure (model,
                                                                                                 first_input,
                                                                                                 first_target,
                                                                                                 second_input,
                                                                                                 second_target,
                                                                                                 options,
                                                                                                 0,
                                                                                                 false,
                                                                                                 &restore_audit)) {
                                                candidate_trace.restore_a_candidate_repair_count =
                                                        restore_audit.a_candidate_repair_count;
                                                candidate_trace.restore_a_useful_repair_count =
                                                        restore_audit.a_useful_repair_count;
                                                candidate_trace.restore_a_target_zero_repair_count =
                                                        restore_audit.a_target_zero_repair_count;
                                                candidate_trace.restore_a_rejected_repair_count =
                                                        restore_audit.a_rejected_repair_count;
                                                candidate_trace.restore_a_no_flip_reason =
                                                        restore_audit.no_flip_reason;
                                                if (restore_audit.trace_count > 0) {
                                                        itty_feed_model_restore_failure_trace_t const *audit =
                                                                &restore_audit.traces[0];
                                                        candidate_trace.restore_audit_decoded_bit = audit->decoded_bit;
                                                        candidate_trace.restore_audit_ones = audit->current_ones;
                                                        candidate_trace.restore_audit_threshold = audit->threshold;
                                                        candidate_trace.restore_audit_max_ones_for_zero = audit->max_ones_for_zero;
                                                        candidate_trace.restore_audit_excess = audit->excess;
                                                        candidate_trace.restore_audit_clearable_votes = audit->clearable_segment_votes;
                                                        candidate_trace.restore_audit_mask_flips = audit->candidate_mask_flips;
                                                }
                                        }
                                }
                                accepted = itty_feed_model_apply_contender_restore_current_state (model,
                                                                                                  first_input,
                                                                                                  first_target,
                                                                                                  second_input,
                                                                                                  second_target,
                                                                                                  options,
                                                                                                  false,
                                                                                                  &contender);
                                candidate_trace.restore_contender_node = contender.contender_node;
                                candidate_trace.restore_available = accepted;
                                if (summary && accepted)
                                        summary->finish_contender_restore_available++;
                                if (accepted &&
                                    itty_feed_model_measure_decoder_objective (model, first_input, first_target, &a_after) &&
                                    itty_feed_model_measure_decoder_objective (model, second_input, second_target, &b_after)) {
                                        candidate_trace.a_distance_after_restore = a_after.selected_distance;
                                        candidate_trace.a_excess_after_restore = a_after.false_positive_vote_excess;
                                        candidate_trace.b_distance_after_restore = b_after.selected_distance;
                                        candidate_trace.b_deficit_after_restore = b_after.false_negative_vote_deficit;
                                        candidate_trace.b_target_one_margin_after_restore = b_after.target_one_margin;
                                        restore_flips = contender.total_flips;
                                        strict = contender.b_strict_preserved;
                                        dist = contender.b_distance_preserved;
                                        progress = contender.b_progress_preserved;
                                        no_reg = contender.b_no_regression;
                                        candidate_trace.finish_margin_met_after_restore =
                                                b_after.selected_distance < b_before->selected_distance &&
                                                b_after.target_one_margin >= b_before->target_one_margin + options->finish_margin;
                                        candidate_trace.strict_preserved = strict;
                                        candidate_trace.distance_preserved = dist;
                                        candidate_trace.progress_preserved = progress;
                                        candidate_trace.no_regression = no_reg;
                                        itty_feed_model_measure_completion_frontier (&b_after,
                                                                                     frontier_top_k,
                                                                                     &frontier_after);
                                        frontier_improved = itty_feed_model_completion_frontier_is_better (&frontier_after,
                                                                                                          b_frontier_before);
                                        candidate_trace.frontier_improved = frontier_improved;
                                        accepted = contender.contender_accepted &&
                                                   contender.b_no_regression &&
                                                   (b_after.selected_distance < b_before->selected_distance ||
                                                    b_after.false_negative_vote_deficit < b_before->false_negative_vote_deficit ||
                                                    frontier_improved);
                                        if (summary && a_after.selected_distance <= a_before->selected_distance &&
                                            a_after.false_positive_vote_excess <= a_before->false_positive_vote_excess)
                                                summary->finish_restores_a++;
                                        if (summary && dist)
                                                summary->finish_post_restore_distance_preserved++;
                                        if (summary && progress)
                                                summary->finish_post_restore_progress_preserved++;
                                        if (summary && candidate_trace.finish_margin_met_after_restore)
                                                summary->finish_post_restore_margin_preserved++;
                                        if (summary &&
                                            b_after.selected_distance >= b_before->selected_distance &&
                                            b_after.false_negative_vote_deficit >= b_before->false_negative_vote_deficit &&
                                            (candidate_trace.b_distance_after_finish < b_before->selected_distance ||
                                             candidate_trace.b_deficit_after_finish < b_before->false_negative_vote_deficit))
                                                summary->finish_restore_erases_b++;
                                } else if (candidate_trace.family_one_finish) {
                                        size_t damage_set_flips = 0;
                                        bool damage_set_distance_preserved = false;
                                        bool damage_set_progress_preserved = false;
                                        itty_feed_model_decoder_objective_t a_after_damage_set = { 0 };
                                        itty_feed_model_decoder_objective_t b_after_damage_set = { 0 };
                                        candidate_trace.damage_set_restore_available =
                                                itty_feed_model_apply_finish_damage_set_restore_current_state (model,
                                                                                                               first_input,
                                                                                                               first_target,
                                                                                                               second_input,
                                                                                                               second_target,
                                                                                                               options,
                                                                                                               a_before,
                                                                                                               b_before,
                                                                                                               &candidate_trace,
                                                                                                               finish_mask_traces,
                                                                                                               finish_mask_flip_count,
                                                                                                               &candidate_trace.family_one_damaged_bits,
                                                                                                               &damage_set_flips,
                                                                                                               &a_after_damage_set,
                                                                                                               &b_after_damage_set,
                                                                                                               &damage_set_distance_preserved,
                                                                                                               &damage_set_progress_preserved);
                                        candidate_trace.damage_set_restore_flips = damage_set_flips;
                                        candidate_trace.damage_set_restore_distance_preserved = damage_set_distance_preserved;
                                        candidate_trace.damage_set_restore_progress_preserved = damage_set_progress_preserved;
                                        if (candidate_trace.damage_set_restore_available) {
                                                candidate_trace.damage_set_restore_restored_a =
                                                        a_after_damage_set.selected_distance <= a_before->selected_distance &&
                                                        a_after_damage_set.false_positive_vote_excess <= a_before->false_positive_vote_excess;
                                                candidate_trace.a_distance_after_restore = a_after_damage_set.selected_distance;
                                                candidate_trace.a_excess_after_restore = a_after_damage_set.false_positive_vote_excess;
                                                candidate_trace.b_distance_after_restore = b_after_damage_set.selected_distance;
                                                candidate_trace.b_deficit_after_restore = b_after_damage_set.false_negative_vote_deficit;
                                                candidate_trace.b_target_one_margin_after_restore = b_after_damage_set.target_one_margin;
                                                strict = b_after_damage_set.selected_distance <= b_before->selected_distance &&
                                                         b_after_damage_set.false_negative_vote_deficit <= b_before->false_negative_vote_deficit;
                                                dist = damage_set_distance_preserved;
                                                progress = damage_set_progress_preserved;
                                                no_reg = b_after_damage_set.selected_distance <= b_before->selected_distance &&
                                                         b_after_damage_set.false_negative_vote_deficit <= b_before->false_negative_vote_deficit;
                                                candidate_trace.finish_margin_met_after_restore =
                                                        b_after_damage_set.selected_distance < b_before->selected_distance &&
                                                        b_after_damage_set.target_one_margin >= b_before->target_one_margin + options->finish_margin;
                                                candidate_trace.strict_preserved = strict;
                                                candidate_trace.distance_preserved = dist;
                                                candidate_trace.progress_preserved = progress;
                                                candidate_trace.no_regression = no_reg;
                                                itty_feed_model_measure_completion_frontier (&b_after_damage_set,
                                                                                             frontier_top_k,
                                                                                             &frontier_after);
                                                frontier_improved = itty_feed_model_completion_frontier_is_better (&frontier_after,
                                                                                                                  b_frontier_before);
                                                candidate_trace.frontier_improved = frontier_improved;
                                                if (summary && candidate_trace.finish_margin_met_after_restore)
                                                        summary->finish_post_restore_margin_preserved++;
                                                accepted = candidate_trace.damage_set_restore_restored_a &&
                                                           candidate_trace.finish_margin_met_after_restore &&
                                                           (dist || progress || frontier_improved);
                                                restore_flips = damage_set_flips;
                                                a_after = a_after_damage_set;
                                                b_after = b_after_damage_set;
                                        }
                                }
                        }
                }

                if (!accepted) {
                        bool no_pre_restore_gain =
                                candidate_trace.b_distance_after_finish >= b_before->selected_distance &&
                                candidate_trace.b_deficit_after_finish >= b_before->false_negative_vote_deficit;
                        bool a_not_restored =
                                candidate_trace.restore_needed &&
                                !(candidate_trace.a_distance_after_restore <= a_before->selected_distance &&
                                  candidate_trace.a_excess_after_restore <= a_before->false_positive_vote_excess);
                        bool b_gain_lost =
                                candidate_trace.restore_needed &&
                                candidate_trace.b_distance_after_finish < b_before->selected_distance &&
                                candidate_trace.b_distance_after_restore >= b_before->selected_distance;
                        candidate_trace.rejected_no_pre_restore_gain = no_pre_restore_gain;
                        candidate_trace.rejected_a_not_restored = a_not_restored;
                        candidate_trace.rejected_b_gain_lost = b_gain_lost;
                        candidate_trace.restore_failed = candidate_trace.restore_needed && !candidate_trace.restore_available;
                        if (summary && no_pre_restore_gain)
                                summary->finish_rejected_no_pre_restore_gain++;
                        if (summary && a_not_restored)
                                summary->finish_rejected_a_not_restored++;
                        if (summary && b_gain_lost)
                                summary->finish_rejected_b_gain_lost++;
                        if (summary && candidate_trace.restore_failed)
                                summary->finish_restore_fails++;
                }

                if (summary &&
                    summary->finish_trace_count < ITTY_FEED_MODEL_FINISH_CANDIDATE_TRACE_LIMIT)
                        summary->finish_traces[summary->finish_trace_count++] = candidate_trace;

                if (accepted &&
                    itty_feed_model_measure_segment_node_selection (model,
                                                                    second_input,
                                                                    second_target,
                                                                    &selection)) {
                        itty_feed_model_transaction_scaffold_round_t candidate_round = {
                                .accepted = true,
                                .used_restore = used_restore,
                                .used_finish_nearest_bit = true,
                                .strict_preserved = strict,
                                .distance_preserved = dist,
                                .progress_preserved = progress,
                                .no_regression = no_reg,
                                .frontier_improved = frontier_improved,
                                .a_distance_after = a_after.selected_distance,
                                .a_false_positive_excess_after = a_after.false_positive_vote_excess,
                                .b_distance_after = b_after.selected_distance,
                                .b_false_negative_deficit_after = b_after.false_negative_vote_deficit,
                                .b_min_deficit_after = frontier_after.min_deficit,
                                .b_deficit_one_bits_after = frontier_after.deficit_one_bits,
                                .b_deficit_two_bits_after = frontier_after.deficit_two_bits,
                                .b_cheapest_completion_cost_after = frontier_after.cheapest_completion_cost,
                                .b_top_k_completion_cost_after = frontier_after.top_k_completion_cost,
                                .b_flips = b_step.flips,
                                .restore_flips = restore_flips,
                        };

                        bool better = !found;
                        if (!better && candidate_round.b_distance_after != best_trace.b_distance_after)
                                better = candidate_round.b_distance_after < best_trace.b_distance_after;
                        if (!better && candidate_round.b_false_negative_deficit_after != best_trace.b_false_negative_deficit_after)
                                better = candidate_round.b_false_negative_deficit_after < best_trace.b_false_negative_deficit_after;
                        if (!better && candidate_round.frontier_improved != best_trace.frontier_improved)
                                better = candidate_round.frontier_improved;
                        if (!better && candidate_round.b_flips + candidate_round.restore_flips !=
                                       best_trace.b_flips + best_trace.restore_flips)
                                better = candidate_round.b_flips + candidate_round.restore_flips <
                                         best_trace.b_flips + best_trace.restore_flips;
                        if (!better && selection.popcount_gap != best_selection.popcount_gap)
                                better = selection.popcount_gap > best_selection.popcount_gap;

                        if (better) {
                                if (best_snapshot)
                                        itty_feed_model_free_layer_state_snapshot (model, best_snapshot);
                                best_snapshot = itty_feed_model_snapshot_layer_state (model, final_layer);
                                best_trace = candidate_round;
                                if (best_candidate_trace) {
                                        candidate_trace.chosen_best = true;
                                        *best_candidate_trace = candidate_trace;
                                }
                                best_selection = selection;
                                found = true;
                        }
                }

                if (a_after_finish_folded)
                        itty_bit_string_free (a_after_finish_folded);
                if (oracle_target)
                        itty_bit_string_free (oracle_target);
                if (current_condensed)
                        itty_bit_string_free (current_condensed);
                itty_feed_model_restore_layer_state_snapshot (model, final_layer, candidate_snapshot);
        }

        itty_feed_model_final_repair_list_clear (&repairs);
        if (a_before_folded)
                itty_bit_string_free (a_before_folded);
        if (final_layer_input != second_input)
                itty_bit_string_list_free (final_layer_input);

        if (!found) {
                if (best_snapshot)
                        itty_feed_model_free_layer_state_snapshot (model, best_snapshot);
                return false;
        }

        itty_feed_model_restore_layer_state_snapshot (model, final_layer, best_snapshot);
        round_trace->used_restore = best_trace.used_restore;
        round_trace->used_finish_nearest_bit = true;
        round_trace->strict_preserved = best_trace.strict_preserved;
        round_trace->distance_preserved = best_trace.distance_preserved;
        round_trace->progress_preserved = best_trace.progress_preserved;
        round_trace->no_regression = best_trace.no_regression;
        round_trace->frontier_improved = best_trace.frontier_improved;
        round_trace->a_distance_after = best_trace.a_distance_after;
        round_trace->a_false_positive_excess_after = best_trace.a_false_positive_excess_after;
        round_trace->b_distance_after = best_trace.b_distance_after;
        round_trace->b_false_negative_deficit_after = best_trace.b_false_negative_deficit_after;
        round_trace->b_min_deficit_after = best_trace.b_min_deficit_after;
        round_trace->b_deficit_one_bits_after = best_trace.b_deficit_one_bits_after;
        round_trace->b_deficit_two_bits_after = best_trace.b_deficit_two_bits_after;
        round_trace->b_cheapest_completion_cost_after = best_trace.b_cheapest_completion_cost_after;
        round_trace->b_top_k_completion_cost_after = best_trace.b_top_k_completion_cost_after;
        round_trace->b_flips = best_trace.b_flips;
        round_trace->restore_flips = best_trace.restore_flips;
        round_trace->accepted = true;
        return true;
}

static bool
itty_feed_model_try_post_restore_replacement_current_state (itty_feed_model_t                      *model,
                                                            itty_bit_string_list_t                 *first_input,
                                                            itty_bit_string_t                      *first_target,
                                                            itty_bit_string_list_t                 *second_input,
                                                            itty_bit_string_t                      *second_target,
                                                            itty_feed_model_train_options_t const *options,
                                                            itty_feed_model_decoder_objective_t const *a_before,
                                                            itty_feed_model_decoder_objective_t const *b_before,
                                                            itty_feed_model_finish_candidate_trace_t *trace,
                                                            size_t                                *flip_count,
                                                            itty_feed_model_decoder_objective_t   *a_after,
                                                            itty_feed_model_decoder_objective_t   *b_after)
{
        size_t final_layer = model->number_of_layers - 1;
        itty_bit_string_list_t *final_layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                        second_input,
                                                                                        final_layer);
        itty_feed_model_final_repair_list_t repairs = { 0 };
        bool collected = final_layer_input &&
                         itty_feed_model_collect_final_repairs (model,
                                                                final_layer_input,
                                                                second_target,
                                                                &repairs);
        if (!collected) {
                if (final_layer_input != second_input)
                        itty_bit_string_list_free (final_layer_input);
                return false;
        }

        bool found = false;
        size_t best_flips = 0;
        itty_feed_model_layer_state_snapshot_t *best_snapshot = NULL;
        itty_feed_model_decoder_objective_t best_a = { 0 };
        itty_feed_model_decoder_objective_t best_b = { 0 };
        size_t best_output_bit = 0;
        size_t best_condensed_bit = 0;

        for (size_t repair_index = 0; repair_index < repairs.count; repair_index++) {
                itty_feed_model_final_repair_t const *repair = &repairs.items[repair_index];
                if (repair->quota_size != 1)
                        continue;

                if (trace)
                        trace->replacement_candidate_count++;

                itty_feed_model_layer_state_snapshot_t *candidate_snapshot =
                        itty_feed_model_snapshot_layer_state (model, final_layer);
                itty_bit_string_t *current_condensed =
                        itty_feed_model_run_node_condensed (final_layer_input,
                                                           model->masks_by_layer_node[final_layer][repair->final_node]);
                itty_bit_string_t *oracle_target = current_condensed ?
                                                   itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                                              itty_bit_string_get_number_of_words (current_condensed)) :
                                                   NULL;
                itty_feed_model_train_stats_t stats = { 0 };
                itty_feed_model_decoder_objective_t a_candidate = { 0 };
                itty_feed_model_decoder_objective_t b_candidate = { 0 };

                if (oracle_target)
                        itty_bit_string_set_bit (oracle_target,
                                                 repair->condensed_bit,
                                                 repair->value);

                bool ok = oracle_target &&
                          itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][repair->final_node],
                                                                final_layer_input,
                                                                oracle_target,
                                                                options,
                                                                &stats) &&
                          itty_feed_model_measure_decoder_objective (model,
                                                                     first_input,
                                                                     first_target,
                                                                     &a_candidate) &&
                          itty_feed_model_measure_decoder_objective (model,
                                                                     second_input,
                                                                     second_target,
                                                                     &b_candidate);

                if (ok) {
                        bool a_safe =
                                a_candidate.selected_distance <= a_before->selected_distance &&
                                a_candidate.false_positive_vote_excess <= a_before->false_positive_vote_excess;
                        bool distance_helpful =
                                b_candidate.selected_distance < b_before->selected_distance;
                        bool progress_helpful =
                                b_candidate.selected_distance <= b_before->selected_distance &&
                                b_candidate.false_negative_vote_deficit < b_before->false_negative_vote_deficit;

                        if (trace) {
                                if (a_safe)
                                        trace->replacement_a_safe_count++;
                                if (distance_helpful)
                                        trace->replacement_distance_helpful_count++;
                                if (progress_helpful)
                                        trace->replacement_progress_helpful_count++;
                        }

                        if (a_safe && (distance_helpful || progress_helpful)) {
                                bool better = !found;
                                if (!better && b_candidate.selected_distance != best_b.selected_distance)
                                        better = b_candidate.selected_distance < best_b.selected_distance;
                                if (!better &&
                                    b_candidate.false_negative_vote_deficit != best_b.false_negative_vote_deficit)
                                        better = b_candidate.false_negative_vote_deficit < best_b.false_negative_vote_deficit;
                                if (!better && stats.flips != best_flips)
                                        better = stats.flips < best_flips;

                                if (better) {
                                        if (best_snapshot)
                                                itty_feed_model_free_layer_state_snapshot (model, best_snapshot);
                                        best_snapshot = itty_feed_model_snapshot_layer_state (model, final_layer);
                                        best_a = a_candidate;
                                        best_b = b_candidate;
                                        best_flips = stats.flips;
                                        best_output_bit = repair->output_bit;
                                        best_condensed_bit = repair->condensed_bit;
                                        found = true;
                                }
                        }
                }

                if (oracle_target)
                        itty_bit_string_free (oracle_target);
                if (current_condensed)
                        itty_bit_string_free (current_condensed);
                itty_feed_model_restore_layer_state_snapshot (model, final_layer, candidate_snapshot);
        }

        itty_feed_model_final_repair_list_clear (&repairs);
        if (final_layer_input != second_input)
                itty_bit_string_list_free (final_layer_input);

        if (!found) {
                if (best_snapshot)
                        itty_feed_model_free_layer_state_snapshot (model, best_snapshot);
                return false;
        }

        itty_feed_model_restore_layer_state_snapshot (model, final_layer, best_snapshot);
        if (best_snapshot)
                best_snapshot = NULL;

        if (flip_count)
                *flip_count = best_flips;
        if (a_after)
                *a_after = best_a;
        if (b_after)
                *b_after = best_b;
        if (trace) {
                trace->replacement_available = true;
                trace->replacement_flips = best_flips;
                trace->replacement_output_bit = best_output_bit;
                trace->replacement_condensed_bit = best_condensed_bit;
                trace->replacement_distance_preserved = best_b.selected_distance < b_before->selected_distance;
                trace->replacement_progress_preserved =
                        best_b.selected_distance <= b_before->selected_distance &&
                        best_b.false_negative_vote_deficit < b_before->false_negative_vote_deficit;
        }
        return true;
}

static bool
itty_feed_model_apply_finish_damage_set_restore_current_state (itty_feed_model_t                      *model,
                                                               itty_bit_string_list_t                 *first_input,
                                                               itty_bit_string_t                      *first_target,
                                                               itty_bit_string_list_t                 *second_input,
                                                               itty_bit_string_t                      *second_target,
                                                               itty_feed_model_train_options_t const *options,
                                                               itty_feed_model_decoder_objective_t const *a_before,
                                                               itty_feed_model_decoder_objective_t const *b_before,
                                                               itty_feed_model_finish_candidate_trace_t *trace,
                                                               itty_feed_model_mask_flip_trace_t const *finish_mask_flips,
                                                               size_t                                   finish_mask_flip_count,
                                                               size_t                                *damaged_bit_count,
                                                               size_t                                *flip_count,
                                                               itty_feed_model_decoder_objective_t   *a_after,
                                                               itty_feed_model_decoder_objective_t   *b_after,
                                                               bool                                  *distance_preserved,
                                                               bool                                  *progress_preserved)
{
        size_t final_layer = model->number_of_layers - 1;
        itty_feed_model_layer_state_snapshot_t *base_snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                       final_layer);
        itty_bit_string_list_t *final_layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                        first_input,
                                                                                        final_layer);
        itty_bit_string_list_t *final_outputs = itty_feed_model_run_layer (model,
                                                                           final_layer,
                                                                           final_layer_input);
        itty_feed_model_output_evaluation_t evaluation = { 0 };
        bool ok = final_outputs &&
                  itty_feed_model_evaluate_output (model,
                                                  final_outputs,
                                                  first_target,
                                                  &evaluation);
        if (!ok) {
                if (final_outputs)
                        itty_bit_string_list_free (final_outputs);
                if (final_layer_input != first_input)
                        itty_bit_string_list_free (final_layer_input);
                if (base_snapshot)
                        itty_feed_model_free_layer_state_snapshot (model, base_snapshot);
                return false;
        }

        itty_bit_string_t *selected_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                    model->masks_by_layer_node[final_layer][evaluation.selected_index]);
        size_t target_bit_capacity = itty_bit_string_get_number_of_words (first_target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t local_damaged_bits = 0;
        size_t damaged_bit_index = (size_t) -1;

        for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                bool target_bit = itty_bit_string_get_bit (first_target, bit_index);
                bool decoded_bit = itty_bit_string_get_bit (evaluation.folded_activation, bit_index);
                if (!target_bit && decoded_bit) {
                        local_damaged_bits++;
                        if (damaged_bit_index == (size_t) -1)
                                damaged_bit_index = bit_index;
                }
        }

        bool restored_a = false;
        size_t local_flip_count = 0;
        bool local_distance_preserved = false;
        bool local_progress_preserved = false;
        bool have_best_trace = false;
        itty_feed_model_contender_clear_set_trace_t best_trace = { 0 };
        itty_feed_model_layer_state_snapshot_t *best_snapshot = NULL;

        if (selected_condensed &&
            local_damaged_bits == 1 &&
            damaged_bit_index != (size_t) -1) {
                size_t votes_needed = 0;
                size_t condensed_bit_capacity = itty_bit_string_get_length (selected_condensed);
                size_t output_bit_capacity = condensed_bit_capacity * 2;
                size_t clear_output_bits[ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_VOTE_LIMIT] = { 0 };
                size_t clear_segments[ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_VOTE_LIMIT] = { 0 };
                size_t clear_count = 0;

                if (itty_feed_model_get_segment_condense_vote_need (model,
                                                                    damaged_bit_index,
                                                                    false,
                                                                    selected_condensed,
                                                                    &votes_needed)) {
                        for (size_t output_bit_index = damaged_bit_index;
                             output_bit_index < output_bit_capacity;
                             output_bit_index += target_bit_capacity) {
                                size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                                            final_layer,
                                                                                                            output_bit_index);
                                if (!itty_bit_string_get_bit (selected_condensed, condensed_bit_index))
                                        continue;
                                if (clear_count < ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_VOTE_LIMIT) {
                                        clear_output_bits[clear_count] = output_bit_index;
                                        clear_segments[clear_count] = output_bit_index / target_bit_capacity;
                                        clear_count++;
                                }
                        }
                }

                if (trace)
                        trace->family_one_damaged_bits = local_damaged_bits;

                if (votes_needed > 0 && clear_count >= votes_needed) {
                        size_t indices[ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_VOTE_LIMIT] = { 0 };
                        for (size_t index = 0; index < votes_needed; index++)
                                indices[index] = index;

                        bool done = false;
                        while (!done) {
                                itty_feed_model_layer_state_snapshot_t *candidate_snapshot =
                                        itty_feed_model_snapshot_layer_state (model, final_layer);
                                itty_bit_string_t *candidate_condensed =
                                        itty_feed_model_bit_string_clone_to_words (selected_condensed,
                                                                                   itty_bit_string_get_number_of_words (selected_condensed));
                                itty_feed_model_contender_clear_set_trace_t clear_trace = {
                                        .clear_vote_count = votes_needed,
                                        .selected_node_before_restore = evaluation.selected_index,
                                };

                                for (size_t vote_index = 0; vote_index < votes_needed; vote_index++) {
                                        clear_trace.segment_indices[vote_index] = clear_segments[indices[vote_index]];
                                        clear_trace.final_output_bits[vote_index] = clear_output_bits[indices[vote_index]];
                                }

                                if (candidate_condensed &&
                                    itty_feed_model_apply_final_output_clear_set (model,
                                                                                  final_layer,
                                                                                  candidate_condensed,
                                                                                  clear_trace.final_output_bits,
                                                                                  clear_trace.clear_vote_count)) {
                                        itty_feed_model_train_stats_t stats = { 0 };
                                        itty_feed_model_segment_node_selection_summary_t a_selection = { 0 };
                                        itty_feed_model_decoder_objective_t a_candidate_after = { 0 };
                                        itty_feed_model_decoder_objective_t b_candidate_after = { 0 };
                                        itty_feed_model_mask_flip_trace_t restore_mask_traces[ITTY_FEED_MODEL_FINISH_CLEAR_SET_TRACE_LIMIT] = { 0 };

                                        itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][evaluation.selected_index],
                                                                              final_layer_input,
                                                                              candidate_condensed,
                                                                              options,
                                                                              &stats);

                                        if (itty_feed_model_measure_decoder_objective (model,
                                                                                       first_input,
                                                                                       first_target,
                                                                                       &a_candidate_after) &&
                                            itty_feed_model_measure_decoder_objective (model,
                                                                                       second_input,
                                                                                       second_target,
                                                                                       &b_candidate_after) &&
                                            itty_feed_model_measure_segment_node_selection (model,
                                                                                            first_input,
                                                                                            first_target,
                                                                                            &a_selection)) {
                                                clear_trace.a_distance_after_restore = a_candidate_after.selected_distance;
                                                clear_trace.a_false_positive_excess_after_restore = a_candidate_after.false_positive_vote_excess;
                                                clear_trace.b_distance_after_restore = b_candidate_after.selected_distance;
                                                clear_trace.b_false_negative_deficit_after_restore = b_candidate_after.false_negative_vote_deficit;
                                                clear_trace.b_false_positive_excess_after_restore = b_candidate_after.false_positive_vote_excess;
                                                clear_trace.total_flips = stats.flips;
                                                clear_trace.selected_node_after_restore = a_candidate_after.selected_node;
                                                clear_trace.selection_margin_after_restore = a_selection.popcount_gap;
                                                clear_trace.mask_flip_count =
                                                        itty_feed_model_collect_layer_mask_flip_traces (model,
                                                                                                        final_layer,
                                                                                                        base_snapshot,
                                                                                                        restore_mask_traces,
                                                                                                        ITTY_FEED_MODEL_FINISH_CLEAR_SET_TRACE_LIMIT);
                                                clear_trace.overlap_final_output_bits = 0;
                                                for (size_t vote_index = 0; vote_index < clear_trace.clear_vote_count; vote_index++) {
                                                        if (clear_trace.final_output_bits[vote_index] == trace->final_output_bit)
                                                                clear_trace.overlap_final_output_bits++;
                                                        if (itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                                           final_layer,
                                                                                                           clear_trace.final_output_bits[vote_index]) ==
                                                            trace->finish_condensed_bit)
                                                                clear_trace.overlap_condensed_bits++;
                                                }
                                                clear_trace.overlap_mask_flip_locations =
                                                        itty_feed_model_count_mask_flip_trace_overlap (finish_mask_flips,
                                                                                                       finish_mask_flip_count,
                                                                                                       restore_mask_traces,
                                                                                                       clear_trace.mask_flip_count,
                                                                                                       false);
                                                clear_trace.overlap_mask_flip_directions =
                                                        itty_feed_model_count_mask_flip_trace_overlap (finish_mask_flips,
                                                                                                       finish_mask_flip_count,
                                                                                                       restore_mask_traces,
                                                                                                       clear_trace.mask_flip_count,
                                                                                                       true);
                                                clear_trace.a_excess_reduced =
                                                        a_candidate_after.false_positive_vote_excess < trace->a_excess_after_finish;
                                                clear_trace.a_restored =
                                                        a_candidate_after.selected_distance <= a_before->selected_distance &&
                                                        a_candidate_after.false_positive_vote_excess <= a_before->false_positive_vote_excess;
                                                clear_trace.selected_node_preserved =
                                                        a_candidate_after.selected_node == evaluation.selected_index;
                                                clear_trace.strict_preserved =
                                                        b_candidate_after.selected_distance <= b_before->selected_distance &&
                                                        b_candidate_after.false_negative_vote_deficit <= b_before->false_negative_vote_deficit;
                                                clear_trace.distance_preserved =
                                                        b_candidate_after.selected_distance < b_before->selected_distance;
                                                clear_trace.progress_preserved =
                                                        b_candidate_after.selected_distance <= b_before->selected_distance &&
                                                        b_candidate_after.false_negative_vote_deficit < b_before->false_negative_vote_deficit;
                                                clear_trace.no_regression =
                                                        b_candidate_after.selected_distance <= b_before->selected_distance &&
                                                        b_candidate_after.false_negative_vote_deficit <= b_before->false_negative_vote_deficit;

                                                if (trace &&
                                                    trace->family1_clear_set_count < ITTY_FEED_MODEL_FINISH_CLEAR_SET_TRACE_LIMIT)
                                                        trace->family1_clear_set_traces[trace->family1_clear_set_count++] = clear_trace;
                                                if (trace && clear_trace.a_excess_reduced)
                                                        trace->family1_excess_reducing_clear_set_count++;
                                                if (trace && clear_trace.a_restored)
                                                        trace->family1_restoring_clear_set_count++;
                                                if (trace && clear_trace.a_restored && clear_trace.distance_preserved)
                                                        trace->family1_restore_distance_clear_set_count++;
                                                else if (trace && clear_trace.a_restored && clear_trace.no_regression)
                                                        trace->family1_restore_progress_clear_set_count++;
                                                else if (trace && clear_trace.a_restored)
                                                        trace->family1_restore_only_clear_set_count++;
                                                if (trace && clear_trace.distance_preserved)
                                                        trace->family1_distance_preserving_clear_set_count++;
                                                if (trace && clear_trace.progress_preserved)
                                                        trace->family1_progress_preserving_clear_set_count++;

                                                if (clear_trace.a_restored &&
                                                    (!have_best_trace ||
                                                     itty_feed_model_contender_clear_set_trace_is_better (&clear_trace, &best_trace))) {
                                                        if (best_snapshot)
                                                                itty_feed_model_free_layer_state_snapshot (model, best_snapshot);
                                                        best_snapshot = itty_feed_model_snapshot_layer_state (model, final_layer);
                                                        best_trace = clear_trace;
                                                        have_best_trace = true;
                                                        if (trace)
                                                                trace->family1_best_clear_set_index =
                                                                        trace->family1_clear_set_count > 0 ?
                                                                        trace->family1_clear_set_count - 1 : 0;
                                                }
                                        }
                                }

                                if (candidate_condensed)
                                        itty_bit_string_free (candidate_condensed);
                                itty_feed_model_restore_layer_state_snapshot (model, final_layer, candidate_snapshot);

                                size_t advance = votes_needed;
                                while (advance > 0) {
                                        advance--;
                                        if (indices[advance] != clear_count - votes_needed + advance) {
                                                indices[advance]++;
                                                for (size_t next = advance + 1; next < votes_needed; next++)
                                                        indices[next] = indices[next - 1] + 1;
                                                break;
                                        }
                                }
                                if (advance == 0 && indices[0] == clear_count - votes_needed)
                                        done = true;
                        }
                }

                if (have_best_trace) {
                        restored_a = true;
                        local_flip_count = best_trace.total_flips;
                        local_distance_preserved = best_trace.distance_preserved;
                        local_progress_preserved = best_trace.progress_preserved;
                        if (trace) {
                                trace->damage_set_restore_selected_node_before = evaluation.selected_index;
                                trace->damage_set_restore_selected_node_after = best_trace.selected_node_after_restore;
                                trace->damage_set_restore_mask_flip_count = best_trace.mask_flip_count;
                                trace->overlap_final_output_bits = best_trace.overlap_final_output_bits;
                                trace->overlap_condensed_bits = best_trace.overlap_condensed_bits;
                                trace->overlap_mask_flip_locations = best_trace.overlap_mask_flip_locations;
                                trace->overlap_mask_flip_directions = best_trace.overlap_mask_flip_directions;
                                trace->damage_set_restore_available = true;
                                trace->damage_set_restore_excess_reduced = best_trace.a_excess_reduced;
                                trace->damage_set_restore_restored_a = true;
                                trace->damage_set_restore_distance_preserved = best_trace.distance_preserved;
                                trace->damage_set_restore_progress_preserved = best_trace.progress_preserved;
                                trace->damage_set_restore_flips = best_trace.total_flips;
                        }
                        if (best_snapshot) {
                                itty_feed_model_restore_layer_state_snapshot (model, final_layer, best_snapshot);
                                best_snapshot = NULL;
                                itty_feed_model_decoder_objective_t restored_a_objective = { 0 };
                                itty_feed_model_decoder_objective_t restored_b_objective = { 0 };
                                if (itty_feed_model_measure_decoder_objective (model,
                                                                              first_input,
                                                                              first_target,
                                                                              &restored_a_objective) &&
                                    itty_feed_model_measure_decoder_objective (model,
                                                                              second_input,
                                                                              second_target,
                                                                              &restored_b_objective)) {
                                        size_t replacement_flips = 0;
                                        itty_feed_model_decoder_objective_t replacement_a = { 0 };
                                        itty_feed_model_decoder_objective_t replacement_b = { 0 };
                                        bool replaced =
                                                itty_feed_model_try_post_restore_replacement_current_state (model,
                                                                                                            first_input,
                                                                                                            first_target,
                                                                                                            second_input,
                                                                                                            second_target,
                                                                                                            options,
                                                                                                            a_before,
                                                                                                            b_before,
                                                                                                            trace,
                                                                                                            &replacement_flips,
                                                                                                            &replacement_a,
                                                                                                            &replacement_b);

                                        if (replaced) {
                                                restored_a_objective = replacement_a;
                                                restored_b_objective = replacement_b;
                                                local_flip_count += replacement_flips;
                                                local_distance_preserved =
                                                        replacement_b.selected_distance < b_before->selected_distance;
                                                local_progress_preserved =
                                                        replacement_b.selected_distance <= b_before->selected_distance &&
                                                        replacement_b.false_negative_vote_deficit < b_before->false_negative_vote_deficit;
                                        }

                                        if (a_after)
                                                *a_after = restored_a_objective;
                                        if (b_after)
                                                *b_after = restored_b_objective;
                                }
                        }
                }
        }

        if (!restored_a && base_snapshot) {
                itty_feed_model_restore_layer_state_snapshot (model, final_layer, base_snapshot);
                base_snapshot = NULL;
        }

        if (trace && !have_best_trace) {
                trace->damage_set_restore_selected_node_before = evaluation.selected_index;
                trace->damage_set_restore_selected_node_after = evaluation.selected_index;
                trace->damage_set_restore_mask_flip_count = 0;
                trace->overlap_final_output_bits = 0;
                trace->overlap_condensed_bits = 0;
                trace->overlap_mask_flip_locations = 0;
                trace->overlap_mask_flip_directions = 0;
                trace->damage_set_restore_available = false;
                trace->damage_set_restore_excess_reduced = false;
                trace->damage_set_restore_restored_a = false;
                trace->damage_set_restore_distance_preserved = false;
                trace->damage_set_restore_progress_preserved = false;
                trace->damage_set_restore_flips = 0;
        }

        if (damaged_bit_count)
                *damaged_bit_count = local_damaged_bits;
        if (flip_count)
                *flip_count = local_flip_count;
        if (distance_preserved)
                *distance_preserved = local_distance_preserved;
        if (progress_preserved)
                *progress_preserved = local_progress_preserved;

        if (best_snapshot)
                itty_feed_model_free_layer_state_snapshot (model, best_snapshot);
        if (base_snapshot)
                itty_feed_model_free_layer_state_snapshot (model, base_snapshot);
        if (selected_condensed)
                itty_bit_string_free (selected_condensed);
        itty_bit_string_free (evaluation.folded_activation);
        itty_bit_string_list_free (final_outputs);
        if (final_layer_input != first_input)
                itty_bit_string_list_free (final_layer_input);
        return restored_a;
}

static bool
itty_feed_model_project_repair_to_previous_layer_outputs (itty_feed_model_t                      *model,
                                                          size_t                                  layer_index,
                                                          itty_bit_string_list_t                 *previous_outputs,
                                                          itty_feed_model_final_repair_t const   *repair,
                                                          itty_feed_model_layer_assignment_list_t *assignments)
{
        if (repair->value) {
                for (size_t input_index = 0; input_index < previous_outputs->count; input_index++) {
                        itty_bit_string_t *previous_output = itty_bit_string_list_fetch (previous_outputs,
                                                                                         input_index);
                        itty_bit_string_t *mask = model->masks_by_layer_node[layer_index][repair->final_node]->bit_strings[input_index];
                        bool modulated_bit = itty_bit_string_get_bit (previous_output,
                                                                      repair->condensed_bit) !=
                                             itty_bit_string_get_bit (mask,
                                                                      repair->condensed_bit);

                        if (!modulated_bit) {
                                itty_feed_model_layer_assignment_list_append (assignments,
                                                                             input_index,
                                                                             repair->condensed_bit,
                                                                             !itty_bit_string_get_bit (mask,
                                                                                                        repair->condensed_bit));
                        }
                }
        } else {
                itty_bit_string_t *mask = model->masks_by_layer_node[layer_index][repair->final_node]->bit_strings[0];
                itty_feed_model_layer_assignment_list_append (assignments,
                                                             0,
                                                             repair->condensed_bit,
                                                             itty_bit_string_get_bit (mask,
                                                                                      repair->condensed_bit));
        }

        return assignments->count > 0;
}

static bool
itty_feed_model_project_layer_repair_to_previous_outputs (itty_feed_model_t                      *model,
                                                          size_t                                  layer_index,
                                                          itty_bit_string_list_t                 *previous_outputs,
                                                          size_t                                  layer_node,
                                                          size_t                                  condensed_bit,
                                                          bool                                    value,
                                                          itty_feed_model_layer_assignment_list_t *assignments)
{
        if (value) {
                for (size_t input_index = 0; input_index < previous_outputs->count; input_index++) {
                        itty_bit_string_t *previous_output = itty_bit_string_list_fetch (previous_outputs,
                                                                                         input_index);
                        itty_bit_string_t *mask = model->masks_by_layer_node[layer_index][layer_node]->bit_strings[input_index];
                        bool modulated_bit = itty_bit_string_get_bit (previous_output,
                                                                      condensed_bit) !=
                                             itty_bit_string_get_bit (mask,
                                                                      condensed_bit);

                        if (!modulated_bit &&
                            !itty_feed_model_layer_assignment_list_append_unique (assignments,
                                                                                 input_index,
                                                                                 condensed_bit,
                                                                                 !itty_bit_string_get_bit (mask,
                                                                                                            condensed_bit)))
                                return false;
                }
        } else {
                itty_bit_string_t *mask = model->masks_by_layer_node[layer_index][layer_node]->bit_strings[0];
                if (!itty_feed_model_layer_assignment_list_append_unique (assignments,
                                                                         0,
                                                                         condensed_bit,
                                                                         itty_bit_string_get_bit (mask,
                                                                                                  condensed_bit)))
                        return false;
        }

        return assignments->count > 0;
}

static bool
itty_feed_model_make_condensed_assignments_from_outputs (itty_feed_model_t                            *model,
                                                         size_t                                        layer_index,
                                                         itty_feed_model_layer_assignment_list_t const *output_assignments,
                                                         itty_feed_model_layer_assignment_list_t       *condensed_assignments)
{
        for (size_t assignment_index = 0; assignment_index < output_assignments->count; assignment_index++) {
                itty_feed_model_layer_assignment_t const *assignment = &output_assignments->items[assignment_index];
                size_t condensed_bit = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                      layer_index,
                                                                                      assignment->bit_index);

                if (!itty_feed_model_layer_assignment_list_append_unique (condensed_assignments,
                                                                         assignment->layer_node,
                                                                         condensed_bit,
                                                                         assignment->value))
                        return false;
        }

        return condensed_assignments->count > 0;
}

static itty_bit_string_list_t *
itty_feed_model_make_condensed_realistic_outputs (itty_feed_model_t                            *model,
                                                  size_t                                        layer_index,
                                                  itty_bit_string_list_t                       *layer_outputs,
                                                  itty_feed_model_layer_assignment_list_t const *condensed_assignments)
{
        size_t condensed_words = model->vocabulary_words << layer_index;
        itty_bit_string_t **condensed_targets = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));
        bool *touched_nodes = calloc (model->nodes_per_layer, sizeof (bool));

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_t *current_output = itty_bit_string_list_fetch (layer_outputs,
                                                                                node_index);
                itty_bit_string_t *current_condensed = itty_feed_model_reduce_desired_output_for_layer (current_output,
                                                                                                        model->rotations_by_layer[layer_index]);
                condensed_targets[node_index] = itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                                           condensed_words);
                itty_bit_string_free (current_condensed);
        }

        for (size_t assignment_index = 0; assignment_index < condensed_assignments->count; assignment_index++) {
                itty_feed_model_layer_assignment_t const *assignment = &condensed_assignments->items[assignment_index];

                itty_bit_string_set_bit (condensed_targets[assignment->layer_node],
                                         assignment->bit_index,
                                         assignment->value);
                condensed_targets[assignment->layer_node]->pop_count_computed = false;
                touched_nodes[assignment->layer_node] = true;
        }

        itty_bit_string_list_t *candidate_outputs = itty_feed_model_bit_string_list_clone (layer_outputs);
        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                if (!touched_nodes[node_index])
                        continue;

                itty_bit_string_t *candidate_output = itty_feed_model_expand_condensed_output (condensed_targets[node_index],
                                                                                               model->rotations_by_layer[layer_index]);
                itty_bit_string_free (candidate_outputs->bit_strings[node_index]);
                candidate_outputs->bit_strings[node_index] = candidate_output;
        }

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++)
                itty_bit_string_free (condensed_targets[node_index]);
        free (touched_nodes);
        free (condensed_targets);

        return candidate_outputs;
}

static bool
itty_feed_model_score_condensed_realistic_candidate (itty_feed_model_t                            *model,
                                                    size_t                                        layer_index,
                                                    itty_bit_string_list_t                       *layer_outputs,
                                                    itty_bit_string_t                            *target,
                                                    itty_feed_model_layer_assignment_list_t const *condensed_assignments,
                                                    itty_feed_model_decoder_objective_t          *candidate_objective)
{
        itty_bit_string_list_t *candidate_outputs = itty_feed_model_make_condensed_realistic_outputs (model,
                                                                                                      layer_index,
                                                                                                      layer_outputs,
                                                                                                      condensed_assignments);
        bool scored = itty_feed_model_evaluate_suffix_decoder_objective (model,
                                                                         candidate_outputs,
                                                                         layer_index,
                                                                         target,
                                                                         candidate_objective);
        itty_bit_string_list_free (candidate_outputs);

        return scored;
}

static void
itty_feed_model_measure_condensed_assignment_cost (itty_feed_model_t                                  *model,
                                                  size_t                                              layer_index,
                                                  itty_bit_string_list_t                             *input,
                                                  itty_feed_model_layer_assignment_list_t const       *condensed_assignments,
                                                  size_t                                             *estimated_flips,
                                                  size_t                                             *already_satisfied_bits,
                                                  size_t                                             *bits_needing_flips,
                                                  size_t                                             *available_flippable_votes)
{
        size_t threshold = itty_bit_string_list_get_length (input) / 2 + 1;

        *estimated_flips = 0;
        *already_satisfied_bits = 0;
        *bits_needing_flips = 0;
        *available_flippable_votes = 0;

        for (size_t assignment_index = 0; assignment_index < condensed_assignments->count; assignment_index++) {
                itty_feed_model_layer_assignment_t const *assignment = &condensed_assignments->items[assignment_index];
                size_t vote_count = itty_feed_model_count_votes (input,
                                                                 model->masks_by_layer_node[layer_index][assignment->layer_node],
                                                                 assignment->bit_index);
                size_t flips_needed = 0;

                if (assignment->value) {
                        *available_flippable_votes += input->count - vote_count;
                        if (vote_count < threshold)
                                flips_needed = threshold - vote_count;
                } else {
                        *available_flippable_votes += vote_count;
                        if (vote_count >= threshold)
                                flips_needed = vote_count - (threshold - 1);
                }

                if (flips_needed == 0)
                        (*already_satisfied_bits)++;
                else
                        (*bits_needing_flips)++;

                *estimated_flips += flips_needed;
        }
}

static size_t
itty_feed_model_measure_condensed_assignment_flip_cost (itty_feed_model_t                            *model,
                                                       size_t                                        layer_index,
                                                       itty_bit_string_list_t                       *input,
                                                       itty_feed_model_layer_assignment_t const     *assignment)
{
        size_t threshold = itty_bit_string_list_get_length (input) / 2 + 1;
        size_t vote_count = itty_feed_model_count_votes (input,
                                                         model->masks_by_layer_node[layer_index][assignment->layer_node],
                                                         assignment->bit_index);

        if (assignment->value) {
                if (vote_count < threshold)
                        return threshold - vote_count;
        } else {
                if (vote_count >= threshold)
                        return vote_count - (threshold - 1);
        }

        return 0;
}

static size_t
itty_feed_model_measure_candidate_marginal_cost (itty_feed_model_t                                  *model,
                                                size_t                                              layer_index,
                                                itty_bit_string_list_t                             *input,
                                                itty_feed_model_projected_repair_candidate_t const *candidate,
                                                itty_bit_string_t                                 **cares,
                                                itty_bit_string_t                                 **targets)
{
        size_t cost = 0;

        for (size_t assignment_index = 0; assignment_index < candidate->condensed_assignments.count; assignment_index++) {
                itty_feed_model_layer_assignment_t *assignment = &candidate->condensed_assignments.items[assignment_index];

                if (itty_bit_string_get_bit (cares[assignment->layer_node],
                                             assignment->bit_index) &&
                    itty_bit_string_get_bit (targets[assignment->layer_node],
                                             assignment->bit_index) == assignment->value)
                        continue;

                cost += itty_feed_model_measure_condensed_assignment_flip_cost (model,
                                                                               layer_index,
                                                                               input,
                                                                               assignment);
        }

        return cost;
}

static void
itty_feed_model_collect_candidate_mask_flips (itty_feed_model_t                                  *model,
                                             size_t                                              layer_index,
                                             itty_bit_string_list_t                             *input,
                                             itty_feed_model_projected_repair_candidate_t const *candidate,
                                             itty_feed_model_mask_flip_list_t                   *flips)
{
        size_t threshold = itty_bit_string_list_get_length (input) / 2 + 1;

        for (size_t assignment_index = 0; assignment_index < candidate->condensed_assignments.count; assignment_index++) {
                itty_feed_model_layer_assignment_t const *assignment = &candidate->condensed_assignments.items[assignment_index];
                size_t vote_count = itty_feed_model_count_votes (input,
                                                                 model->masks_by_layer_node[layer_index][assignment->layer_node],
                                                                 assignment->bit_index);

                if (assignment->value && vote_count >= threshold)
                        continue;
                if (!assignment->value && vote_count < threshold)
                        continue;

                for (size_t input_index = 0; input_index < input->count; input_index++) {
                        if (assignment->value && vote_count >= threshold)
                                break;
                        if (!assignment->value && vote_count < threshold)
                                break;

                        itty_bit_string_t *input_bit_string = input->bit_strings[input_index];
                        itty_bit_string_t *mask = model->masks_by_layer_node[layer_index][assignment->layer_node]->bit_strings[input_index];
                        bool masked_bit = itty_feed_model_masked_input_bit (input_bit_string,
                                                                            mask,
                                                                            assignment->bit_index);

                        if (masked_bit == assignment->value)
                                continue;

                        itty_feed_model_mask_flip_list_append (flips,
                                                               assignment->layer_node,
                                                               input_index,
                                                               assignment->bit_index);
                        if (assignment->value)
                                vote_count++;
                        else
                                vote_count--;
                }
        }
}

static size_t
itty_feed_model_score_replay_mask_flip (itty_feed_model_t                                      *model,
                                        size_t                                                  layer_index,
                                        itty_feed_model_mask_flip_t const                      *flip,
                                        itty_feed_model_replay_example_t const                 *replay_examples,
                                        size_t                                                  replay_example_count,
                                        itty_feed_model_decoder_objective_t const              *before_objectives,
                                        itty_bit_string_t                                     **before_folded,
                                        size_t                                                  false_positive_penalty,
                                        size_t                                                  false_negative_penalty,
                                        bool                                                   *unsafe,
                                        bool                                                   *false_positive,
                                        bool                                                   *false_negative,
                                        bool                                                   *weakening,
                                        size_t                                                 *false_positive_damage)
{
        itty_bit_string_t *mask = model->masks_by_layer_node[layer_index][flip->layer_node]->bit_strings[flip->input_index];
        itty_feed_model_refreshed_projected_repair_round_t replay_stats = { 0 };
        size_t margin_penalty = false_positive_penalty == 0 ? 1 : false_positive_penalty;
        size_t cost = 0;

        *unsafe = false;
        *false_positive = false;
        *false_negative = false;
        *weakening = false;
        *false_positive_damage = 0;

        itty_feed_model_flip_mask_bit (mask,
                                       flip->bit_index);
        itty_feed_model_score_replay_after_batch (model,
                                                  replay_examples,
                                                  replay_example_count,
                                                  false,
                                                  before_objectives,
                                                  before_folded,
                                                  &replay_stats);
        itty_feed_model_flip_mask_bit (mask,
                                       flip->bit_index);

        if (replay_stats.replay_transitions.correct_zero_to_false_positive_bits > 0) {
                *unsafe = true;
                *false_positive = true;
                *false_positive_damage = replay_stats.replay_transitions.correct_zero_to_false_positive_bits;
                cost = false_positive_penalty == 0 ? 1 : false_positive_penalty;
        } else if (replay_stats.replay_transitions.correct_one_to_false_negative_bits > 0) {
                *unsafe = true;
                *false_negative = true;
                cost = false_negative_penalty == 0 ? 1 : false_negative_penalty;
        } else if (replay_stats.replay_false_positive_excess_delta < 0 ||
                   replay_stats.replay_target_zero_safety_delta < 0 ||
                   replay_stats.replay_target_one_margin_delta < 0) {
                *weakening = true;
                cost = margin_penalty;
        }

        return cost;
}

static bool
itty_feed_model_build_replay_taboo_map_for_layer (itty_feed_model_t                             *model,
                                                  size_t                                         layer_index,
                                                  itty_feed_model_replay_example_t const        *replay_examples,
                                                  size_t                                         replay_example_count,
                                                  itty_feed_model_decoder_objective_t const     *before_objectives,
                                                  itty_bit_string_t                            **before_folded,
                                                  itty_feed_model_replay_taboo_map_t            *map)
{
        if (replay_example_count == 0)
                return true;

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_list_t *masks = model->masks_by_layer_node[layer_index][node_index];

                for (size_t input_index = 0; input_index < masks->count; input_index++) {
                        itty_bit_string_t *mask = masks->bit_strings[input_index];
                        size_t bit_capacity = itty_bit_string_get_length (mask);

                        for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                                itty_feed_model_mask_flip_t flip = {
                                        .layer_node = node_index,
                                        .input_index = input_index,
                                        .bit_index = bit_index,
                                };
                                bool unsafe = false;
                                bool false_positive = false;
                                bool false_negative = false;
                                bool weakening = false;
                                size_t false_positive_damage = 0;
                                bool desired_value = !itty_bit_string_get_bit (mask,
                                                                               bit_index);

                                itty_feed_model_score_replay_mask_flip (model,
                                                                        layer_index,
                                                                        &flip,
                                                                        replay_examples,
                                                                        replay_example_count,
                                                                        before_objectives,
                                                                        before_folded,
                                                                        1,
                                                                        1,
                                                                        &unsafe,
                                                                        &false_positive,
                                                                        &false_negative,
                                                                        &weakening,
                                                                        &false_positive_damage);
                                if (!false_positive && !false_negative)
                                        continue;

                                if (!itty_feed_model_replay_taboo_map_record (map,
                                                                              layer_index,
                                                                              node_index,
                                                                              input_index,
                                                                              bit_index,
                                                                              desired_value,
                                                                              false_positive,
                                                                              false_negative))
                                        return false;
                        }
                }
        }

        return true;
}

static size_t
itty_feed_model_measure_replay_taboo_penalty_for_assignments (itty_feed_model_t                              *model,
                                                              size_t                                          layer_index,
                                                              itty_bit_string_list_t                         *layer_input,
                                                              itty_feed_model_layer_assignment_list_t const  *condensed_assignments,
                                                              itty_feed_model_replay_taboo_map_t const       *taboo_map,
                                                              size_t                                          taboo_flip_penalty,
                                                              bool                                           *has_taboo_flip,
                                                              itty_feed_model_projected_repair_stats_t       *stats)
{
        if (has_taboo_flip)
                *has_taboo_flip = false;
        if (!taboo_map || taboo_map->count == 0 || taboo_flip_penalty == 0)
                return 0;
        itty_feed_model_mask_flip_list_t flips = { 0 };
        size_t taboo_penalty = 0;
        bool candidate_taboo = false;

        itty_feed_model_projected_repair_candidate_t candidate = {
                .condensed_assignments = *condensed_assignments,
        };
        itty_feed_model_collect_candidate_mask_flips (model,
                                                     layer_index,
                                                     layer_input,
                                                     &candidate,
                                                     &flips);

        for (size_t flip_index = 0; flip_index < flips.count; flip_index++) {
                itty_feed_model_mask_flip_t *flip = &flips.items[flip_index];
                itty_bit_string_t *mask = model->masks_by_layer_node[layer_index][flip->layer_node]->bit_strings[flip->input_index];
                bool desired_value = !itty_bit_string_get_bit (mask,
                                                               flip->bit_index);
                itty_feed_model_replay_taboo_entry_t *entry =
                        itty_feed_model_replay_taboo_map_find ((itty_feed_model_replay_taboo_map_t *) taboo_map,
                                                               layer_index,
                                                               flip->layer_node,
                                                               flip->input_index,
                                                               flip->bit_index,
                                                               desired_value);

                if (!entry)
                        continue;

                candidate_taboo = true;
                if (has_taboo_flip)
                        *has_taboo_flip = true;
                taboo_penalty += taboo_flip_penalty;
                if (stats)
                        stats->replay_taboo_mask_flips++;
        }

        if (stats && candidate_taboo) {
                stats->replay_taboo_vote_candidates++;
                stats->replay_taboo_penalty_total += taboo_penalty;
        }

        itty_feed_model_mask_flip_list_clear (&flips);

        return taboo_penalty;
}

static void
itty_feed_model_measure_minus_one_bad_flip_candidate (itty_feed_model_t                                  *model,
                                                      size_t                                              layer_index,
                                                      itty_bit_string_list_t                             *layer_input,
                                                      itty_bit_string_t                                  *target,
                                                      itty_feed_model_projected_repair_candidate_t const *candidate,
                                                      itty_feed_model_replay_example_t const             *replay_examples,
                                                      size_t                                              replay_example_count,
                                                      itty_feed_model_decoder_objective_t const          *before_objective,
                                                      itty_feed_model_decoder_objective_t const          *before_replay_objectives,
                                                      itty_bit_string_t                                 **before_replay_folded,
                                                      itty_feed_model_projected_repair_stats_t           *stats)
{
        if (!stats || replay_example_count == 0 || candidate->use_residual)
                return;

        itty_feed_model_mask_flip_list_t flips = { 0 };
        size_t bad_flip_index = (size_t) -1;
        size_t bad_flip_count = 0;

        itty_feed_model_collect_candidate_mask_flips (model,
                                                     layer_index,
                                                     layer_input,
                                                     candidate,
                                                     &flips);

        for (size_t flip_index = 0; flip_index < flips.count; flip_index++) {
                bool unsafe = false;
                bool false_positive = false;
                bool false_negative = false;
                bool weakening = false;
                size_t false_positive_damage = 0;

                itty_feed_model_score_replay_mask_flip (model,
                                                        layer_index,
                                                        &flips.items[flip_index],
                                                        replay_examples,
                                                        replay_example_count,
                                                        before_replay_objectives,
                                                        before_replay_folded,
                                                        1,
                                                        1,
                                                        &unsafe,
                                                        &false_positive,
                                                        &false_negative,
                                                        &weakening,
                                                        &false_positive_damage);
                if (!false_positive)
                        continue;

                bad_flip_index = flip_index;
                bad_flip_count++;
                if (bad_flip_count > 1)
                        break;
        }

        if (bad_flip_count != 1) {
                itty_feed_model_mask_flip_list_clear (&flips);
                return;
        }

        stats->replay_minus_one_bad_candidates++;

        itty_feed_model_layer_state_snapshot_t *snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                 layer_index);
        for (size_t flip_index = 0; flip_index < flips.count; flip_index++) {
                if (flip_index == bad_flip_index)
                        continue;
                itty_feed_model_mask_flip_t *flip = &flips.items[flip_index];
                itty_bit_string_t *mask = model->masks_by_layer_node[layer_index][flip->layer_node]->bit_strings[flip->input_index];

                itty_feed_model_flip_mask_bit (mask,
                                               flip->bit_index);
        }

        itty_feed_model_refreshed_projected_repair_round_t replay_stats = { 0 };
        bool replay_safe = itty_feed_model_score_replay_after_batch (model,
                                                                     replay_examples,
                                                                     replay_example_count,
                                                                     true,
                                                                     before_replay_objectives,
                                                                     before_replay_folded,
                                                                     &replay_stats);
        if (replay_safe &&
            replay_stats.replay_distance_delta >= 0 &&
            replay_stats.replay_false_positive_excess_delta >= 0 &&
            replay_stats.replay_target_one_margin_delta >= 0 &&
            replay_stats.replay_target_zero_safety_delta >= 0) {
                stats->replay_minus_one_bad_safe_candidates++;

                itty_bit_string_list_t *after_layer_outputs = itty_feed_model_run_layer (model,
                                                                                         layer_index,
                                                                                         layer_input);
                itty_feed_model_decoder_objective_t after_objective = { 0 };
                if (itty_feed_model_evaluate_suffix_decoder_objective (model,
                                                                       after_layer_outputs,
                                                                       layer_index,
                                                                       target,
                                                                       &after_objective)) {
                        if (after_objective.selected_distance < before_objective->selected_distance)
                                stats->replay_minus_one_bad_strict_candidates++;
                        else if (after_objective.false_negative_vote_deficit < before_objective->false_negative_vote_deficit)
                                stats->replay_minus_one_bad_deficit_candidates++;
                }
                itty_bit_string_list_free (after_layer_outputs);
        }

        itty_feed_model_restore_layer_state_snapshot (model,
                                                      layer_index,
                                                      snapshot);
        itty_feed_model_mask_flip_list_clear (&flips);
}

static void
itty_feed_model_measure_alternate_replay_realization (itty_feed_model_t                                      *model,
                                                     size_t                                                  layer_index,
                                                     itty_bit_string_list_t                                 *input,
                                                     itty_feed_model_projected_repair_candidate_t const     *candidate,
                                                     itty_feed_model_replay_example_t const                 *replay_examples,
                                                     size_t                                                  replay_example_count,
                                                     itty_feed_model_decoder_objective_t const              *before_objectives,
                                                     itty_bit_string_t                                     **before_folded,
                                                     size_t                                                  false_positive_penalty,
                                                     size_t                                                  false_negative_penalty,
                                                     size_t                                                 *alternate_flips,
                                                     size_t                                                 *alternate_unsafe_flips,
                                                     size_t                                                 *alternate_cost)
{
        size_t threshold = itty_bit_string_list_get_length (input) / 2 + 1;

        *alternate_flips = 0;
        *alternate_unsafe_flips = 0;
        *alternate_cost = 0;

        for (size_t assignment_index = 0; assignment_index < candidate->condensed_assignments.count; assignment_index++) {
                itty_feed_model_layer_assignment_t const *assignment = &candidate->condensed_assignments.items[assignment_index];
                size_t vote_count = itty_feed_model_count_votes (input,
                                                                 model->masks_by_layer_node[layer_index][assignment->layer_node],
                                                                 assignment->bit_index);
                size_t flips_needed = 0;
                itty_feed_model_scored_mask_flip_t *scored_flips = NULL;
                size_t scored_flip_count = 0;
                size_t scored_flip_capacity = 0;

                if (assignment->value) {
                        if (vote_count < threshold)
                                flips_needed = threshold - vote_count;
                } else if (vote_count >= threshold) {
                        flips_needed = vote_count - (threshold - 1);
                }

                if (flips_needed == 0)
                        continue;

                for (size_t input_index = 0; input_index < input->count; input_index++) {
                        itty_bit_string_t *input_bit_string = input->bit_strings[input_index];
                        itty_bit_string_t *mask = model->masks_by_layer_node[layer_index][assignment->layer_node]->bit_strings[input_index];
                        bool masked_bit = itty_feed_model_masked_input_bit (input_bit_string,
                                                                            mask,
                                                                            assignment->bit_index);

                        if (masked_bit == assignment->value)
                                continue;

                        if (scored_flip_count == scored_flip_capacity) {
                                scored_flip_capacity = scored_flip_capacity == 0 ? 8 : scored_flip_capacity * 2;
                                scored_flips = realloc (scored_flips,
                                                        scored_flip_capacity * sizeof (itty_feed_model_scored_mask_flip_t));
                        }

                        itty_feed_model_mask_flip_t flip = {
                                .layer_node = assignment->layer_node,
                                .input_index = input_index,
                                .bit_index = assignment->bit_index
                        };
                        bool unsafe = false;
                        bool false_positive = false;
                        bool false_negative = false;
                        bool weakening = false;
                        size_t false_positive_damage = 0;
                        size_t cost = itty_feed_model_score_replay_mask_flip (model,
                                                                              layer_index,
                                                                              &flip,
                                                                              replay_examples,
                                                                              replay_example_count,
                                                                              before_objectives,
                                                                              before_folded,
                                                                              false_positive_penalty,
                                                                              false_negative_penalty,
                                                                              &unsafe,
                                                                              &false_positive,
                                                                              &false_negative,
                                                                              &weakening,
                                                                              &false_positive_damage);

                        scored_flips[scored_flip_count++] = (itty_feed_model_scored_mask_flip_t) {
                                .flip = flip,
                                .cost = cost,
                                .unsafe = unsafe || weakening
                        };
                }

                qsort (scored_flips,
                       scored_flip_count,
                       sizeof (itty_feed_model_scored_mask_flip_t),
                       compare_scored_mask_flips_by_cost);

                for (size_t scored_flip_index = 0;
                     scored_flip_index < scored_flip_count && scored_flip_index < flips_needed;
                     scored_flip_index++) {
                        (*alternate_flips)++;
                        *alternate_cost += scored_flips[scored_flip_index].cost;
                        if (scored_flips[scored_flip_index].unsafe)
                                (*alternate_unsafe_flips)++;
                }

                free (scored_flips);
        }
}

static size_t
itty_feed_model_measure_replay_vote_protection_penalty (itty_feed_model_t                       *model,
                                                        itty_feed_model_replay_example_t const  *replay_examples,
                                                        size_t                                   replay_example_count,
                                                        size_t                                   final_node,
                                                        size_t                                   output_bit,
                                                        size_t                                   decoded_bit,
                                                        bool                                     candidate_value,
                                                        size_t                                   zero_penalty,
                                                        size_t                                   one_penalty)
{
        size_t penalty = 0;

        if (replay_example_count == 0 ||
            (zero_penalty == 0 && one_penalty == 0))
                return 0;

        for (size_t replay_index = 0; replay_index < replay_example_count; replay_index++) {
                itty_feed_model_decoder_objective_t replay_objective = { 0 };
                itty_bit_string_t *folded = NULL;

                if (!itty_feed_model_evaluate_replay_example (model,
                                                              &replay_examples[replay_index],
                                                              &replay_objective,
                                                              &folded))
                        continue;

                if (replay_objective.selected_distance == 0) {
                        itty_bit_string_list_t *outputs = itty_feed_model_run_to_layer_input (model,
                                                                                              replay_examples[replay_index].input,
                                                                                              model->number_of_layers);
                        itty_bit_string_t *node_output = itty_bit_string_list_fetch (outputs,
                                                                                     final_node);
                        bool replay_target_bit = itty_bit_string_get_bit (replay_examples[replay_index].target,
                                                                          decoded_bit);
                        bool replay_vote_bit = itty_bit_string_get_bit (node_output,
                                                                        output_bit);

                        if (candidate_value && !replay_target_bit && !replay_vote_bit)
                                penalty += zero_penalty;
                        else if (!candidate_value && replay_target_bit && replay_vote_bit)
                                penalty += one_penalty;

                        itty_bit_string_list_free (outputs);
                }

                itty_bit_string_free (folded);
        }

        return penalty;
}

static bool
itty_feed_model_collect_segment_final_repairs_for_penultimate (itty_feed_model_t                  *model,
                                                               itty_bit_string_list_t             *penultimate_input,
                                                               itty_bit_string_list_t             *penultimate_outputs,
                                                               itty_bit_string_t                  *target,
                                                               itty_feed_model_projected_repair_options_t const *options,
                                                               itty_feed_model_final_repair_list_t *repairs,
                                                               itty_feed_model_projected_repair_stats_t *stats)
{
        size_t penultimate_layer = model->number_of_layers - 2;
        size_t final_layer = model->number_of_layers - 1;
        itty_bit_string_list_t *final_outputs = itty_feed_model_run_layer (model,
                                                                           final_layer,
                                                                           penultimate_outputs);
        itty_feed_model_decoder_objective_t before_objective = { 0 };
        if (!itty_feed_model_evaluate_decoder_objective (model,
                                                         final_outputs,
                                                         target,
                                                         &before_objective)) {
                itty_bit_string_list_free (final_outputs);
                return false;
        }

        size_t condensed_words = model->vocabulary_words << final_layer;
        size_t bit_capacity = condensed_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t target_bit_capacity = itty_bit_string_get_number_of_words (target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t output_bit_capacity = bit_capacity * 2;
        size_t top_k_segment_vote_alternatives = options && options->top_k_segment_vote_alternatives != 0 ?
                                                 options->top_k_segment_vote_alternatives :
                                                 1;
        itty_feed_model_replay_example_t const *replay_examples = options ? options->replay_examples : NULL;
        size_t replay_example_count = options ? options->replay_example_count : 0;
        size_t replay_zero_protection_penalty = options ? options->replay_zero_protection_penalty : 0;
        size_t replay_one_protection_penalty = options ? options->replay_one_protection_penalty : 0;
        size_t replay_taboo_flip_penalty = options ? options->replay_taboo_flip_penalty : 0;
        bool strict_replay_taboo_rejection = options ? options->strict_replay_taboo_rejection : false;
        bool reserve_replay_protected_zero_votes = options ? options->reserve_replay_protected_zero_votes : false;
        itty_feed_model_decoder_objective_t *before_replay_objectives = NULL;
        itty_bit_string_t **before_replay_folded = NULL;
        itty_feed_model_replay_taboo_map_t replay_taboo_map = { 0 };

        if (replay_example_count > 0 &&
            replay_taboo_flip_penalty > 0) {
                before_replay_objectives = calloc (replay_example_count,
                                                   sizeof (itty_feed_model_decoder_objective_t));
                before_replay_folded = calloc (replay_example_count,
                                               sizeof (itty_bit_string_t *));
                if (!itty_feed_model_measure_replay_examples (model,
                                                              replay_examples,
                                                              replay_example_count,
                                                              before_replay_objectives,
                                                              before_replay_folded)) {
                        itty_feed_model_replay_taboo_map_clear (&replay_taboo_map);
                        free (before_replay_folded);
                        free (before_replay_objectives);
                        itty_bit_string_list_free (final_outputs);
                        return false;
                }
                if (!itty_feed_model_build_replay_taboo_map_for_layer (model,
                                                                       penultimate_layer,
                                                                       replay_examples,
                                                                       replay_example_count,
                                                                       before_replay_objectives,
                                                                       before_replay_folded,
                                                                       &replay_taboo_map)) {
                        itty_feed_model_replay_taboo_map_clear (&replay_taboo_map);
                        for (size_t replay_index = 0; replay_index < replay_example_count; replay_index++)
                                if (before_replay_folded[replay_index])
                                        itty_bit_string_free (before_replay_folded[replay_index]);
                        free (before_replay_folded);
                        free (before_replay_objectives);
                        itty_bit_string_list_free (final_outputs);
                        return false;
                }
        }

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_t *current_condensed = itty_feed_model_run_node_condensed (penultimate_outputs,
                                                                                           model->masks_by_layer_node[final_layer][node_index]);
                itty_bit_string_t *working_condensed = itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                                                  condensed_words);
                itty_feed_model_decoder_objective_t current_objective = before_objective;

                for (size_t target_bit_index = 0; target_bit_index < target_bit_capacity; target_bit_index++) {
                        itty_bit_string_t *working_output = itty_feed_model_expand_condensed_output (working_condensed,
                                                                                                     model->rotations_by_layer[final_layer]);
                        itty_bit_string_list_t *working_outputs = itty_feed_model_bit_string_list_clone (final_outputs);
                        itty_bit_string_free (working_outputs->bit_strings[node_index]);
                        working_outputs->bit_strings[node_index] = working_output;
                        itty_feed_model_output_evaluation_t working_evaluation = { 0 };
                        if (!itty_feed_model_evaluate_output (model,
                                                              working_outputs,
                                                              target,
                                                              &working_evaluation)) {
                                itty_bit_string_list_free (working_outputs);
                                continue;
                        }

                        bool target_bit = itty_bit_string_get_bit (target,
                                                                   target_bit_index);
                        bool folded_bit = itty_bit_string_get_bit (working_evaluation.folded_activation,
                                                                   target_bit_index);
                        itty_bit_string_free (working_evaluation.folded_activation);
                        itty_bit_string_list_free (working_outputs);

                        if (folded_bit == target_bit)
                                continue;

                        size_t votes_needed = 0;
                        if (!itty_feed_model_get_segment_condense_vote_need (model,
                                                                             target_bit_index,
                                                                             target_bit,
                                                                             working_condensed,
                                                                             &votes_needed))
                                continue;

                        itty_feed_model_quota_vote_candidate_t *vote_candidates = NULL;
                        size_t vote_candidate_count = 0;
                        size_t vote_candidate_capacity = 0;
                        size_t original_index = 0;

                        for (size_t output_bit_index = target_bit_index;
                             output_bit_index < output_bit_capacity;
                             output_bit_index += target_bit_capacity) {
                                size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                                            final_layer,
                                                                                                            output_bit_index);
                                if (itty_bit_string_get_bit (working_condensed,
                                                             condensed_bit_index) == target_bit) {
                                        original_index++;
                                        continue;
                                }

                                size_t replay_protection_penalty =
                                        itty_feed_model_measure_replay_vote_protection_penalty (model,
                                                                                                replay_examples,
                                                                                                replay_example_count,
                                                                                                node_index,
                                                                                                output_bit_index,
                                                                                                target_bit_index,
                                                                                                target_bit,
                                                                                                replay_zero_protection_penalty,
                                                                                                replay_one_protection_penalty);
                                if (stats &&
                                    target_bit &&
                                    replay_protection_penalty > 0)
                                        stats->replay_direct_protected_zero_hit_candidates++;
                                if (reserve_replay_protected_zero_votes &&
                                    target_bit &&
                                    replay_protection_penalty > 0) {
                                        if (stats)
                                                stats->replay_reserved_zero_votes++;
                                        original_index++;
                                        continue;
                                }

                                itty_feed_model_final_repair_t repair = {
                                        .final_node = node_index,
                                        .condensed_bit = condensed_bit_index,
                                        .quota_size = votes_needed,
                                        .value = target_bit
                                };
                                itty_feed_model_layer_assignment_list_t output_assignments = { 0 };
                                itty_feed_model_layer_assignment_list_t condensed_assignments = { 0 };
                                size_t estimated_flips = 0;
                                size_t already_satisfied_bits = 0;
                                size_t bits_needing_flips = 0;
                                size_t available_flippable_votes = 0;

                                if (itty_feed_model_project_repair_to_previous_layer_outputs (model,
                                                                                              final_layer,
                                                                                              penultimate_outputs,
                                                                                              &repair,
                                                                                              &output_assignments) &&
                                    itty_feed_model_make_condensed_assignments_from_outputs (model,
                                                                                             penultimate_layer,
                                                                                             &output_assignments,
                                                                                             &condensed_assignments)) {
                                        itty_feed_model_measure_condensed_assignment_cost (model,
                                                                                          penultimate_layer,
                                                                                          penultimate_input,
                                                                                          &condensed_assignments,
                                                                                          &estimated_flips,
                                                                                          &already_satisfied_bits,
                                                                                          &bits_needing_flips,
                                                                                          &available_flippable_votes);
                                        bool has_taboo_flip = false;
                                        size_t replay_taboo_penalty =
                                                itty_feed_model_measure_replay_taboo_penalty_for_assignments (model,
                                                                                                              penultimate_layer,
                                                                                                              penultimate_input,
                                                                                                              &condensed_assignments,
                                                                                                              &replay_taboo_map,
                                                                                                              replay_taboo_flip_penalty,
                                                                                                              &has_taboo_flip,
                                                                                                              stats);
                                        if (has_taboo_flip &&
                                            strict_replay_taboo_rejection) {
                                                itty_feed_model_projected_repair_candidate_t taboo_candidate = {
                                                        .condensed_assignments = condensed_assignments,
                                                };

                                                itty_feed_model_measure_minus_one_bad_flip_candidate (model,
                                                                                                      penultimate_layer,
                                                                                                      penultimate_input,
                                                                                                      target,
                                                                                                      &taboo_candidate,
                                                                                                      replay_examples,
                                                                                                      replay_example_count,
                                                                                                      &current_objective,
                                                                                                      before_replay_objectives,
                                                                                                      before_replay_folded,
                                                                                                      stats);
                                                if (stats)
                                                        stats->replay_taboo_rejected_vote_candidates++;
                                                itty_feed_model_layer_assignment_list_clear (&condensed_assignments);
                                                itty_feed_model_layer_assignment_list_clear (&output_assignments);
                                                original_index++;
                                                continue;
                                        }
                                        if (vote_candidate_count == vote_candidate_capacity) {
                                                vote_candidate_capacity = vote_candidate_capacity == 0 ? 16 : vote_candidate_capacity * 2;
                                                vote_candidates = realloc (vote_candidates,
                                                                          vote_candidate_capacity * sizeof (itty_feed_model_quota_vote_candidate_t));
                                        }
                                        vote_candidates[vote_candidate_count++] = (itty_feed_model_quota_vote_candidate_t) {
                                                .condensed_bit = condensed_bit_index,
                                                .output_bit = output_bit_index,
                                                .estimated_flips = estimated_flips,
                                                .replay_protection_penalty = replay_protection_penalty,
                                                .replay_taboo_penalty = replay_taboo_penalty,
                                                .original_index = original_index
                                        };
                                }

                                itty_feed_model_layer_assignment_list_clear (&condensed_assignments);
                                itty_feed_model_layer_assignment_list_clear (&output_assignments);
                                original_index++;
                        }

                        if (vote_candidate_count < votes_needed) {
                                free (vote_candidates);
                                continue;
                        }

                        qsort (vote_candidates,
                               vote_candidate_count,
                               sizeof (itty_feed_model_quota_vote_candidate_t),
                               compare_quota_vote_candidates_by_cost);

                        itty_bit_string_t *next_working_condensed = NULL;
                        itty_feed_model_decoder_objective_t next_working_objective = current_objective;
                        for (size_t alternative_index = 0;
                             alternative_index < top_k_segment_vote_alternatives;
                             alternative_index++) {
                                if (alternative_index > 0 &&
                                    votes_needed + alternative_index - 1 >= vote_candidate_count)
                                        break;

                                itty_bit_string_t *candidate_condensed = itty_feed_model_bit_string_clone_to_words (working_condensed,
                                                                                                                     condensed_words);
                                for (size_t vote_index = 0; vote_index < votes_needed; vote_index++) {
                                        size_t candidate_index = vote_index;
                                        if (alternative_index > 0 &&
                                            vote_index == (alternative_index - 1) % votes_needed)
                                                candidate_index = votes_needed + alternative_index - 1;

                                        itty_bit_string_set_bit (candidate_condensed,
                                                                 vote_candidates[candidate_index].condensed_bit,
                                                                 target_bit);
                                }
                                candidate_condensed->pop_count_computed = false;

                                itty_bit_string_t *candidate_output = itty_feed_model_expand_condensed_output (candidate_condensed,
                                                                                                               model->rotations_by_layer[final_layer]);
                                itty_bit_string_list_t *candidate_outputs = itty_feed_model_bit_string_list_clone (final_outputs);
                                itty_bit_string_free (candidate_outputs->bit_strings[node_index]);
                                candidate_outputs->bit_strings[node_index] = candidate_output;

                                itty_feed_model_decoder_objective_t candidate_objective = { 0 };
                                if (itty_feed_model_evaluate_decoder_objective (model,
                                                                                candidate_outputs,
                                                                                target,
                                                                                &candidate_objective) &&
                                    itty_feed_model_decoder_objective_accepts (&current_objective,
                                                                               &candidate_objective)) {
                                        for (size_t vote_index = 0; vote_index < votes_needed; vote_index++) {
                                                size_t candidate_index = vote_index;
                                                if (alternative_index > 0 &&
                                                    vote_index == (alternative_index - 1) % votes_needed)
                                                        candidate_index = votes_needed + alternative_index - 1;

                                                itty_feed_model_final_repair_list_append (repairs,
                                                                                          node_index,
                                                                                          vote_candidates[candidate_index].condensed_bit,
                                                                                          vote_candidates[candidate_index].output_bit,
                                                                                          target_bit_index,
                                                                                          votes_needed,
                                                                                          target_bit);
                                        }
                                        if (!next_working_condensed) {
                                                next_working_condensed = itty_feed_model_bit_string_clone_to_words (candidate_condensed,
                                                                                                                   condensed_words);
                                                next_working_objective = candidate_objective;
                                        }
                                }

                                itty_bit_string_list_free (candidate_outputs);
                                itty_bit_string_free (candidate_condensed);
                        }
                        if (next_working_condensed) {
                                itty_bit_string_free (working_condensed);
                                working_condensed = next_working_condensed;
                                current_objective = next_working_objective;
                        }
                        free (vote_candidates);
                }

                itty_bit_string_free (working_condensed);
                itty_bit_string_free (current_condensed);
        }

        if (before_replay_folded) {
                for (size_t replay_index = 0; replay_index < replay_example_count; replay_index++)
                        if (before_replay_folded[replay_index])
                                itty_bit_string_free (before_replay_folded[replay_index]);
        }
        free (before_replay_folded);
        free (before_replay_objectives);
        itty_feed_model_replay_taboo_map_clear (&replay_taboo_map);
        itty_bit_string_list_free (final_outputs);

        return true;
}

static bool
itty_feed_model_measure_or_residual_assignment_cost (itty_feed_model_t                            *model,
                                                    size_t                                        layer_index,
                                                    itty_bit_string_list_t                       *input,
                                                    itty_feed_model_layer_assignment_list_t const *condensed_assignments,
                                                    size_t                                       *estimated_flips,
                                                    size_t                                       *enable_flips,
                                                    size_t                                       *mask_flips)
{
        *estimated_flips = 0;
        *enable_flips = 0;
        *mask_flips = 0;

        if (input->count == 0)
                return false;

        for (size_t assignment_index = 0; assignment_index < condensed_assignments->count; assignment_index++) {
                itty_feed_model_layer_assignment_t const *assignment = &condensed_assignments->items[assignment_index];

                if (!assignment->value)
                        return false;

                itty_bit_string_t *condensed = itty_feed_model_run_node_condensed (input,
                                                                                   model->masks_by_layer_node[layer_index][assignment->layer_node]);
                bool condensed_bit = itty_bit_string_get_bit (condensed,
                                                              assignment->bit_index);
                itty_bit_string_free (condensed);
                if (condensed_bit)
                        continue;

                itty_bit_string_t *enable = model->residual_enable_by_layer_node[layer_index][assignment->layer_node];
                itty_bit_string_t *mask = model->residual_mask_by_layer_node[layer_index][assignment->layer_node];
                itty_bit_string_t *skip_input = itty_bit_string_list_fetch (input,
                                                                            assignment->layer_node % input->count);
                bool skip_bit = itty_bit_string_get_bit (skip_input,
                                                         assignment->bit_index) !=
                                itty_bit_string_get_bit (mask,
                                                         assignment->bit_index);

                if (!itty_bit_string_get_bit (enable,
                                              assignment->bit_index))
                        (*enable_flips)++;
                if (!skip_bit)
                        (*mask_flips)++;
        }

        *estimated_flips = *enable_flips + *mask_flips;
        return true;
}

static bool
itty_feed_model_projected_candidate_has_decoder_effect (itty_feed_model_projected_repair_candidate_t const *candidate,
                                                        itty_feed_model_decoder_objective_t const          *before_objective,
                                                        itty_feed_model_projected_repair_stats_t           *stats)
{
        bool decoded_vote_count_changes = candidate->vote_deficit_delta > 0 ||
                                          candidate->target_one_margin_delta > 0 ||
                                          candidate->false_negative_count_delta > 0 ||
                                          candidate->distance_delta > 0 ||
                                          candidate->objective.false_positive_vote_excess != before_objective->false_positive_vote_excess ||
                                          candidate->objective.target_zero_safety_min != before_objective->target_zero_safety_min;
        bool false_negative_quota_cost_reduces = candidate->vote_deficit_delta > 0 ||
                                                 candidate->false_negative_count_delta > 0;
        bool best_completion_cost_reduces =
                itty_feed_model_weighted_deficit_histogram_cost (&candidate->objective) <
                itty_feed_model_weighted_deficit_histogram_cost (before_objective);

        if (decoded_vote_count_changes ||
            false_negative_quota_cost_reduces ||
            best_completion_cost_reduces)
                return true;

        if (stats) {
                stats->no_effect_candidates++;
                if (candidate->already_satisfied_bits == candidate->condensed_assignments.count)
                        stats->no_effect_candidate_already_satisfied++;
                else if (candidate->final_node != before_objective->selected_node)
                        stats->no_effect_candidate_unselected_node++;
                else if (candidate->bits_needing_flips > 0)
                        stats->no_effect_candidate_no_majority_crossing++;
                else if (!decoded_vote_count_changes)
                        stats->no_effect_candidate_irrelevant_segment++;
                else
                        stats->no_effect_candidate_vote_tied++;
        }

        return false;
}

static bool
itty_feed_model_projected_candidate_is_replay_safe (itty_feed_model_t                                      *model,
                                                    size_t                                                  layer_index,
                                                    itty_bit_string_list_t                                 *layer_input,
                                                    itty_feed_model_projected_repair_candidate_t const     *candidate,
                                                    itty_feed_model_replay_example_t const                 *replay_examples,
                                                    size_t                                                  replay_example_count,
                                                    bool                                                    strict_replay_guard,
                                                    itty_feed_model_refreshed_projected_repair_round_t     *replay_stats_out)
{
        if (replay_example_count == 0)
                return true;

        itty_feed_model_decoder_objective_t *before_objectives = calloc (replay_example_count,
                                                                         sizeof (itty_feed_model_decoder_objective_t));
        itty_bit_string_t **before_folded = calloc (replay_example_count,
                                                    sizeof (itty_bit_string_t *));
        if (!itty_feed_model_measure_replay_examples (model,
                                                      replay_examples,
                                                      replay_example_count,
                                                      before_objectives,
                                                      before_folded)) {
                for (size_t replay_index = 0; replay_index < replay_example_count; replay_index++)
                        if (before_folded[replay_index])
                                itty_bit_string_free (before_folded[replay_index]);
                free (before_folded);
                free (before_objectives);
                return false;
        }

        itty_feed_model_layer_state_snapshot_t *snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                 layer_index);
        size_t condensed_words = model->vocabulary_words << layer_index;
        itty_bit_string_t **targets = calloc (model->nodes_per_layer,
                                             sizeof (itty_bit_string_t *));
        itty_bit_string_t **cares = calloc (model->nodes_per_layer,
                                           sizeof (itty_bit_string_t *));

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_t *current_condensed = itty_feed_model_run_node_condensed (layer_input,
                                                                                           model->masks_by_layer_node[layer_index][node_index]);
                targets[node_index] = itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                                 condensed_words);
                cares[node_index] = itty_feed_model_zero_mask_new (condensed_words);
                itty_bit_string_free (current_condensed);
        }

        for (size_t assignment_index = 0; assignment_index < candidate->condensed_assignments.count; assignment_index++) {
                itty_feed_model_layer_assignment_t *assignment = &candidate->condensed_assignments.items[assignment_index];

                itty_bit_string_set_bit (cares[assignment->layer_node],
                                         assignment->bit_index,
                                         true);
                itty_bit_string_set_bit (targets[assignment->layer_node],
                                         assignment->bit_index,
                                         assignment->value);
        }

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_feed_model_train_stats_t node_stats = { 0 };
                itty_feed_model_train_layer_one_node_with_care (model->masks_by_layer_node[layer_index][node_index],
                                                                layer_input,
                                                                targets[node_index],
                                                                cares[node_index],
                                                                0,
                                                                &node_stats);
        }

        itty_feed_model_refreshed_projected_repair_round_t replay_stats = { 0 };
        bool replay_safe = itty_feed_model_score_replay_after_batch (model,
                                                                     replay_examples,
                                                                     replay_example_count,
                                                                     strict_replay_guard,
                                                                     before_objectives,
                                                                     before_folded,
                                                                     &replay_stats);
        if (replay_stats_out)
                *replay_stats_out = replay_stats;

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_free (targets[node_index]);
                itty_bit_string_free (cares[node_index]);
        }
        free (targets);
        free (cares);

        itty_feed_model_restore_layer_state_snapshot (model,
                                                      layer_index,
                                                      snapshot);

        for (size_t replay_index = 0; replay_index < replay_example_count; replay_index++)
                itty_bit_string_free (before_folded[replay_index]);
        free (before_folded);
        free (before_objectives);

        if (!replay_safe)
                return false;

        return replay_stats.replay_distance_delta >= 0 &&
               replay_stats.replay_false_positive_excess_delta >= 0 &&
               replay_stats.replay_target_one_margin_delta >= 0 &&
               replay_stats.replay_target_zero_safety_delta >= 0;
}

static size_t
itty_feed_model_measure_replay_mask_flip_sensitivity (itty_feed_model_t                                      *model,
                                                      size_t                                                  layer_index,
                                                      itty_bit_string_list_t                                 *layer_input,
                                                      itty_bit_string_t                                      *target,
                                                      itty_feed_model_decoder_objective_t const              *before_objective,
                                                      itty_feed_model_projected_repair_candidate_t const     *candidate,
                                                      itty_feed_model_replay_example_t const                 *replay_examples,
                                                      size_t                                                  replay_example_count,
                                                      itty_feed_model_decoder_objective_t const              *before_objectives,
                                                      itty_bit_string_t                                     **before_folded,
                                                      size_t                                                  false_positive_penalty,
                                                      size_t                                                  false_negative_penalty,
                                                      itty_feed_model_bad_flip_frequency_list_t              *bad_flip_frequencies,
                                                      itty_feed_model_projected_repair_stats_t               *stats)
{
        if (replay_example_count == 0 ||
            candidate->use_residual)
                return 0;

        itty_feed_model_mask_flip_list_t flips = { 0 };
        itty_feed_model_collect_candidate_mask_flips (model,
                                                     layer_index,
                                                     layer_input,
                                                     candidate,
                                                     &flips);

        size_t collateral_cost = 0;
        size_t unsafe_flips = 0;

        for (size_t flip_index = 0; flip_index < flips.count; flip_index++) {
                itty_feed_model_mask_flip_t *flip = &flips.items[flip_index];
                bool unsafe = false;
                bool false_positive = false;
                bool false_negative = false;
                bool weakening = false;
                size_t false_positive_damage = 0;
                itty_bit_string_t *mask = model->masks_by_layer_node[layer_index][flip->layer_node]->bit_strings[flip->input_index];
                bool new_mask_bit = !itty_bit_string_get_bit (mask,
                                                              flip->bit_index);
                size_t flip_cost = itty_feed_model_score_replay_mask_flip (model,
                                                                           layer_index,
                                                                           flip,
                                                                           replay_examples,
                                                                           replay_example_count,
                                                                           before_objectives,
                                                                           before_folded,
                                                                           false_positive_penalty,
                                                                           false_negative_penalty,
                                                                           &unsafe,
                                                                           &false_positive,
                                                                           &false_negative,
                                                                           &weakening,
                                                                           &false_positive_damage);

                if (stats)
                        stats->replay_sensitive_mask_flips++;

                if (false_positive) {
                        if (stats)
                                stats->replay_false_positive_mask_flips++;
                } else if (false_negative) {
                        if (stats)
                                stats->replay_false_negative_mask_flips++;
                } else if (weakening) {
                        if (stats)
                                stats->replay_margin_or_safety_weakening_mask_flips++;
                } else if (stats) {
                        stats->replay_safe_mask_flips++;
                }

                if (unsafe || weakening)
                        unsafe_flips++;
                if (bad_flip_frequencies)
                        itty_feed_model_bad_flip_frequency_list_record (bad_flip_frequencies,
                                                                        layer_index,
                                                                        flip,
                                                                        new_mask_bit,
                                                                        false_positive,
                                                                        false_positive_damage,
                                                                        candidate->decoded_bit);
                collateral_cost += flip_cost;
        }

        if (stats) {
                size_t alternate_flips = 0;
                size_t alternate_unsafe_flips = 0;
                size_t alternate_cost = 0;

                stats->replay_collateral_cost += collateral_cost;
                stats->replay_decomposed_candidates++;
                stats->replay_decomposed_mask_flips += flips.count;
                stats->replay_decomposed_unsafe_mask_flips += unsafe_flips;
                if (unsafe_flips == 1)
                        stats->replay_one_bad_flip_candidates++;
                if (flips.count > 1 && unsafe_flips * 2 >= flips.count)
                        stats->replay_mostly_unsafe_candidates++;
                itty_feed_model_measure_minus_one_bad_flip_candidate (model,
                                                                      layer_index,
                                                                      layer_input,
                                                                      target,
                                                                      candidate,
                                                                      replay_examples,
                                                                      replay_example_count,
                                                                      before_objective,
                                                                      before_objectives,
                                                                      before_folded,
                                                                      stats);

                itty_feed_model_measure_alternate_replay_realization (model,
                                                                      layer_index,
                                                                      layer_input,
                                                                      candidate,
                                                                      replay_examples,
                                                                      replay_example_count,
                                                                      before_objectives,
                                                                      before_folded,
                                                                      false_positive_penalty,
                                                                      false_negative_penalty,
                                                                      &alternate_flips,
                                                                      &alternate_unsafe_flips,
                                                                      &alternate_cost);
                stats->replay_alternate_mask_flips += alternate_flips;
                stats->replay_alternate_unsafe_mask_flips += alternate_unsafe_flips;
                stats->replay_alternate_collateral_cost += alternate_cost;
                if (alternate_cost < collateral_cost ||
                    (alternate_cost == collateral_cost && alternate_unsafe_flips < unsafe_flips))
                        stats->replay_alternate_better_candidates++;
        }

        itty_feed_model_mask_flip_list_clear (&flips);
        return collateral_cost;
}

static void
itty_feed_model_accumulate_replay_unsafe_candidate_stats (itty_feed_model_projected_repair_stats_t                 *stats,
                                                          itty_feed_model_refreshed_projected_repair_round_t const *replay_stats)
{
        if (!stats || !replay_stats)
                return;

        if (replay_stats->replay_distance_delta < 0)
                stats->replay_unsafe_distance_regressions++;
        if (replay_stats->replay_false_positive_excess_delta < 0)
                stats->replay_unsafe_false_positive_excess_regressions++;
        if (replay_stats->replay_target_one_margin_delta < 0)
                stats->replay_unsafe_target_one_margin_regressions++;
        if (replay_stats->replay_target_zero_safety_delta < 0)
                stats->replay_unsafe_target_zero_safety_regressions++;
        stats->replay_unsafe_selected_node_switches += replay_stats->replay_selected_node_switches;
        stats->replay_unsafe_best_decoded_node_switches += replay_stats->replay_best_decoded_node_switches;
        stats->replay_unsafe_transitions.correct_zero_to_false_positive_bits += replay_stats->replay_transitions.correct_zero_to_false_positive_bits;
        stats->replay_unsafe_transitions.correct_one_to_false_negative_bits += replay_stats->replay_transitions.correct_one_to_false_negative_bits;
        stats->replay_unsafe_transitions.false_positive_to_correct_zero_bits += replay_stats->replay_transitions.false_positive_to_correct_zero_bits;
        stats->replay_unsafe_transitions.false_negative_to_correct_one_bits += replay_stats->replay_transitions.false_negative_to_correct_one_bits;
        stats->replay_unsafe_transitions.false_positive_to_false_negative_bits += replay_stats->replay_transitions.false_positive_to_false_negative_bits;
        stats->replay_unsafe_transitions.false_negative_to_false_positive_bits += replay_stats->replay_transitions.false_negative_to_false_positive_bits;
        stats->replay_unsafe_transitions.unchanged_wrong_bits += replay_stats->replay_transitions.unchanged_wrong_bits;
        stats->replay_unsafe_transitions.unchanged_correct_bits += replay_stats->replay_transitions.unchanged_correct_bits;
}

static void
itty_feed_model_accumulate_replay_safe_candidate_usefulness (itty_feed_model_projected_repair_stats_t           *stats,
                                                             itty_feed_model_projected_repair_candidate_t const *candidate,
                                                             itty_feed_model_decoder_objective_t const          *before_objective)
{
        if (!stats || !candidate)
                return;

        if (candidate->distance_delta > 0)
                stats->replay_safe_strict_distance_candidates++;
        else if (candidate->vote_deficit_delta > 0 ||
                 candidate->false_negative_count_delta > 0)
                stats->replay_safe_deficit_candidates++;
        else if (candidate->target_one_margin_delta > 0 ||
                 candidate->blocker_delta > 0)
                stats->replay_safe_frontier_candidates++;
        else if (before_objective &&
                 (candidate->objective.selected_distance != before_objective->selected_distance ||
                  candidate->objective.false_negative_vote_deficit != before_objective->false_negative_vote_deficit ||
                  candidate->objective.false_positive_vote_excess != before_objective->false_positive_vote_excess ||
                  candidate->objective.target_one_margin != before_objective->target_one_margin ||
                  candidate->objective.target_zero_safety_min != before_objective->target_zero_safety_min))
                stats->replay_safe_vote_movement_candidates++;
        else if (candidate->final_node != candidate->objective.selected_node)
                stats->replay_safe_irrelevant_candidates++;
        else
                stats->replay_safe_noop_candidates++;
}

static void
itty_feed_model_apply_or_residual_assignments (itty_feed_model_t                            *model,
                                               size_t                                        layer_index,
                                               itty_bit_string_list_t                       *input,
                                               itty_feed_model_layer_assignment_list_t const *condensed_assignments,
                                               size_t                                       *enable_flips,
                                               size_t                                       *mask_flips)
{
        *enable_flips = 0;
        *mask_flips = 0;

        for (size_t assignment_index = 0; assignment_index < condensed_assignments->count; assignment_index++) {
                itty_feed_model_layer_assignment_t const *assignment = &condensed_assignments->items[assignment_index];
                itty_bit_string_t *enable = model->residual_enable_by_layer_node[layer_index][assignment->layer_node];
                itty_bit_string_t *mask = model->residual_mask_by_layer_node[layer_index][assignment->layer_node];
                itty_bit_string_t *skip_input = itty_bit_string_list_fetch (input,
                                                                            assignment->layer_node % input->count);
                itty_bit_string_t *condensed = itty_feed_model_run_node_condensed (input,
                                                                                   model->masks_by_layer_node[layer_index][assignment->layer_node]);
                bool condensed_bit = itty_bit_string_get_bit (condensed,
                                                              assignment->bit_index);
                bool skip_bit = itty_bit_string_get_bit (skip_input,
                                                         assignment->bit_index) !=
                                itty_bit_string_get_bit (mask,
                                                         assignment->bit_index);

                itty_bit_string_free (condensed);
                if (condensed_bit)
                        continue;

                if (!skip_bit) {
                        itty_feed_model_flip_mask_bit (mask,
                                                       assignment->bit_index);
                        (*mask_flips)++;
                }
                if (!itty_bit_string_get_bit (enable,
                                              assignment->bit_index)) {
                        itty_feed_model_set_mutable_bit (enable,
                                                         assignment->bit_index,
                                                         true);
                        (*enable_flips)++;
                }
        }
}

static void
itty_feed_model_measure_previous_layer_projection (itty_feed_model_t                            *model,
                                                   size_t                                        layer_index,
                                                   itty_bit_string_list_t                       *previous_outputs,
                                                   itty_bit_string_t                            *target,
                                                   itty_feed_model_decoder_objective_t          *before_objective,
                                                   itty_feed_model_layer_assignment_list_t const *layer_repairs,
                                                   itty_feed_model_projected_repair_stats_t     *stats)
{
        if (layer_index == 0)
                return;

        size_t previous_layer = layer_index - 1;
        itty_feed_model_layer_assignment_list_t previous_output_assignments = { 0 };
        itty_feed_model_layer_assignment_list_t previous_condensed_assignments = { 0 };

        for (size_t repair_index = 0; repair_index < layer_repairs->count; repair_index++) {
                itty_feed_model_layer_assignment_t const *repair = &layer_repairs->items[repair_index];

                if (!itty_feed_model_project_layer_repair_to_previous_outputs (model,
                                                                               layer_index,
                                                                               previous_outputs,
                                                                               repair->layer_node,
                                                                               repair->bit_index,
                                                                               repair->value,
                                                                               &previous_output_assignments)) {
                        itty_feed_model_layer_assignment_list_clear (&previous_output_assignments);
                        return;
                }
        }

        if (!itty_feed_model_make_condensed_assignments_from_outputs (model,
                                                                      previous_layer,
                                                                      &previous_output_assignments,
                                                                      &previous_condensed_assignments)) {
                itty_feed_model_layer_assignment_list_clear (&previous_condensed_assignments);
                itty_feed_model_layer_assignment_list_clear (&previous_output_assignments);
                return;
        }

        itty_feed_model_decoder_objective_t candidate_objective = { 0 };
        itty_bit_string_list_t *candidate_outputs = itty_feed_model_make_condensed_realistic_outputs (model,
                                                                                                      previous_layer,
                                                                                                      previous_outputs,
                                                                                                      &previous_condensed_assignments);
        if (itty_feed_model_evaluate_suffix_decoder_objective (model,
                                                               candidate_outputs,
                                                               previous_layer,
                                                               target,
                                                               &candidate_objective)) {
                stats->previous_layer_projected_blocks++;
                if (candidate_objective.selected_distance < before_objective->selected_distance)
                        stats->previous_layer_strict_distance_helpful_blocks++;
                else if (candidate_objective.selected_distance > before_objective->selected_distance) {
                        stats->previous_layer_harmful_blocks++;
                        stats->previous_layer_harmful_distance_blocks++;
                        itty_bit_string_t *before_folded = NULL;
                        itty_bit_string_t *after_folded = NULL;
                        if (itty_feed_model_fold_suffix_selected_output (model,
                                                                         previous_outputs,
                                                                         previous_layer,
                                                                         target,
                                                                         &before_folded) &&
                            itty_feed_model_fold_suffix_selected_output (model,
                                                                         candidate_outputs,
                                                                         previous_layer,
                                                                         target,
                                                                         &after_folded)) {
                                itty_feed_model_measure_decoded_bit_transitions (before_folded,
                                                                                after_folded,
                                                                                target,
                                                                                stats);
                        }
                        if (before_folded)
                                itty_bit_string_free (before_folded);
                        if (after_folded)
                                itty_bit_string_free (after_folded);
                } else if (candidate_objective.false_negative_blocker_bits < before_objective->false_negative_blocker_bits) {
                        stats->previous_layer_blocker_helpful_blocks++;
                } else if (itty_feed_model_decoder_objective_accepts (before_objective,
                                                                      &candidate_objective)) {
                        stats->previous_layer_objective_helpful_blocks++;
                } else {
                        stats->previous_layer_neutral_blocks++;
                        if (candidate_objective.false_negative_blocker_bits > before_objective->false_negative_blocker_bits)
                                stats->previous_layer_harmful_blocker_blocks++;
                        else if (candidate_objective.nearest_wrong_margin < before_objective->nearest_wrong_margin)
                                stats->previous_layer_harmful_margin_blocks++;
                        else if (candidate_objective.zero_veto_safety_bits < before_objective->zero_veto_safety_bits)
                                stats->previous_layer_harmful_safety_blocks++;
                }

                itty_feed_model_decoder_objective_t pinned_before_objective = { 0 };
                itty_feed_model_decoder_objective_t pinned_candidate_objective = { 0 };
                if (itty_feed_model_evaluate_suffix_decoder_objective_for_node (model,
                                                                                previous_outputs,
                                                                                previous_layer,
                                                                                target,
                                                                                before_objective->selected_node,
                                                                                &pinned_before_objective) &&
                    itty_feed_model_evaluate_suffix_decoder_objective_for_node (model,
                                                                                candidate_outputs,
                                                                                previous_layer,
                                                                                target,
                                                                                before_objective->selected_node,
                                                                                &pinned_candidate_objective)) {
                        stats->previous_layer_pinned_projected_blocks++;
                        if (pinned_candidate_objective.selected_distance < pinned_before_objective.selected_distance)
                                stats->previous_layer_pinned_strict_distance_helpful_blocks++;
                        else if (pinned_candidate_objective.selected_distance > pinned_before_objective.selected_distance)
                                stats->previous_layer_pinned_harmful_blocks++;
                        else if (pinned_candidate_objective.false_negative_blocker_bits < pinned_before_objective.false_negative_blocker_bits)
                                stats->previous_layer_pinned_blocker_helpful_blocks++;
                        else if (itty_feed_model_decoder_objective_accepts (&pinned_before_objective,
                                                                            &pinned_candidate_objective))
                                stats->previous_layer_pinned_objective_helpful_blocks++;
                        else
                                stats->previous_layer_pinned_neutral_blocks++;
                }
        }

        itty_bit_string_list_free (candidate_outputs);
        itty_feed_model_layer_assignment_list_clear (&previous_condensed_assignments);
        itty_feed_model_layer_assignment_list_clear (&previous_output_assignments);
}

static itty_bit_string_t *
itty_feed_model_make_decoder_tri_state_target_for_final_node (itty_feed_model_t      *model,
                                                              itty_bit_string_list_t *layer_input,
                                                              size_t                  node_index,
                                                              itty_bit_string_t      *target)
{
        size_t final_layer = model->number_of_layers - 1;
        size_t condensed_words = model->vocabulary_words << final_layer;
        size_t condensed_bit_capacity = condensed_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t output_bit_capacity = condensed_bit_capacity * 2;
        size_t target_bit_capacity = itty_bit_string_get_number_of_words (target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        itty_bit_string_t *current_condensed = itty_feed_model_run_node_condensed (layer_input,
                                                                                   model->masks_by_layer_node[final_layer][node_index]);
        itty_bit_string_t *node_target = itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                                    condensed_words);
        itty_bit_string_t *current_output = itty_feed_model_expand_condensed_output (current_condensed,
                                                                                     model->rotations_by_layer[final_layer]);
        itty_bit_string_t *folded_output = NULL;

        if (!itty_feed_model_fold_activation_to_target_width (model,
                                                              current_output,
                                                              target,
                                                              node_index,
                                                              &folded_output)) {
                itty_bit_string_free (current_output);
                itty_bit_string_free (current_condensed);
                itty_bit_string_free (node_target);
                return NULL;
        }

        for (size_t target_bit_index = 0; target_bit_index < target_bit_capacity; target_bit_index++) {
                bool target_bit = itty_bit_string_get_bit (target,
                                                           target_bit_index);

                if (target_bit) {
                        for (size_t output_bit_index = target_bit_index;
                             output_bit_index < output_bit_capacity;
                             output_bit_index += target_bit_capacity) {
                                size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                                            final_layer,
                                                                                                            output_bit_index);
                                itty_bit_string_set_bit (node_target,
                                                         condensed_bit_index,
                                                         true);
                        }
                } else if (itty_bit_string_get_bit (folded_output, target_bit_index)) {
                        size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                                    final_layer,
                                                                                                    target_bit_index);
                        itty_bit_string_set_bit (node_target,
                                                 condensed_bit_index,
                                                 false);
                }
        }

        node_target->pop_count_computed = false;

        if (folded_output != current_output)
                itty_bit_string_free (folded_output);
        itty_bit_string_free (current_output);
        itty_bit_string_free (current_condensed);

        return node_target;
}

static bool
itty_feed_model_collect_final_repairs (itty_feed_model_t                  *model,
                                       itty_bit_string_list_t             *final_layer_input,
                                       itty_bit_string_t                  *target,
                                       itty_feed_model_final_repair_list_t *repairs)
{
        size_t final_layer = model->number_of_layers - 1;
        itty_bit_string_list_t *final_layer_outputs = itty_feed_model_run_layer (model,
                                                                                  final_layer,
                                                                                  final_layer_input);
        itty_feed_model_decoder_objective_t before_objective = { 0 };
        if (!itty_feed_model_evaluate_decoder_objective (model,
                                                         final_layer_outputs,
                                                         target,
                                                         &before_objective)) {
                itty_bit_string_list_free (final_layer_outputs);
                return false;
        }

        size_t condensed_words = model->vocabulary_words << final_layer;
        size_t bit_capacity = condensed_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        itty_bit_string_t *expanded_target_raw = itty_feed_model_expand_target_for_layer (target,
                                                                                          final_layer);
        itty_bit_string_t *expanded_target = itty_feed_model_bit_string_clone_to_words (expanded_target_raw,
                                                                                         condensed_words);
        itty_bit_string_free (expanded_target_raw);

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_t *current_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                           model->masks_by_layer_node[final_layer][node_index]);
                itty_bit_string_t *working_condensed = itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                                                  condensed_words);
                size_t target_bit_capacity = itty_bit_string_get_number_of_words (target) *
                                             ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
                itty_feed_model_decoder_objective_t current_objective = before_objective;

                if (model->decoder != ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE) {
                        for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                                bool target_bit = itty_bit_string_get_bit (expanded_target,
                                                                           bit_index);

                                if (itty_bit_string_get_bit (working_condensed, bit_index) == target_bit)
                                        continue;

                                itty_bit_string_t *candidate_condensed = itty_feed_model_bit_string_clone_to_words (working_condensed,
                                                                                                                     condensed_words);
                                itty_bit_string_set_bit (candidate_condensed,
                                                         bit_index,
                                                         target_bit);
                                candidate_condensed->pop_count_computed = false;

                                itty_bit_string_t *candidate_output = itty_feed_model_expand_condensed_output (candidate_condensed,
                                                                                                               model->rotations_by_layer[final_layer]);
                                itty_bit_string_list_t *candidate_outputs = itty_feed_model_bit_string_list_clone (final_layer_outputs);
                                itty_bit_string_free (candidate_outputs->bit_strings[node_index]);
                                candidate_outputs->bit_strings[node_index] = candidate_output;

                                itty_feed_model_decoder_objective_t candidate_objective = { 0 };
                                if (itty_feed_model_evaluate_decoder_objective (model,
                                                                                candidate_outputs,
                                                                                target,
                                                                                &candidate_objective) &&
                                    itty_feed_model_decoder_objective_accepts (&current_objective,
                                                                               &candidate_objective)) {
                                        itty_feed_model_final_repair_list_append (repairs,
                                                                                 node_index,
                                                                                 bit_index,
                                                                                 bit_index,
                                                                                 bit_index,
                                                                                 1,
                                                                                 target_bit);
                                        itty_bit_string_set_bit (working_condensed,
                                                                 bit_index,
                                                                 target_bit);
                                        working_condensed->pop_count_computed = false;
                                        current_objective = candidate_objective;
                                }

                                itty_bit_string_list_free (candidate_outputs);
                                itty_bit_string_free (candidate_condensed);
                        }
                }

                itty_bit_string_t *working_output = itty_feed_model_expand_condensed_output (working_condensed,
                                                                                             model->rotations_by_layer[final_layer]);
                itty_bit_string_list_t *working_outputs = itty_feed_model_bit_string_list_clone (final_layer_outputs);
                itty_bit_string_free (working_outputs->bit_strings[node_index]);
                working_outputs->bit_strings[node_index] = working_output;
                itty_feed_model_output_evaluation_t working_evaluation = { 0 };
                if (itty_feed_model_evaluate_output (model,
                                                     working_outputs,
                                                     target,
                                                     &working_evaluation)) {
                        for (size_t target_bit_index = 0; target_bit_index < target_bit_capacity; target_bit_index++) {
                                bool target_bit = itty_bit_string_get_bit (target,
                                                                           target_bit_index);
                                bool folded_bit = itty_bit_string_get_bit (working_evaluation.folded_activation,
                                                                           target_bit_index);

                                if (folded_bit == target_bit)
                                        continue;

                                itty_bit_string_t *candidate_condensed = itty_feed_model_bit_string_clone_to_words (working_condensed,
                                                                                                                     condensed_words);
                                bool changed = itty_feed_model_final_layer_apply_decoder_block (model,
                                                                                                target_bit_index,
                                                                                                target_bit,
                                                                                                candidate_condensed);
                                if (!changed) {
                                        itty_bit_string_free (candidate_condensed);
                                        continue;
                                }

                                candidate_condensed->pop_count_computed = false;

                                itty_bit_string_t *candidate_output = itty_feed_model_expand_condensed_output (candidate_condensed,
                                                                                                               model->rotations_by_layer[final_layer]);
                                itty_bit_string_list_t *candidate_outputs = itty_feed_model_bit_string_list_clone (final_layer_outputs);
                                itty_bit_string_free (candidate_outputs->bit_strings[node_index]);
                                candidate_outputs->bit_strings[node_index] = candidate_output;

                                itty_feed_model_decoder_objective_t candidate_objective = { 0 };
                                if (itty_feed_model_evaluate_decoder_objective (model,
                                                                                candidate_outputs,
                                                                                target,
                                                                                &candidate_objective) &&
                                    itty_feed_model_decoder_objective_accepts (&current_objective,
                                                                               &candidate_objective)) {
                                        size_t quota_size = 0;

                                        for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                                                bool candidate_bit = itty_bit_string_get_bit (candidate_condensed,
                                                                                              bit_index);
                                                if (itty_bit_string_get_bit (working_condensed, bit_index) != candidate_bit)
                                                        quota_size++;
                                        }

                                        for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                                                bool candidate_bit = itty_bit_string_get_bit (candidate_condensed,
                                                                                              bit_index);
                                                if (itty_bit_string_get_bit (working_condensed, bit_index) == candidate_bit)
                                                        continue;

                                                itty_feed_model_final_repair_list_append (repairs,
                                                                                         node_index,
                                                                                         bit_index,
                                                                                         bit_index,
                                                                                         target_bit_index,
                                                                                         quota_size,
                                                                                         candidate_bit);
                                        }

                                        itty_bit_string_free (working_condensed);
                                        working_condensed = itty_feed_model_bit_string_clone_to_words (candidate_condensed,
                                                                                                      condensed_words);
                                        current_objective = candidate_objective;
                                }

                                itty_bit_string_list_free (candidate_outputs);
                                itty_bit_string_free (candidate_condensed);
                        }
                        itty_bit_string_free (working_evaluation.folded_activation);
                }

                itty_bit_string_list_free (working_outputs);
                itty_bit_string_free (working_condensed);
                itty_bit_string_free (current_condensed);
        }

        itty_bit_string_free (expanded_target);
        itty_bit_string_list_free (final_layer_outputs);

        return true;
}

static itty_bit_string_t *
itty_feed_model_segment_condense_desired_output_for_layer (itty_feed_model_t *model,
                                                           itty_bit_string_t *desired_output,
                                                           size_t             layer_index)
{
        size_t *set_votes = NULL;
        size_t layer_words = 0;
        size_t layer_bit_capacity = 0;
        size_t votes_per_bit = 0;

        if (!itty_feed_model_count_segment_votes_for_layer (model,
                                                            desired_output,
                                                            layer_index,
                                                            &set_votes,
                                                            &layer_words,
                                                            &layer_bit_capacity,
                                                            &votes_per_bit))
                return NULL;

        itty_bit_string_t *condensed = itty_feed_model_make_majority_target_from_votes (set_votes,
                                                                                        layer_words,
                                                                                        layer_bit_capacity,
                                                                                        votes_per_bit);
        free (set_votes);

        return condensed;
}

static bool
itty_feed_model_count_segment_votes_for_layer (itty_feed_model_t  *model,
                                               itty_bit_string_t  *desired_output,
                                               size_t              layer_index,
                                               size_t            **set_votes,
                                               size_t             *layer_words,
                                               size_t             *layer_bit_capacity,
                                               size_t             *votes_per_bit)
{
        size_t desired_words = itty_bit_string_get_number_of_words (desired_output);
        *layer_words = model->vocabulary_words << layer_index;

        if (*layer_words == 0 || desired_words == 0 || desired_words % *layer_words != 0)
                return false;

        *layer_bit_capacity = *layer_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t desired_bit_capacity = desired_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        *votes_per_bit = desired_bit_capacity / *layer_bit_capacity;
        *set_votes = calloc (*layer_bit_capacity, sizeof (size_t));

        for (size_t bit_index = 0; bit_index < desired_bit_capacity; bit_index++) {
                if (!itty_bit_string_get_bit (desired_output, bit_index))
                        continue;

                size_t source_bit_index = itty_feed_model_trace_expanded_bit_to_layer (model,
                                                                                       layer_index,
                                                                                       bit_index);
                (*set_votes)[source_bit_index]++;
        }

        return true;
}

static itty_bit_string_t *
itty_feed_model_make_majority_target_from_votes (size_t const *set_votes,
                                                 size_t        layer_words,
                                                 size_t        layer_bit_capacity,
                                                 size_t        votes_per_bit)
{
        size_t majority_threshold = votes_per_bit / 2 + 1;
        itty_bit_string_t *condensed = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        condensed->number_of_words = layer_words;
        condensed->words = calloc (layer_words, ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);

        for (size_t bit_index = 0; bit_index < layer_bit_capacity; bit_index++) {
                if (set_votes[bit_index] >= majority_threshold)
                        itty_bit_string_set_bit (condensed, bit_index, true);
        }

        return condensed;
}

static bool
itty_feed_model_count_segment_partition_votes_for_layer (itty_feed_model_t  *model,
                                                         itty_bit_string_t  *desired_output,
                                                         size_t              layer_index,
                                                         size_t              partition_count,
                                                         size_t            **set_votes_by_partition,
                                                         size_t            **vote_counts_by_partition,
                                                         size_t             *layer_words,
                                                         size_t             *layer_bit_capacity)
{
        size_t desired_words = itty_bit_string_get_number_of_words (desired_output);
        *layer_words = model->vocabulary_words << layer_index;

        if (*layer_words == 0 || desired_words == 0 || desired_words % *layer_words != 0 || partition_count == 0)
                return false;

        *layer_bit_capacity = *layer_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t desired_bit_capacity = desired_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t *seen_votes = calloc (*layer_bit_capacity, sizeof (size_t));
        *set_votes_by_partition = calloc (partition_count * *layer_bit_capacity, sizeof (size_t));
        *vote_counts_by_partition = calloc (partition_count * *layer_bit_capacity, sizeof (size_t));

        for (size_t bit_index = 0; bit_index < desired_bit_capacity; bit_index++) {
                size_t source_bit_index = itty_feed_model_trace_expanded_bit_to_layer (model,
                                                                                       layer_index,
                                                                                       bit_index);
                size_t partition_index = seen_votes[source_bit_index] % partition_count;
                size_t partition_offset = partition_index * *layer_bit_capacity + source_bit_index;

                seen_votes[source_bit_index]++;
                (*vote_counts_by_partition)[partition_offset]++;

                if (itty_bit_string_get_bit (desired_output, bit_index))
                        (*set_votes_by_partition)[partition_offset]++;
        }

        free (seen_votes);

        return true;
}

static itty_bit_string_t *
itty_feed_model_make_partition_target_from_votes (size_t const *set_votes,
                                                  size_t const *vote_counts,
                                                  size_t        layer_words,
                                                  size_t        layer_bit_capacity)
{
        itty_bit_string_t *target = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        target->number_of_words = layer_words;
        target->words = calloc (layer_words, ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);

        for (size_t bit_index = 0; bit_index < layer_bit_capacity; bit_index++) {
                size_t votes = vote_counts[bit_index];
                size_t majority_threshold = votes / 2 + 1;

                if (votes > 0 && set_votes[bit_index] >= majority_threshold)
                        itty_bit_string_set_bit (target, bit_index, true);
        }

        return target;
}

static itty_bit_string_t *
itty_feed_model_make_downstream_mask_aware_target_for_node (itty_feed_model_t      *model,
                                                            itty_bit_string_list_t *layer_input,
                                                            size_t                  layer_index,
                                                            size_t                  node_index,
                                                            itty_bit_string_t      *next_desired_condensed,
                                                            itty_feed_model_downstream_request_mode_t request_mode)
{
        size_t layer_words = model->vocabulary_words << layer_index;
        size_t layer_bit_capacity = layer_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t next_layer_words = model->vocabulary_words << (layer_index + 1);
        size_t next_layer_bit_capacity = next_layer_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t *set_request_weights = calloc (layer_bit_capacity, sizeof (size_t));
        size_t *unset_request_weights = calloc (layer_bit_capacity, sizeof (size_t));
        itty_bit_string_list_t *current_layer_output = itty_feed_model_run_layer (model,
                                                                                  layer_index,
                                                                                  layer_input);
        itty_bit_string_t *current_condensed = itty_feed_model_run_node_condensed (layer_input,
                                                                                   model->masks_by_layer_node[layer_index][node_index]);
        itty_bit_string_t *target = itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                               layer_words);

        if (request_mode == ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_EQUAL) {
                size_t *set_requests = calloc (layer_bit_capacity, sizeof (size_t));

                for (size_t downstream_node_index = 0; downstream_node_index < model->nodes_per_layer; downstream_node_index++) {
                        itty_bit_string_t *outgoing_mask = model->masks_by_layer_node[layer_index + 1][downstream_node_index]->bit_strings[node_index];
                        itty_bit_string_t *desired_modulated_output = itty_bit_string_exclusive_or (next_desired_condensed,
                                                                                                    outgoing_mask);
                        itty_bit_string_t *desired_condensed = itty_feed_model_reduce_desired_output_for_layer (desired_modulated_output,
                                                                                                                model->rotations_by_layer[layer_index]);

                        for (size_t bit_index = 0; bit_index < layer_bit_capacity; bit_index++) {
                                if (itty_bit_string_get_bit (desired_condensed, bit_index))
                                        set_requests[bit_index]++;
                        }

                        itty_bit_string_free (desired_condensed);
                        itty_bit_string_free (desired_modulated_output);
                }

                size_t majority_threshold = model->nodes_per_layer / 2 + 1;
                for (size_t bit_index = 0; bit_index < layer_bit_capacity; bit_index++) {
                        if (set_requests[bit_index] >= majority_threshold)
                                itty_bit_string_set_bit (target, bit_index, true);
                        else if (model->nodes_per_layer - set_requests[bit_index] >= majority_threshold)
                                itty_bit_string_set_bit (target, bit_index, false);
                }

                target->pop_count_computed = false;

                free (set_requests);
                free (set_request_weights);
                free (unset_request_weights);
                itty_bit_string_list_free (current_layer_output);
                itty_bit_string_free (current_condensed);

                return target;
        }

        size_t downstream_threshold = current_layer_output->count / 2 + 1;

        for (size_t downstream_node_index = 0; downstream_node_index < model->nodes_per_layer; downstream_node_index++) {
                itty_bit_string_t *outgoing_mask = model->masks_by_layer_node[layer_index + 1][downstream_node_index]->bit_strings[node_index];
                itty_bit_string_list_t *downstream_masks = model->masks_by_layer_node[layer_index + 1][downstream_node_index];

                for (size_t downstream_bit_index = 0; downstream_bit_index < next_layer_bit_capacity; downstream_bit_index++) {
                        bool desired_bit = itty_bit_string_get_bit (next_desired_condensed,
                                                                    downstream_bit_index);
                        size_t request_weight = 1;

                        if (request_mode != ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_EQUAL) {
                                size_t vote_count = itty_feed_model_count_votes (current_layer_output,
                                                                                 downstream_masks,
                                                                                 downstream_bit_index);

                                request_weight = 0;

                                if (desired_bit) {
                                        if (vote_count < downstream_threshold)
                                                request_weight = request_mode == ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_WEIGHTED ?
                                                                 downstream_threshold - vote_count + 2 :
                                                                 1;
                                        else if (vote_count == downstream_threshold)
                                                request_weight = 1;
                                } else {
                                        size_t max_votes_for_zero = downstream_threshold - 1;

                                        if (vote_count > max_votes_for_zero)
                                                request_weight = request_mode == ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_WEIGHTED ?
                                                                 vote_count - max_votes_for_zero + 2 :
                                                                 1;
                                        else if (vote_count == max_votes_for_zero)
                                                request_weight = 1;
                                }
                        }

                        if (request_weight == 0)
                                continue;

                        bool requested_output_bit = desired_bit != itty_bit_string_get_bit (outgoing_mask,
                                                                                            downstream_bit_index);
                        size_t target_bit_index = downstream_bit_index;
                        if (target_bit_index >= layer_bit_capacity) {
                                size_t rotation = model->rotations_by_layer[layer_index] % layer_bit_capacity;

                                target_bit_index -= layer_bit_capacity;
                                target_bit_index = (target_bit_index + layer_bit_capacity - rotation) %
                                                   layer_bit_capacity;
                        }

                        if (requested_output_bit)
                                set_request_weights[target_bit_index] += request_weight;
                        else
                                unset_request_weights[target_bit_index] += request_weight;
                }
        }

        for (size_t bit_index = 0; bit_index < layer_bit_capacity; bit_index++) {
                if (set_request_weights[bit_index] > unset_request_weights[bit_index])
                        itty_bit_string_set_bit (target, bit_index, true);
                else if (unset_request_weights[bit_index] > set_request_weights[bit_index])
                        itty_bit_string_set_bit (target, bit_index, false);
        }

        target->pop_count_computed = false;

        free (set_request_weights);
        free (unset_request_weights);
        itty_bit_string_list_free (current_layer_output);
        itty_bit_string_free (current_condensed);

        return target;
}

static itty_bit_string_t *
itty_feed_model_make_disagreement_target_for_node (itty_feed_model_t      *model,
                                                   itty_bit_string_list_t *layer_input,
                                                   size_t                  layer_index,
                                                   size_t                  node_index,
                                                   size_t const           *set_votes,
                                                   size_t                  layer_words,
                                                   size_t                  layer_bit_capacity,
                                                   size_t                  votes_per_bit,
                                                   itty_bit_string_t      *majority_target)
{
        if (node_index == 0)
                return itty_feed_model_bit_string_clone (majority_target);

        itty_bit_string_t *actual_condensed = itty_feed_model_run_node_condensed (layer_input,
                                                                                  model->masks_by_layer_node[layer_index][node_index]);
        itty_bit_string_t *target = itty_feed_model_bit_string_clone_to_words (actual_condensed,
                                                                               layer_words);
        size_t majority_threshold = votes_per_bit / 2 + 1;

        for (size_t bit_index = 0; bit_index < layer_bit_capacity; bit_index++) {
                if (set_votes[bit_index] == 0 || set_votes[bit_index] == votes_per_bit)
                        continue;

                bool majority_bit = set_votes[bit_index] >= majority_threshold;
                itty_bit_string_set_bit (target, bit_index, !majority_bit);
        }

        target->pop_count_computed = false;
        itty_bit_string_free (actual_condensed);

        return target;
}

static itty_bit_string_t *
itty_feed_model_make_node_target_for_layer (itty_feed_model_t                              *model,
                                            itty_bit_string_list_t                         *layer_input,
                                            itty_bit_string_t                              *target,
                                            itty_bit_string_t                              *desired_output,
                                            itty_feed_model_suffix_oracle_options_t const   *options)
{
        size_t layer_index = options->layer_index;
        size_t node_index = options->node_index;
        itty_feed_model_backward_node_target_t node_target = options->backward_node_target;
        itty_bit_string_t *desired_condensed = itty_feed_model_segment_condense_desired_output_for_layer (model,
                                                                                                          desired_output,
                                                                                                          layer_index);

        if (!desired_condensed)
                return NULL;

        if (node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SAME)
                return desired_condensed;

        if (node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DECODER_TRI_STATE) {
                itty_bit_string_t *tri_state_target = NULL;

                if (layer_index + 1 == model->number_of_layers)
                        tri_state_target = itty_feed_model_make_decoder_tri_state_target_for_final_node (model,
                                                                                                         layer_input,
                                                                                                         node_index,
                                                                                                         target);

                itty_bit_string_free (desired_condensed);
                return tri_state_target;
        }

        if (node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DISAGREEMENT) {
                size_t *set_votes = NULL;
                size_t layer_words = 0;
                size_t layer_bit_capacity = 0;
                size_t votes_per_bit = 0;
                itty_bit_string_t *target = NULL;

                if (itty_feed_model_count_segment_votes_for_layer (model,
                                                                   desired_output,
                                                                   layer_index,
                                                                   &set_votes,
                                                                   &layer_words,
                                                                   &layer_bit_capacity,
                                                                   &votes_per_bit))
                        target = itty_feed_model_make_disagreement_target_for_node (model,
                                                                                    layer_input,
                                                                                    layer_index,
                                                                                    node_index,
                                                                                    set_votes,
                                                                                    layer_words,
                                                                                    layer_bit_capacity,
                                                                                    votes_per_bit,
                                                                                    desired_condensed);

                free (set_votes);
                itty_bit_string_free (desired_condensed);
                return target;
        }

        if (node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SEGMENT_PARTITION) {
                size_t *partition_set_votes = NULL;
                size_t *partition_vote_counts = NULL;
                size_t layer_words = 0;
                size_t layer_bit_capacity = 0;
                itty_bit_string_t *target = NULL;

                if (itty_feed_model_count_segment_partition_votes_for_layer (model,
                                                                             desired_output,
                                                                             layer_index,
                                                                             model->nodes_per_layer,
                                                                             &partition_set_votes,
                                                                             &partition_vote_counts,
                                                                             &layer_words,
                                                                             &layer_bit_capacity)) {
                        size_t partition_offset = node_index * layer_bit_capacity;

                        target = itty_feed_model_make_partition_target_from_votes (partition_set_votes + partition_offset,
                                                                                   partition_vote_counts + partition_offset,
                                                                                   layer_words,
                                                                                   layer_bit_capacity);
                }

                free (partition_set_votes);
                free (partition_vote_counts);
                itty_bit_string_free (desired_condensed);
                return target;
        }

        if (node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE ||
            node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_WEIGHTED ||
            node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_GATED) {
                itty_bit_string_t *target = NULL;

                if (layer_index + 1 < model->number_of_layers) {
                        itty_feed_model_downstream_request_mode_t request_mode = ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_EQUAL;
                        itty_bit_string_t *next_desired_condensed = itty_feed_model_segment_condense_desired_output_for_layer (model,
                                                                                                                               desired_output,
                                                                                                                               layer_index + 1);
                        if (node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_WEIGHTED)
                                request_mode = ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_WEIGHTED;
                        else if (node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_GATED)
                                request_mode = ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_GATED;

                        target = itty_feed_model_make_downstream_mask_aware_target_for_node (model,
                                                                                             layer_input,
                                                                                             layer_index,
                                                                                             node_index,
                                                                                             next_desired_condensed,
                                                                                             request_mode);
                        itty_bit_string_free (next_desired_condensed);
                } else {
                        target = itty_feed_model_bit_string_clone (desired_condensed);
                }

                itty_bit_string_free (desired_condensed);
                return target;
        }

        itty_bit_string_free (desired_condensed);
        return NULL;
}

static itty_bit_string_list_t *
itty_feed_model_make_desired_layer_input (itty_bit_string_t      *desired_condensed,
                                          itty_bit_string_list_t *masks)
{
        itty_bit_string_list_t *desired_input = itty_bit_string_list_new ();

        for (size_t mask_index = 0; mask_index < masks->count; mask_index++) {
                itty_bit_string_t *desired_input_bit_string = itty_bit_string_exclusive_or (desired_condensed,
                                                                                            masks->bit_strings[mask_index]);
                itty_bit_string_list_append (desired_input, desired_input_bit_string);
        }

        return desired_input;
}

static itty_bit_string_t *
itty_feed_model_run_node_condensed (itty_bit_string_list_t *input,
                                    itty_bit_string_list_t *masks)
{
        return itty_feed_model_run_node_condensed_with_threshold (input,
                                                                  masks,
                                                                  0);
}

static itty_bit_string_t *
itty_feed_model_run_node_condensed_with_threshold (itty_bit_string_list_t *input,
                                                   itty_bit_string_list_t *masks,
                                                   size_t                  threshold_override)
{
        itty_bit_string_list_t *modulated_inputs = itty_bit_string_list_exclusive_or (input,
                                                                                      masks);
        itty_bit_string_t *condensed_output = itty_feed_model_condense_with_threshold (modulated_inputs,
                                                                                       threshold_override);
        itty_bit_string_list_free (modulated_inputs);

        if (!condensed_output)
                condensed_output = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);

        return condensed_output;
}

static void
itty_feed_model_accumulate_train_stats (itty_feed_model_train_stats_t       *stats,
                                        itty_feed_model_train_stats_t const *current_layer_stats)
{
        if (!stats)
                return;

        stats->flips += current_layer_stats->flips;
        stats->candidate_bits += current_layer_stats->candidate_bits;
        if (current_layer_stats->largest_error > stats->largest_error)
                stats->largest_error = current_layer_stats->largest_error;
}

static bool
itty_feed_model_can_train_with_options (itty_feed_model_train_options_t const *options)
{
        if (!options)
                return true;

        if (options->budget_policy != ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST)
                return false;

        if (options->backward_fold != ITTY_FEED_MODEL_BACKWARD_FOLD_CHAINED_REDUCE &&
            options->backward_fold != ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE)
                return false;

        return options->backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SAME ||
               options->backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DISAGREEMENT ||
               options->backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_PAIRWISE_AND_SEGMENT ||
               options->backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SEGMENT_PARTITION ||
               options->backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE ||
               options->backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_WEIGHTED ||
               options->backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_GATED ||
               options->backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DECODER_TRI_STATE;
}

static itty_feed_model_backward_fold_t
itty_feed_model_get_backward_fold (itty_feed_model_train_options_t const *options)
{
        return options ? options->backward_fold : ITTY_FEED_MODEL_BACKWARD_FOLD_CHAINED_REDUCE;
}

static itty_feed_model_backward_node_target_t
itty_feed_model_get_backward_node_target (itty_feed_model_train_options_t const *options)
{
        return options ? options->backward_node_target : ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SAME;
}

itty_feed_model_t *
itty_feed_model_new (size_t number_of_layers,
                     size_t nodes_per_layer,
                     size_t inputs_per_node,
                     size_t vocabulary_words)
{
        itty_feed_model_t *model = malloc (sizeof (itty_feed_model_t));
        model->number_of_layers = number_of_layers;
        model->nodes_per_layer = nodes_per_layer;
        model->inputs_per_node = inputs_per_node;
        model->vocabulary_words = vocabulary_words;
        model->masks_by_layer_node = calloc (number_of_layers, sizeof (itty_bit_string_list_t **));
        model->input_route_adapters = calloc (nodes_per_layer, sizeof (itty_bit_string_list_t *));
        model->residual_enable_by_layer_node = calloc (number_of_layers, sizeof (itty_bit_string_t **));
        model->residual_mask_by_layer_node = calloc (number_of_layers, sizeof (itty_bit_string_t **));
        model->gray_offset_override_enabled_by_route = calloc (nodes_per_layer, sizeof (bool));
        model->gray_offset_override_payload_start_by_route = calloc (nodes_per_layer, sizeof (size_t));
        model->rotations_by_layer = calloc (number_of_layers, sizeof (size_t));
        model->within_node_condense_threshold_override = 0;
        model->residual_merge_enabled = true;
        model->input_route_adapter_enabled = false;
        model->decoder = ITTY_FEED_MODEL_DECODER_REPEATED_AND_FOLD;

        for (size_t route_index = 0; route_index < nodes_per_layer; route_index++) {
                itty_bit_string_list_t *adapters = itty_bit_string_list_new ();
                for (size_t input_index = 0; input_index < inputs_per_node; input_index++)
                        itty_bit_string_list_append (adapters, itty_feed_model_zero_mask_new (vocabulary_words));
                model->input_route_adapters[route_index] = adapters;
        }

        for (size_t layer_index = 0; layer_index < number_of_layers; layer_index++) {
                size_t layer_words = vocabulary_words << layer_index;
                size_t layer_input_count = itty_feed_model_get_layer_input_count (model,
                                                                                  layer_index);
                model->masks_by_layer_node[layer_index] = calloc (nodes_per_layer, sizeof (itty_bit_string_list_t *));
                model->residual_enable_by_layer_node[layer_index] = calloc (nodes_per_layer, sizeof (itty_bit_string_t *));
                model->residual_mask_by_layer_node[layer_index] = calloc (nodes_per_layer, sizeof (itty_bit_string_t *));

                for (size_t node_index = 0; node_index < nodes_per_layer; node_index++) {
                        itty_bit_string_list_t *masks = itty_bit_string_list_new ();
                        for (size_t input_index = 0; input_index < layer_input_count; input_index++)
                                itty_bit_string_list_append (masks, itty_feed_model_zero_mask_new (layer_words));
                        model->masks_by_layer_node[layer_index][node_index] = masks;
                        model->residual_enable_by_layer_node[layer_index][node_index] = itty_feed_model_zero_mask_new (layer_words);
                        model->residual_mask_by_layer_node[layer_index][node_index] = itty_feed_model_zero_mask_new (layer_words);
                }
        }

        return model;
}

void
itty_feed_model_free (itty_feed_model_t *model)
{
        if (!model)
                return;

        for (size_t layer_index = 0; layer_index < model->number_of_layers; layer_index++) {
                for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                        itty_bit_string_list_free (model->masks_by_layer_node[layer_index][node_index]);
                        itty_bit_string_free (model->residual_enable_by_layer_node[layer_index][node_index]);
                        itty_bit_string_free (model->residual_mask_by_layer_node[layer_index][node_index]);
                }
                free (model->masks_by_layer_node[layer_index]);
                free (model->residual_enable_by_layer_node[layer_index]);
                free (model->residual_mask_by_layer_node[layer_index]);
        }
        for (size_t route_index = 0; route_index < model->nodes_per_layer; route_index++)
                itty_bit_string_list_free (model->input_route_adapters[route_index]);
        free (model->rotations_by_layer);
        free (model->masks_by_layer_node);
        free (model->input_route_adapters);
        free (model->residual_enable_by_layer_node);
        free (model->residual_mask_by_layer_node);
        free (model->gray_offset_override_enabled_by_route);
        free (model->gray_offset_override_payload_start_by_route);
        free (model);
}

void
itty_feed_model_set_layer_rotation (itty_feed_model_t *model,
                                    size_t             layer_index,
                                    size_t             rotation)
{
        if (layer_index < model->number_of_layers)
                model->rotations_by_layer[layer_index] = rotation;
}

void
itty_feed_model_set_decoder (itty_feed_model_t        *model,
                             itty_feed_model_decoder_t decoder)
{
        model->decoder = decoder;
}

void
itty_feed_model_set_within_node_condense_threshold_override (itty_feed_model_t *model,
                                                             size_t             threshold)
{
        if (!model)
                return;

        model->within_node_condense_threshold_override = threshold;
}

void
itty_feed_model_set_residual_merge_enabled (itty_feed_model_t *model,
                                            bool               enabled)
{
        if (!model)
                return;

        model->residual_merge_enabled = enabled;
}

void
itty_feed_model_set_input_route_adapter_enabled (itty_feed_model_t *model,
                                                 bool               enabled)
{
        if (!model)
                return;

        model->input_route_adapter_enabled = enabled;
}

void
itty_feed_model_set_gray_offset_override (itty_feed_model_t *model,
                                          size_t             route_index,
                                          bool               enabled,
                                          size_t             payload_start)
{
        if (!model || route_index >= model->nodes_per_layer)
                return;

        model->gray_offset_override_enabled_by_route[route_index] = enabled;
        model->gray_offset_override_payload_start_by_route[route_index] = payload_start;
}

bool
itty_feed_model_train_input_route_adapter (itty_feed_model_t                     *model,
                                           itty_bit_string_list_t                *input,
                                           itty_bit_string_t                     *target,
                                           size_t                                 route_index,
                                           itty_feed_model_train_options_t const *options,
                                           itty_feed_model_train_stats_t         *stats)
{
        size_t max_flips;
        itty_bit_string_list_t *adapter_masks;
        itty_feed_model_decoder_objective_t current_objective = { 0 };

        if (stats)
                *stats = (itty_feed_model_train_stats_t) { 0 };
        if (!model ||
            !input ||
            !target ||
            route_index >= model->nodes_per_layer ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node)
                return false;

        adapter_masks = model->input_route_adapters[route_index];
        if (!adapter_masks)
                return false;

        if (!itty_feed_model_measure_decoder_objective_for_node (model,
                                                                 input,
                                                                 target,
                                                                 route_index,
                                                                 &current_objective))
                return false;

        max_flips = options ? options->max_flips : 0;
        for (size_t accepted_flips = 0; accepted_flips < max_flips; accepted_flips++) {
                bool found = false;
                size_t best_input_index = 0;
                size_t best_bit_index = 0;
                itty_feed_model_decoder_objective_t best_objective = current_objective;

                for (size_t input_index = 0; input_index < adapter_masks->count; input_index++) {
                        itty_bit_string_t *mask = itty_bit_string_list_fetch (adapter_masks, input_index);
                        size_t bit_capacity = itty_bit_string_get_length (mask);

                        for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                                bool current_bit = itty_bit_string_get_bit (mask, bit_index);
                                itty_feed_model_decoder_objective_t candidate_objective = { 0 };

                                if (stats)
                                        stats->candidate_bits++;

                                itty_bit_string_set_bit (mask, bit_index, !current_bit);
                                mask->pop_count_computed = false;

                                if (itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                         input,
                                                                                         target,
                                                                                         route_index,
                                                                                         &candidate_objective) &&
                                    itty_feed_model_decoder_objective_accepts (&best_objective,
                                                                               &candidate_objective) &&
                                    !itty_feed_model_decoder_objective_accepts (&candidate_objective,
                                                                                &best_objective)) {
                                        found = true;
                                        best_input_index = input_index;
                                        best_bit_index = bit_index;
                                        best_objective = candidate_objective;
                                }

                                itty_bit_string_set_bit (mask, bit_index, current_bit);
                                mask->pop_count_computed = false;
                        }
                }

                if (!found)
                        break;

                {
                        itty_bit_string_t *mask = itty_bit_string_list_fetch (adapter_masks, best_input_index);
                        bool current_bit = itty_bit_string_get_bit (mask, best_bit_index);
                        itty_bit_string_set_bit (mask, best_bit_index, !current_bit);
                        mask->pop_count_computed = false;
                }
                current_objective = best_objective;
                if (stats)
                        stats->flips++;
        }

        if (stats)
                stats->largest_error = current_objective.selected_distance;
        return true;
}

bool
itty_feed_model_randomize_masks (itty_feed_model_t *model,
                                 size_t             seed,
                                 size_t             numerator,
                                 size_t             denominator)
{
        if (!model || denominator == 0 || numerator > denominator)
                return false;

        itty_feed_model_random_t random = {
                .state = seed == 0 ? 0x9e3779b97f4a7c15ULL : seed
        };

        for (size_t layer_index = 0; layer_index < model->number_of_layers; layer_index++) {
                for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                        itty_bit_string_list_t *masks = model->masks_by_layer_node[layer_index][node_index];

                        for (size_t mask_index = 0; mask_index < masks->count; mask_index++) {
                                itty_bit_string_t *mask = masks->bit_strings[mask_index];
                                size_t bit_capacity = itty_bit_string_get_length (mask);

                                memset (mask->words, 0, mask->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
                                for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                                        if (itty_feed_model_random_next (&random) % denominator < numerator)
                                                itty_bit_string_set_bit (mask, bit_index, true);
                                }

                                mask->pop_count_computed = false;
                        }
                }
        }

        return true;
}

bool
itty_feed_model_measure_masks (itty_feed_model_t                *model,
                               itty_model_metrics_bit_summary_t *summary)
{
        if (!model || !summary)
                return false;

        *summary = (itty_model_metrics_bit_summary_t) { 0 };

        for (size_t layer_index = 0; layer_index < model->number_of_layers; layer_index++) {
                itty_model_metrics_bit_summary_t layer_summary;

                if (!itty_feed_model_measure_layer_masks (model, layer_index, &layer_summary))
                        continue;

                summary->bit_count += layer_summary.bit_count;
                summary->set_bits += layer_summary.set_bits;
        }

        summary->unset_bits = summary->bit_count - summary->set_bits;
        summary->set_density = summary->bit_count == 0 ? 0.0 :
                               (double) summary->set_bits / (double) summary->bit_count;
        summary->entropy = itty_model_metrics_entropy_for_counts (summary->set_bits,
                                                                  summary->bit_count);

        return summary->bit_count > 0;
}

bool
itty_feed_model_measure_layer_masks (itty_feed_model_t                *model,
                                     size_t                            layer_index,
                                     itty_model_metrics_bit_summary_t *summary)
{
        if (!model || !summary || layer_index >= model->number_of_layers)
                return false;

        *summary = (itty_model_metrics_bit_summary_t) { 0 };

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_model_metrics_bit_summary_t node_summary;

                if (!itty_model_metrics_measure_bit_string_list (model->masks_by_layer_node[layer_index][node_index],
                                                                 &node_summary))
                        continue;

                summary->bit_count += node_summary.bit_count;
                summary->set_bits += node_summary.set_bits;
        }

        summary->unset_bits = summary->bit_count - summary->set_bits;
        summary->set_density = summary->bit_count == 0 ? 0.0 :
                               (double) summary->set_bits / (double) summary->bit_count;
        summary->entropy = itty_model_metrics_entropy_for_counts (summary->set_bits,
                                                                  summary->bit_count);

        return summary->bit_count > 0;
}

bool
itty_feed_model_measure_backward_layer_diagnostics (itty_feed_model_t                           *model,
                                                    itty_bit_string_list_t                      *input,
                                                    itty_bit_string_t                           *target,
                                                    itty_feed_model_train_options_t const       *options,
                                                    itty_feed_model_backward_layer_diagnostic_t *diagnostics,
                                                    size_t                                       diagnostic_count)
{
        if (!model ||
            !input ||
            !target ||
            !diagnostics ||
            diagnostic_count < model->number_of_layers ||
            model->number_of_layers == 0 ||
            model->nodes_per_layer != 1 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words ||
            !itty_feed_model_can_train_with_options (options))
                return false;

        for (size_t i = 0; i < diagnostic_count; i++)
                diagnostics[i] = (itty_feed_model_backward_layer_diagnostic_t) { 0 };

        itty_feed_model_backward_fold_t backward_fold = itty_feed_model_get_backward_fold (options);
        itty_bit_string_t *desired_output = itty_feed_model_expand_target_for_layer (target,
                                                                                     model->number_of_layers);

        for (size_t layer_count = model->number_of_layers; layer_count > 0; layer_count--) {
                size_t layer_index = layer_count - 1;
                itty_bit_string_list_t *layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                          input,
                                                                                          layer_index);
                itty_bit_string_list_t *masks = model->masks_by_layer_node[layer_index][0];
                itty_bit_string_t *desired_condensed = backward_fold == ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE ?
                                                       itty_feed_model_segment_condense_desired_output_for_layer (model,
                                                                                                                  desired_output,
                                                                                                                  layer_index) :
                                                       itty_feed_model_reduce_desired_output_for_layer (desired_output,
                                                                                                        model->rotations_by_layer[layer_index]);
                if (!desired_condensed) {
                        if (layer_input != input)
                                itty_bit_string_list_free (layer_input);
                        itty_bit_string_free (desired_output);
                        return false;
                }

                itty_bit_string_t *actual_condensed = itty_feed_model_run_node_condensed (layer_input,
                                                                                          masks);
                itty_bit_string_t *difference = itty_bit_string_exclusive_or (desired_condensed,
                                                                              actual_condensed);
                diagnostics[layer_index] = (itty_feed_model_backward_layer_diagnostic_t) {
                        .layer_index = layer_index,
                        .desired_bits = itty_bit_string_get_pop_count (desired_condensed),
                        .actual_bits = itty_bit_string_get_pop_count (actual_condensed),
                        .mismatched_bits = itty_bit_string_get_pop_count (difference),
                        .bit_count = (model->vocabulary_words << layer_index) * ITTY_BIT_STRING_WORD_SIZE_IN_BITS
                };

                itty_bit_string_free (difference);
                itty_bit_string_free (actual_condensed);

                if (layer_input != input)
                        itty_bit_string_list_free (layer_input);

                if (backward_fold == ITTY_FEED_MODEL_BACKWARD_FOLD_CHAINED_REDUCE) {
                        itty_bit_string_list_t *desired_layer_input = itty_feed_model_make_desired_layer_input (desired_condensed,
                                                                                                                masks);

                        itty_bit_string_free (desired_output);
                        desired_output = NULL;

                        if (layer_index > 0)
                                desired_output = itty_feed_model_bit_string_clone (itty_bit_string_list_fetch (desired_layer_input, 0));

                        itty_bit_string_list_free (desired_layer_input);
                }

                itty_bit_string_free (desired_condensed);
        }

        itty_bit_string_free (desired_output);
        return true;
}

bool
itty_feed_model_measure_final_layer_node_diagnostics (itty_feed_model_t                 *model,
                                                      itty_bit_string_list_t            *input,
                                                      itty_bit_string_t                 *target,
                                                      itty_feed_model_node_diagnostic_t *diagnostics,
                                                      size_t                             diagnostic_count)
{
        if (!model ||
            !input ||
            !target ||
            !diagnostics ||
            diagnostic_count < model->nodes_per_layer ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        for (size_t i = 0; i < diagnostic_count; i++)
                diagnostics[i] = (itty_feed_model_node_diagnostic_t) { 0 };

        size_t final_layer = model->number_of_layers - 1;
        itty_bit_string_list_t *final_layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                        input,
                                                                                        final_layer);
        itty_bit_string_t *expanded_target = itty_feed_model_expand_target_for_layer (target,
                                                                                      final_layer);

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_t *actual_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                          model->masks_by_layer_node[final_layer][node_index]);
                itty_bit_string_t *difference = itty_bit_string_exclusive_or (expanded_target,
                                                                              actual_condensed);

                diagnostics[node_index] = (itty_feed_model_node_diagnostic_t) {
                        .node_index = node_index,
                        .desired_bits = itty_bit_string_get_pop_count (expanded_target),
                        .actual_bits = itty_bit_string_get_pop_count (actual_condensed),
                        .mismatched_bits = itty_bit_string_get_pop_count (difference),
                        .bit_count = (model->vocabulary_words << final_layer) * ITTY_BIT_STRING_WORD_SIZE_IN_BITS
                };

                itty_bit_string_free (difference);
                itty_bit_string_free (actual_condensed);
        }

        itty_bit_string_free (expanded_target);
        if (final_layer_input != input)
                itty_bit_string_list_free (final_layer_input);

        return true;
}

bool
itty_feed_model_measure_suffix_oracle (itty_feed_model_t                              *model,
                                       itty_bit_string_list_t                         *input,
                                       itty_bit_string_t                              *target,
                                       itty_feed_model_suffix_oracle_options_t const   *options,
                                       itty_feed_model_suffix_oracle_summary_t         *summary)
{
        if (!model ||
            !input ||
            !target ||
            !options ||
            !summary ||
            model->number_of_layers == 0 ||
            options->layer_index >= model->number_of_layers ||
            options->node_index >= model->nodes_per_layer ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        *summary = (itty_feed_model_suffix_oracle_summary_t) { 0 };
        summary->best_distance = (size_t) -1;

        itty_bit_string_list_t *layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                  input,
                                                                                  options->layer_index);
        itty_bit_string_list_t *layer_outputs = itty_feed_model_run_layer (model,
                                                                           options->layer_index,
                                                                           layer_input);
        itty_feed_model_output_evaluation_t before_evaluation = { 0 };
        if (!itty_feed_model_evaluate_suffix_output (model,
                                                     layer_outputs,
                                                     options->layer_index,
                                                     target,
                                                     &before_evaluation)) {
                if (layer_input != input)
                        itty_bit_string_list_free (layer_input);
                itty_bit_string_list_free (layer_outputs);
                return false;
        }

        summary->before_distance = before_evaluation.distance;
        summary->best_distance = summary->before_distance;
        summary->worst_distance = summary->before_distance;

        itty_feed_model_decoder_objective_t before_objective = { 0 };
        if (!itty_feed_model_evaluate_suffix_decoder_objective (model,
                                                                layer_outputs,
                                                                options->layer_index,
                                                                target,
                                                                &before_objective)) {
                itty_bit_string_free (before_evaluation.folded_activation);
                if (layer_input != input)
                        itty_bit_string_list_free (layer_input);
                itty_bit_string_list_free (layer_outputs);
                return false;
        }

        itty_bit_string_t *desired_output = itty_feed_model_expand_target_for_layer (target,
                                                                                     model->number_of_layers);
        itty_bit_string_t *node_target = itty_feed_model_make_node_target_for_layer (model,
                                                                                     layer_input,
                                                                                     target,
                                                                                     desired_output,
                                                                                     options);
        itty_bit_string_free (desired_output);

        if (!node_target) {
                if (layer_input != input)
                        itty_bit_string_list_free (layer_input);
                itty_bit_string_list_free (layer_outputs);
                return false;
        }

        itty_bit_string_t *current_condensed = itty_feed_model_run_node_condensed (layer_input,
                                                                                   model->masks_by_layer_node[options->layer_index][options->node_index]);
        size_t bit_capacity = itty_bit_string_get_length (node_target);
        size_t candidate_limit = options->max_candidate_bits;
        itty_feed_model_random_t random = {
                .state = options->random_seed == 0 ? 0x9e3779b97f4a7c15ULL : options->random_seed
        };

        for (size_t bit_cursor = 0; bit_cursor < bit_capacity; bit_cursor++) {
                if (candidate_limit != 0 && summary->candidate_bits >= candidate_limit)
                        break;

                size_t bit_index = bit_cursor;
                bool candidate_bit = itty_bit_string_get_bit (node_target, bit_index);

                if (options->candidate_source == ITTY_FEED_MODEL_SUFFIX_ORACLE_CANDIDATES_RANDOM_ONE_BIT) {
                        bit_index = itty_feed_model_random_next (&random) % bit_capacity;
                        candidate_bit = !itty_bit_string_get_bit (current_condensed, bit_index);
                } else if (itty_bit_string_get_bit (current_condensed, bit_index) == candidate_bit) {
                        continue;
                }

                itty_bit_string_t *candidate_condensed = itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                                                     itty_bit_string_get_number_of_words (node_target));
                itty_bit_string_set_bit (candidate_condensed,
                                         bit_index,
                                         candidate_bit);
                candidate_condensed->pop_count_computed = false;

                itty_bit_string_t *candidate_output = itty_feed_model_expand_condensed_output (candidate_condensed,
                                                                                               model->rotations_by_layer[options->layer_index]);
                itty_bit_string_list_t *candidate_layer_outputs = itty_feed_model_bit_string_list_clone (layer_outputs);
                itty_bit_string_free (candidate_layer_outputs->bit_strings[options->node_index]);
                candidate_layer_outputs->bit_strings[options->node_index] = candidate_output;

                itty_feed_model_output_evaluation_t candidate_evaluation = { 0 };
                itty_feed_model_decoder_objective_t candidate_objective = { 0 };
                if (itty_feed_model_evaluate_suffix_output (model,
                                                            candidate_layer_outputs,
                                                            options->layer_index,
                                                            target,
                                                            &candidate_evaluation) &&
                    itty_feed_model_evaluate_suffix_decoder_objective (model,
                                                                       candidate_layer_outputs,
                                                                       options->layer_index,
                                                                       target,
                                                                       &candidate_objective)) {
                        summary->candidate_bits++;
                        if (candidate_evaluation.distance < summary->before_distance) {
                                summary->helpful_bits++;
                        } else if (candidate_evaluation.distance > summary->before_distance) {
                                summary->harmful_bits++;
                        } else {
                                summary->neutral_bits++;
                                if (candidate_evaluation.selected_index != before_evaluation.selected_index)
                                        summary->neutral_changed_selected_output_bits++;
                                else if (itty_bit_string_compare (candidate_evaluation.folded_activation,
                                                                  before_evaluation.folded_activation) == 0)
                                        summary->neutral_same_folded_output_bits++;
                                else
                                        summary->neutral_changed_folded_output_bits++;
                        }

                        if (candidate_evaluation.distance == summary->before_distance) {
                                if (candidate_objective.false_negative_blocker_bits < before_objective.false_negative_blocker_bits)
                                        summary->blocker_helpful_bits++;
                                else if (!itty_feed_model_decoder_objective_accepts (&before_objective,
                                                                                     &candidate_objective))
                                        summary->true_neutral_bits++;
                        }

                        if (candidate_evaluation.distance < summary->best_distance)
                                summary->best_distance = candidate_evaluation.distance;
                        if (candidate_evaluation.distance > summary->worst_distance)
                                summary->worst_distance = candidate_evaluation.distance;

                        itty_bit_string_free (candidate_evaluation.folded_activation);
                }

                itty_bit_string_list_free (candidate_layer_outputs);
                itty_bit_string_free (candidate_condensed);
        }

        itty_bit_string_free (before_evaluation.folded_activation);
        itty_bit_string_free (current_condensed);
        itty_bit_string_free (node_target);
        itty_bit_string_list_free (layer_outputs);
        if (layer_input != input)
                itty_bit_string_list_free (layer_input);

        return true;
}

bool
itty_feed_model_measure_residual_decode (itty_feed_model_t                         *model,
                                         itty_bit_string_list_t                    *input,
                                         itty_bit_string_t                         *target,
                                         itty_feed_model_residual_decode_summary_t *summary)
{
        if (!model ||
            !input ||
            !target ||
            !summary ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        itty_bit_string_list_t *outputs = itty_feed_model_run_outputs (model,
                                                                       input);
        if (!outputs)
                return false;

        itty_feed_model_output_evaluation_t evaluation = { 0 };
        bool measured = itty_feed_model_evaluate_output (model,
                                                         outputs,
                                                         target,
                                                         &evaluation);
        if (measured) {
                itty_feed_model_summarize_residual_decode (&evaluation,
                                                           target,
                                                           summary);
                itty_feed_model_decoder_objective_t objective = { 0 };
                if (itty_feed_model_evaluate_decoder_objective (model,
                                                                outputs,
                                                                target,
                                                                &objective)) {
                        summary->false_negative_blocker_bits = objective.false_negative_blocker_bits;
                        summary->zero_veto_safety_bits = objective.zero_veto_safety_bits;
                }
                itty_bit_string_free (evaluation.folded_activation);
        }

        itty_bit_string_list_free (outputs);
        return measured;
}

bool
itty_feed_model_measure_decoder_objective (itty_feed_model_t                   *model,
                                           itty_bit_string_list_t              *input,
                                           itty_bit_string_t                   *target,
                                           itty_feed_model_decoder_objective_t *objective)
{
        if (!model ||
            !input ||
            !target ||
            !objective ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        itty_bit_string_list_t *outputs = itty_feed_model_run_outputs (model,
                                                                       input);
        if (!outputs)
                return false;

        bool measured = itty_feed_model_evaluate_decoder_objective (model,
                                                                    outputs,
                                                                    target,
                                                                    objective);
        itty_bit_string_list_free (outputs);

        return measured;
}

bool
itty_feed_model_measure_decoder_objective_with_lane_split (itty_feed_model_t                   *model,
                                                           itty_bit_string_list_t              *input,
                                                           itty_bit_string_t                   *target,
                                                           size_t                               selector_lane_bit_offset,
                                                           size_t                               selector_lane_bit_count,
                                                           size_t                               decoder_lane_bit_offset,
                                                           size_t                               decoder_lane_bit_count,
                                                           itty_feed_model_decoder_objective_t *objective)
{
        if (!model ||
            !input ||
            !target ||
            !objective ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        itty_bit_string_list_t *outputs = itty_feed_model_run_outputs (model,
                                                                       input);
        if (!outputs)
                return false;

        bool measured = itty_feed_model_measure_decoder_objective_with_lane_split_from_outputs (model,
                                                                                                outputs,
                                                                                                target,
                                                                                                selector_lane_bit_offset,
                                                                                                selector_lane_bit_count,
                                                                                                decoder_lane_bit_offset,
                                                                                                decoder_lane_bit_count,
                                                                                                objective);
        itty_bit_string_list_free (outputs);
        return measured;
}

bool
itty_feed_model_measure_decoder_objective_for_node (itty_feed_model_t                   *model,
                                                    itty_bit_string_list_t              *input,
                                                    itty_bit_string_t                   *target,
                                                    size_t                               selected_node,
                                                    itty_feed_model_decoder_objective_t *objective)
{
        if (!model ||
            !input ||
            !target ||
            !objective ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        itty_bit_string_list_t *outputs = itty_feed_model_run_to_layer_input (model,
                                                                              input,
                                                                              model->number_of_layers);
        if (!outputs)
                return false;

        bool measured = itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                             outputs,
                                                                             target,
                                                                             selected_node,
                                                                             objective);
        itty_bit_string_list_free (outputs);

        return measured;
}

bool
itty_feed_model_measure_decoder_objective_for_node_with_lane_split (itty_feed_model_t                   *model,
                                                                    itty_bit_string_list_t              *input,
                                                                    itty_bit_string_t                   *target,
                                                                    size_t                               selected_node,
                                                                    size_t                               selector_lane_bit_offset,
                                                                    size_t                               selector_lane_bit_count,
                                                                    size_t                               decoder_lane_bit_offset,
                                                                    size_t                               decoder_lane_bit_count,
                                                                    itty_feed_model_decoder_objective_t *objective)
{
        if (!model ||
            !input ||
            !target ||
            !objective ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        itty_bit_string_list_t *outputs = itty_feed_model_run_outputs (model,
                                                                       input);
        if (!outputs)
                return false;

        bool measured = itty_feed_model_measure_decoder_objective_for_node_with_lane_split_from_outputs (model,
                                                                                                         outputs,
                                                                                                         target,
                                                                                                         selected_node,
                                                                                                         selector_lane_bit_offset,
                                                                                                         selector_lane_bit_count,
                                                                                                         decoder_lane_bit_offset,
                                                                                                         decoder_lane_bit_count,
                                                                                                         objective);
        itty_bit_string_list_free (outputs);
        return measured;
}

bool
itty_feed_model_measure_segment_node_selection (itty_feed_model_t                                *model,
                                                itty_bit_string_list_t                           *input,
                                                itty_bit_string_t                                *target,
                                                itty_feed_model_segment_node_selection_summary_t *summary)
{
        if (!model ||
            !input ||
            !target ||
            !summary ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        *summary = (itty_feed_model_segment_node_selection_summary_t) { 0 };

        itty_feed_model_decoder_t previous_decoder = model->decoder;
        model->decoder = previous_decoder == ITTY_FEED_MODEL_DECODER_SEGMENT_WEIGHTED_CONDENSE ?
                         ITTY_FEED_MODEL_DECODER_SEGMENT_WEIGHTED_CONDENSE :
                         ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE;

        itty_network_t *network = itty_feed_model_build_network (model);
        itty_bit_string_list_t *outputs = itty_network_feed (network,
                                                             input);
        itty_network_free (network);
        if (!outputs) {
                model->decoder = previous_decoder;
                return false;
        }

        size_t selected_index = 0;
        if (!itty_network_select_output (outputs, &selected_index)) {
                itty_bit_string_list_free (outputs);
                model->decoder = previous_decoder;
                return false;
        }
        itty_feed_model_decoder_objective_t selected_objective = { 0 };
        if (!itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                  outputs,
                                                                  target,
                                                                  selected_index,
                                                                  &selected_objective)) {
                itty_bit_string_list_free (outputs);
                model->decoder = previous_decoder;
                return false;
        }

        summary->selected_by_popcount = selected_objective.selected_node;
        summary->selected_popcount = selected_objective.selected_popcount;
        summary->best_by_target_distance = selected_objective.best_decoded_node;
        summary->selected_distance = selected_objective.selected_distance;
        summary->best_target_distance = selected_objective.best_decoded_distance;
        summary->distance_gap_between_selected_and_best =
                selected_objective.selected_distance >= selected_objective.best_decoded_distance ?
                selected_objective.selected_distance - selected_objective.best_decoded_distance :
                0;
        summary->selected_false_negative_deficit = selected_objective.false_negative_vote_deficit;
        summary->selected_false_positive_excess = selected_objective.false_positive_vote_excess;
        summary->best_false_negative_deficit = selected_objective.false_negative_vote_deficit;
        summary->best_false_positive_excess = selected_objective.false_positive_vote_excess;
        summary->best_by_false_negative_deficit = selected_objective.selected_node;
        summary->best_by_false_positive_excess = selected_objective.selected_node;

        size_t runner_up_popcount = 0;
        for (size_t node_index = 0; node_index < outputs->count; node_index++) {
                itty_bit_string_t *activation = itty_bit_string_list_fetch (outputs,
                                                                            node_index);
                size_t popcount = itty_bit_string_get_pop_count (activation);

                if (node_index != selected_objective.selected_node &&
                    popcount > runner_up_popcount)
                        runner_up_popcount = popcount;

                itty_feed_model_decoder_objective_t node_objective = { 0 };
                if (!itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                          outputs,
                                                                          target,
                                                                          node_index,
                                                                          &node_objective))
                        continue;

                if (node_objective.false_negative_vote_deficit < summary->best_false_negative_deficit) {
                        summary->best_false_negative_deficit = node_objective.false_negative_vote_deficit;
                        summary->best_by_false_negative_deficit = node_index;
                }
                if (node_objective.false_positive_vote_excess < summary->best_false_positive_excess) {
                        summary->best_false_positive_excess = node_objective.false_positive_vote_excess;
                        summary->best_by_false_positive_excess = node_index;
                }
        }

        summary->popcount_gap = summary->selected_popcount >= runner_up_popcount ?
                                summary->selected_popcount - runner_up_popcount :
                                0;

        itty_bit_string_list_free (outputs);
        model->decoder = previous_decoder;
        return true;
}

bool
itty_feed_model_measure_segment_node_selection_with_lane_split (itty_feed_model_t                                *model,
                                                                itty_bit_string_list_t                           *input,
                                                                itty_bit_string_t                                *target,
                                                                size_t                                            selector_lane_bit_offset,
                                                                size_t                                            selector_lane_bit_count,
                                                                size_t                                            decoder_lane_bit_offset,
                                                                size_t                                            decoder_lane_bit_count,
                                                                itty_feed_model_segment_node_selection_summary_t *summary)
{
        if (!model ||
            !input ||
            !target ||
            !summary ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        *summary = (itty_feed_model_segment_node_selection_summary_t) { 0 };

        itty_network_t *network = itty_feed_model_build_network (model);
        itty_bit_string_list_t *outputs = itty_network_feed (network, input);
        itty_network_free (network);
        if (!outputs)
                return false;

        itty_bit_string_list_t *selector_outputs = itty_feed_model_clone_outputs_lane_range (outputs,
                                                                                              selector_lane_bit_offset,
                                                                                              selector_lane_bit_count);
        itty_bit_string_list_t *decoder_outputs = itty_feed_model_clone_outputs_lane_range (outputs,
                                                                                             decoder_lane_bit_offset,
                                                                                             decoder_lane_bit_count);
        if (!selector_outputs || !decoder_outputs) {
                if (selector_outputs)
                        itty_bit_string_list_free (selector_outputs);
                if (decoder_outputs)
                        itty_bit_string_list_free (decoder_outputs);
                itty_bit_string_list_free (outputs);
                return false;
        }

        size_t selected_index = 0;
        size_t selected_popcount = 0;
        size_t runner_up_popcount = 0;

        for (size_t node_index = 0; node_index < selector_outputs->count; node_index++) {
                size_t popcount = itty_bit_string_get_pop_count (itty_bit_string_list_fetch (selector_outputs,
                                                                                              node_index));
                if (node_index == 0 || popcount > selected_popcount) {
                        runner_up_popcount = selected_popcount;
                        selected_popcount = popcount;
                        selected_index = node_index;
                } else if (popcount > runner_up_popcount) {
                        runner_up_popcount = popcount;
                }
        }

        itty_feed_model_decoder_objective_t selected_objective = { 0 };
        bool measured = itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                             decoder_outputs,
                                                                             target,
                                                                             selected_index,
                                                                             &selected_objective);
        if (measured) {
                summary->selected_by_popcount = selected_index;
                summary->selected_popcount = selected_popcount;
                summary->best_by_target_distance = selected_objective.best_decoded_node;
                summary->selected_distance = selected_objective.selected_distance;
                summary->best_target_distance = selected_objective.best_decoded_distance;
                summary->distance_gap_between_selected_and_best =
                        selected_objective.selected_distance >= selected_objective.best_decoded_distance ?
                        selected_objective.selected_distance - selected_objective.best_decoded_distance :
                        0;
                summary->selected_false_negative_deficit = selected_objective.false_negative_vote_deficit;
                summary->selected_false_positive_excess = selected_objective.false_positive_vote_excess;
                summary->best_false_negative_deficit = selected_objective.false_negative_vote_deficit;
                summary->best_false_positive_excess = selected_objective.false_positive_vote_excess;
                summary->best_by_false_negative_deficit = selected_index;
                summary->best_by_false_positive_excess = selected_index;
                summary->popcount_gap = selected_popcount >= runner_up_popcount ?
                                        selected_popcount - runner_up_popcount :
                                        0;

                for (size_t node_index = 0; node_index < decoder_outputs->count; node_index++) {
                        itty_feed_model_decoder_objective_t node_objective = { 0 };
                        if (!itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                                  decoder_outputs,
                                                                                  target,
                                                                                  node_index,
                                                                                  &node_objective))
                                continue;
                        if (node_objective.false_negative_vote_deficit < summary->best_false_negative_deficit) {
                                summary->best_false_negative_deficit = node_objective.false_negative_vote_deficit;
                                summary->best_by_false_negative_deficit = node_index;
                        }
                        if (node_objective.false_positive_vote_excess < summary->best_false_positive_excess) {
                                summary->best_false_positive_excess = node_objective.false_positive_vote_excess;
                                summary->best_by_false_positive_excess = node_index;
                        }
                }
        }

        itty_bit_string_list_free (selector_outputs);
        itty_bit_string_list_free (decoder_outputs);
        itty_bit_string_list_free (outputs);
        return measured;
}

bool
itty_feed_model_measure_segment_node_polarity_selection (itty_feed_model_t                                *model,
                                                         itty_bit_string_list_t                           *input,
                                                         itty_bit_string_t                                *target,
                                                         itty_feed_model_segment_node_polarity_summary_t *summary)
{
        if (!model ||
            !input ||
            !target ||
            !summary ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        *summary = (itty_feed_model_segment_node_polarity_summary_t) { 0 };

        itty_feed_model_decoder_t previous_decoder = model->decoder;
        model->decoder = previous_decoder == ITTY_FEED_MODEL_DECODER_SEGMENT_WEIGHTED_CONDENSE ?
                         ITTY_FEED_MODEL_DECODER_SEGMENT_WEIGHTED_CONDENSE :
                         ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE;

        itty_network_t *network = itty_feed_model_build_network (model);
        itty_bit_string_list_t *outputs = itty_network_feed (network,
                                                             input);
        itty_network_free (network);
        if (!outputs) {
                model->decoder = previous_decoder;
                return false;
        }

        itty_feed_model_decoder_objective_t best_objective = { .selected_distance = (size_t) -1 };
        itty_feed_model_decoder_objective_t best_normal_objective = { .selected_distance = (size_t) -1 };
        itty_feed_model_decoder_objective_t best_complement_objective = { .selected_distance = (size_t) -1 };
        bool best_complemented = false;
        size_t best_normal_node = 0;
        size_t best_complement_node = 0;

        for (size_t node_index = 0; node_index < outputs->count; node_index++) {
                itty_bit_string_t *activation = itty_bit_string_list_fetch (outputs,
                                                                            node_index);
                itty_feed_model_decoder_objective_t normal_objective = { 0 };
                itty_feed_model_decoder_objective_t complement_objective = { 0 };
                itty_bit_string_t *complement = NULL;

                if (itty_feed_model_evaluate_decoder_objective_for_activation (model,
                                                                               activation,
                                                                               target,
                                                                               node_index,
                                                                               &normal_objective)) {
                        if (itty_feed_model_decoder_objective_is_better (&normal_objective,
                                                                         &best_normal_objective)) {
                                best_normal_objective = normal_objective;
                                best_normal_node = node_index;
                        }
                        if (itty_feed_model_decoder_objective_is_better (&normal_objective,
                                                                         &best_objective)) {
                                best_objective = normal_objective;
                                summary->best_node = node_index;
                                best_complemented = false;
                        }
                }

                complement = itty_feed_model_bit_string_complement_clone (activation);
                if (itty_feed_model_evaluate_decoder_objective_for_activation (model,
                                                                               complement,
                                                                               target,
                                                                               node_index,
                                                                               &complement_objective)) {
                        if (itty_feed_model_decoder_objective_is_better (&complement_objective,
                                                                         &best_complement_objective)) {
                                best_complement_objective = complement_objective;
                                best_complement_node = node_index;
                        }
                        if (itty_feed_model_decoder_objective_is_better (&complement_objective,
                                                                         &best_objective)) {
                                best_objective = complement_objective;
                                summary->best_node = node_index;
                                best_complemented = true;
                        }
                }
                itty_bit_string_free (complement);
        }

        summary->best_complemented = best_complemented;
        summary->best_distance = best_objective.selected_distance;
        summary->best_false_negative_deficit = best_objective.false_negative_vote_deficit;
        summary->best_false_positive_excess = best_objective.false_positive_vote_excess;
        summary->best_target_zero_safety = best_objective.target_zero_safety;
        summary->best_popcount = best_objective.selected_popcount;
        summary->best_normal_node = best_normal_node;
        summary->best_normal_distance = best_normal_objective.selected_distance;
        summary->best_normal_false_negative_deficit = best_normal_objective.false_negative_vote_deficit;
        summary->best_complement_node = best_complement_node;
        summary->best_complement_distance = best_complement_objective.selected_distance;
        summary->best_complement_false_negative_deficit = best_complement_objective.false_negative_vote_deficit;

        itty_bit_string_list_free (outputs);
        model->decoder = previous_decoder;
        return true;
}

bool
itty_feed_model_measure_segment_transform (itty_feed_model_t                            *model,
                                           itty_bit_string_list_t                       *input,
                                           itty_bit_string_t                            *target,
                                           itty_feed_model_output_transform_t           transform,
                                           itty_feed_model_segment_transform_summary_t *summary)
{
        if (!model ||
            !input ||
            !target ||
            !summary ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        *summary = (itty_feed_model_segment_transform_summary_t) {
                .transform = transform,
        };

        itty_feed_model_decoder_t previous_decoder = model->decoder;
        model->decoder = previous_decoder == ITTY_FEED_MODEL_DECODER_SEGMENT_WEIGHTED_CONDENSE ?
                         ITTY_FEED_MODEL_DECODER_SEGMENT_WEIGHTED_CONDENSE :
                         ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE;

        itty_network_t *network = itty_feed_model_build_network (model);
        itty_bit_string_list_t *outputs = itty_network_feed (network,
                                                             input);
        itty_network_free (network);
        if (!outputs) {
                model->decoder = previous_decoder;
                return false;
        }

        itty_bit_string_list_t *transformed_outputs = itty_feed_model_bit_string_list_clone (outputs);
        itty_feed_model_apply_output_transform (transformed_outputs,
                                                transform);

        itty_feed_model_decoder_objective_t objective = { 0 };
        bool measured = itty_feed_model_evaluate_decoder_objective (model,
                                                                    transformed_outputs,
                                                                    target,
                                                                    &objective);
        if (measured) {
            summary->selected_distance = objective.selected_distance;
            summary->false_negative_vote_deficit = objective.false_negative_vote_deficit;
            summary->false_positive_vote_excess = objective.false_positive_vote_excess;
            summary->target_zero_safety = objective.target_zero_safety;
            summary->selected_node = objective.selected_node;
            summary->selected_popcount = objective.selected_popcount;
            summary->best_decoded_node = objective.best_decoded_node;
            summary->best_decoded_distance = objective.best_decoded_distance;
        }

        itty_bit_string_list_free (transformed_outputs);
        itty_bit_string_list_free (outputs);
        model->decoder = previous_decoder;
        return measured;
}

bool
itty_feed_model_measure_final_layer_replay_transaction (itty_feed_model_t                            *model,
                                                        itty_bit_string_list_t                       *first_input,
                                                        itty_bit_string_t                            *first_target,
                                                        itty_bit_string_list_t                       *second_input,
                                                        itty_bit_string_t                            *second_target,
                                                        itty_feed_model_train_options_t const       *options,
                                                        size_t                                        max_second_steps,
                                                        size_t                                        max_first_repair_steps,
                                                        itty_feed_model_replay_transaction_summary_t *summary)
{
        if (summary)
                *summary = (itty_feed_model_replay_transaction_summary_t) { 0 };

        if (!model ||
            !first_input ||
            !first_target ||
            !second_input ||
            !second_target ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (first_input) != model->inputs_per_node ||
            itty_bit_string_list_get_length (second_input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (first_target) > model->vocabulary_words ||
            itty_bit_string_get_number_of_words (second_target) > model->vocabulary_words ||
            !itty_feed_model_can_train_with_options (options))
                return false;

        size_t final_layer = model->number_of_layers - 1;
        itty_feed_model_layer_state_snapshot_t *snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                 final_layer);
        itty_feed_model_decoder_objective_t first_before = { 0 };
        itty_feed_model_decoder_objective_t second_before = { 0 };
        itty_feed_model_decoder_objective_t first_after_b = { 0 };
        itty_feed_model_decoder_objective_t second_after_b = { 0 };
        itty_feed_model_decoder_objective_t first_after_repair = { 0 };
        itty_feed_model_decoder_objective_t second_after_repair = { 0 };

        if (!itty_feed_model_measure_decoder_objective (model,
                                                        first_input,
                                                        first_target,
                                                        &first_before) ||
            !itty_feed_model_measure_decoder_objective (model,
                                                        second_input,
                                                        second_target,
                                                        &second_before)) {
                itty_feed_model_restore_layer_state_snapshot (model,
                                                              final_layer,
                                                              snapshot);
                return false;
        }

        first_after_b = first_before;
        second_after_b = second_before;
        first_after_repair = first_before;
        second_after_repair = second_before;

        size_t second_steps = 0;
        size_t first_repair_steps = 0;
        size_t second_flips = 0;
        size_t first_repair_flips = 0;

        for (size_t step = 0; step < max_second_steps; step++) {
                itty_feed_model_train_stats_t step_stats = { 0 };

                if (!itty_feed_model_train_final_layer_with_suffix_oracle (model,
                                                                           second_input,
                                                                           second_target,
                                                                           options,
                                                                           &step_stats))
                        break;

                second_flips += step_stats.flips;
                if (!itty_feed_model_measure_decoder_objective (model,
                                                                first_input,
                                                                first_target,
                                                                &first_after_b) ||
                    !itty_feed_model_measure_decoder_objective (model,
                                                                second_input,
                                                                second_target,
                                                                &second_after_b)) {
                        itty_feed_model_restore_layer_state_snapshot (model,
                                                                      final_layer,
                                                                      snapshot);
                        return false;
                }

                second_steps++;
                if (first_after_b.selected_distance > first_before.selected_distance &&
                    (second_after_b.selected_distance < second_before.selected_distance ||
                     second_after_b.false_negative_vote_deficit < second_before.false_negative_vote_deficit))
                        break;
        }

        first_after_repair = first_after_b;
        second_after_repair = second_after_b;
        for (size_t step = 0; step < max_first_repair_steps; step++) {
                if (first_after_repair.selected_distance <= first_before.selected_distance &&
                    first_after_repair.false_positive_vote_excess <= first_before.false_positive_vote_excess)
                        break;

                itty_feed_model_train_stats_t step_stats = { 0 };

                if (!itty_feed_model_train_final_layer_with_suffix_oracle (model,
                                                                           first_input,
                                                                           first_target,
                                                                           options,
                                                                           &step_stats))
                        break;

                first_repair_flips += step_stats.flips;
                if (!itty_feed_model_measure_decoder_objective (model,
                                                                first_input,
                                                                first_target,
                                                                &first_after_repair) ||
                    !itty_feed_model_measure_decoder_objective (model,
                                                                second_input,
                                                                second_target,
                                                                &second_after_repair)) {
                        itty_feed_model_restore_layer_state_snapshot (model,
                                                                      final_layer,
                                                                      snapshot);
                        return false;
                }

                first_repair_steps++;
        }

        bool accepted = first_after_repair.selected_distance <= first_before.selected_distance &&
                        first_after_repair.false_positive_vote_excess <= first_before.false_positive_vote_excess &&
                        (second_after_repair.selected_distance < second_before.selected_distance ||
                         (second_after_repair.selected_distance == second_before.selected_distance &&
                          second_after_repair.false_negative_vote_deficit < second_before.false_negative_vote_deficit));
        bool a_restored = first_after_repair.selected_distance <= first_before.selected_distance &&
                          first_after_repair.false_positive_vote_excess <= first_before.false_positive_vote_excess;
        bool b_preserved = second_after_repair.selected_distance < second_before.selected_distance ||
                           (second_after_repair.selected_distance == second_before.selected_distance &&
                            second_after_repair.false_negative_vote_deficit < second_before.false_negative_vote_deficit);

        if (summary) {
                *summary = (itty_feed_model_replay_transaction_summary_t) {
                        .a_distance_before = first_before.selected_distance,
                        .a_distance_after_b = first_after_b.selected_distance,
                        .a_distance_after_repair = first_after_repair.selected_distance,
                        .a_false_positive_excess_before = first_before.false_positive_vote_excess,
                        .a_false_positive_excess_after_b = first_after_b.false_positive_vote_excess,
                        .a_false_positive_excess_after_repair = first_after_repair.false_positive_vote_excess,
                        .b_distance_before = second_before.selected_distance,
                        .b_distance_after_b = second_after_b.selected_distance,
                        .b_distance_after_repair = second_after_repair.selected_distance,
                        .b_false_negative_deficit_before = second_before.false_negative_vote_deficit,
                        .b_false_negative_deficit_after_b = second_after_b.false_negative_vote_deficit,
                        .b_false_negative_deficit_after_repair = second_after_repair.false_negative_vote_deficit,
                        .b_steps = second_steps,
                        .a_repair_steps = first_repair_steps,
                        .b_flips = second_flips,
                        .a_repair_flips = first_repair_flips,
                        .a_restored = a_restored,
                        .b_preserved = b_preserved,
                        .accepted = accepted
                };
        }

        itty_feed_model_restore_layer_state_snapshot (model,
                                                      final_layer,
                                                      snapshot);
        return true;
}

bool
itty_feed_model_measure_final_layer_restore_failure (itty_feed_model_t                          *model,
                                                     itty_bit_string_list_t                     *first_input,
                                                     itty_bit_string_t                          *first_target,
                                                     itty_bit_string_list_t                     *second_input,
                                                     itty_bit_string_t                          *second_target,
                                                     itty_feed_model_train_options_t const     *options,
                                                     size_t                                      max_second_steps,
                                                     bool                                        preserve_second,
                                                     itty_feed_model_restore_failure_summary_t *summary)
{
        if (!summary)
                return false;
        *summary = (itty_feed_model_restore_failure_summary_t) { 0 };

        if (!model ||
            !first_input ||
            !first_target ||
            !second_input ||
            !second_target ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (first_input) != model->inputs_per_node ||
            itty_bit_string_list_get_length (second_input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (first_target) > model->vocabulary_words ||
            itty_bit_string_get_number_of_words (second_target) > model->vocabulary_words ||
            !itty_feed_model_can_train_with_options (options))
                return false;

        size_t final_layer = model->number_of_layers - 1;
        itty_feed_model_layer_state_snapshot_t *snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                 final_layer);
        itty_feed_model_decoder_objective_t first_before = { 0 };
        itty_feed_model_decoder_objective_t first_after_b = { 0 };
        itty_feed_model_decoder_objective_t second_before = { 0 };
        itty_feed_model_decoder_objective_t second_after_b = { 0 };

        if (!itty_feed_model_measure_decoder_objective (model,
                                                        first_input,
                                                        first_target,
                                                        &first_before) ||
            !itty_feed_model_measure_decoder_objective (model,
                                                        second_input,
                                                        second_target,
                                                        &second_before)) {
                itty_feed_model_restore_layer_state_snapshot (model,
                                                              final_layer,
                                                              snapshot);
                return false;
        }

        first_after_b = first_before;
        second_after_b = second_before;
        for (size_t step = 0; step < max_second_steps; step++) {
                itty_feed_model_train_stats_t step_stats = { 0 };

                if (!itty_feed_model_train_final_layer_with_suffix_oracle (model,
                                                                           second_input,
                                                                           second_target,
                                                                           options,
                                                                           &step_stats))
                        break;

                if (!itty_feed_model_measure_decoder_objective (model,
                                                                first_input,
                                                                first_target,
                                                                &first_after_b) ||
                    !itty_feed_model_measure_decoder_objective (model,
                                                                second_input,
                                                                second_target,
                                                                &second_after_b)) {
                        itty_feed_model_restore_layer_state_snapshot (model,
                                                                      final_layer,
                                                                      snapshot);
                        return false;
                }

                if (first_after_b.selected_distance > first_before.selected_distance &&
                    (second_after_b.selected_distance < second_before.selected_distance ||
                     second_after_b.false_negative_vote_deficit < second_before.false_negative_vote_deficit))
                        break;
        }

        summary->a_distance_after_b = first_after_b.selected_distance;
        summary->a_false_positive_excess_after_b = first_after_b.false_positive_vote_excess;

        if (first_after_b.false_positive_vote_excess == 0) {
                if (summary)
                        summary->no_flip_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NO_FALSE_POSITIVES;
                itty_feed_model_restore_layer_state_snapshot (model,
                                                              final_layer,
                                                              snapshot);
                return true;
        }

        itty_bit_string_list_t *final_layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                        first_input,
                                                                                        final_layer);
        itty_bit_string_list_t *final_outputs = itty_feed_model_run_layer (model,
                                                                           final_layer,
                                                                           final_layer_input);
        itty_feed_model_output_evaluation_t evaluation = { 0 };
        bool ok = final_outputs &&
                  itty_feed_model_evaluate_output (model,
                                                  final_outputs,
                                                  first_target,
                                                  &evaluation);
        if (!ok) {
                if (final_outputs)
                        itty_bit_string_list_free (final_outputs);
                if (final_layer_input != first_input)
                        itty_bit_string_list_free (final_layer_input);
                itty_feed_model_restore_layer_state_snapshot (model,
                                                              final_layer,
                                                              snapshot);
                return false;
        }

        size_t target_bit_capacity = itty_bit_string_get_number_of_words (first_target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        itty_bit_string_t *selected_activation = itty_bit_string_list_fetch (final_outputs,
                                                                             evaluation.selected_index);
        size_t activation_bit_capacity = itty_bit_string_get_length (selected_activation);
        size_t segments_per_bit = target_bit_capacity == 0 ? 0 : activation_bit_capacity / target_bit_capacity;
        itty_bit_string_t *selected_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                    model->masks_by_layer_node[final_layer][evaluation.selected_index]);
        itty_feed_model_decoder_objective_t first_selected_before = { 0 };
        bool have_candidates = false;
        bool have_useful_repairs = false;
        bool have_b_safe_repairs = false;
        bool have_flipping_repairs = false;

        if (!selected_condensed ||
            !itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                  final_outputs,
                                                                  first_target,
                                                                  evaluation.selected_index,
                                                                  &first_selected_before)) {
                if (selected_condensed)
                        itty_bit_string_free (selected_condensed);
                itty_bit_string_free (evaluation.folded_activation);
                itty_bit_string_list_free (final_outputs);
                if (final_layer_input != first_input)
                        itty_bit_string_list_free (final_layer_input);
                itty_feed_model_restore_layer_state_snapshot (model,
                                                              final_layer,
                                                              snapshot);
                return false;
        }

        for (size_t target_bit_index = 0;
             target_bit_index < target_bit_capacity &&
             summary &&
             summary->trace_count < ITTY_FEED_MODEL_RESTORE_TRACE_LIMIT;
             target_bit_index++) {
                bool target_bit = itty_bit_string_get_bit (first_target,
                                                           target_bit_index);
                bool folded_bit = itty_bit_string_get_bit (evaluation.folded_activation,
                                                           target_bit_index);
                size_t blocker_bits = 0;
                size_t safety_bits = 0;
                size_t positive_margin = 0;
                size_t negative_excess = 0;

                if (target_bit || !folded_bit)
                        continue;

                itty_feed_model_count_decoder_ancestor_state (model,
                                                              itty_bit_string_list_fetch (final_outputs,
                                                                                          evaluation.selected_index),
                                                              first_target,
                                                              target_bit_index,
                                                              false,
                                                              true,
                                                              &blocker_bits,
                                                              &safety_bits,
                                                              &positive_margin,
                                                              &negative_excess);

                size_t current_ones = 0;
                size_t clearable_segment_votes = 0;
                for (size_t output_bit_index = target_bit_index;
                     output_bit_index < activation_bit_capacity;
                     output_bit_index += target_bit_capacity) {
                        if (itty_bit_string_get_bit (itty_bit_string_list_fetch (final_outputs,
                                                                                 evaluation.selected_index),
                                                     output_bit_index)) {
                                current_ones++;
                                clearable_segment_votes++;
                        }
                }

                itty_feed_model_restore_failure_trace_t trace = {
                        .decoded_bit = target_bit_index,
                        .target_bit = target_bit,
                        .current_decoded_bit = folded_bit,
                        .current_ones = current_ones,
                        .threshold = segments_per_bit / 2 + 1,
                        .max_ones_for_zero = segments_per_bit == 0 ? 0 : (segments_per_bit / 2 + 1) - 1,
                        .excess = negative_excess,
                        .segment_votes_currently_one = current_ones,
                        .clearable_segment_votes = clearable_segment_votes,
                        .candidate_final_output_bits = clearable_segment_votes,
                        .candidate_segment_votes = current_ones,
                        .min_final_layer_mask_flips_needed = (size_t) -1,
                        .rejection_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NO_CANDIDATE_REPAIRS
                };

                size_t useful_candidates = 0;
                size_t safe_candidates = 0;
                size_t candidate_mask_flips = 0;
                bool accepted = false;

                itty_bit_string_t *candidate_condensed = itty_feed_model_bit_string_clone_to_words (selected_condensed,
                                                                                                    itty_bit_string_get_number_of_words (selected_condensed));
                size_t projected_condensed_bits = 0;
                bool *projected_seen = calloc (itty_bit_string_get_length (selected_condensed),
                                               sizeof (bool));
                for (size_t output_bit_index = target_bit_index;
                     output_bit_index < activation_bit_capacity;
                     output_bit_index += target_bit_capacity) {
                        if (!itty_bit_string_get_bit (selected_activation,
                                                      output_bit_index))
                                continue;
                        size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                                    final_layer,
                                                                                                    output_bit_index);
                        if (!projected_seen[condensed_bit_index]) {
                                projected_seen[condensed_bit_index] = true;
                                projected_condensed_bits++;
                        }
                }
                bool changed = itty_feed_model_final_layer_apply_decoder_block (model,
                                                                                target_bit_index,
                                                                                false,
                                                                                candidate_condensed);
                free (projected_seen);
                if (changed) {
                        size_t changed_condensed_bits = 0;
                        for (size_t bit_index = 0;
                             bit_index < itty_bit_string_get_length (candidate_condensed);
                             bit_index++) {
                                if (itty_bit_string_get_bit (candidate_condensed, bit_index) !=
                                    itty_bit_string_get_bit (selected_condensed, bit_index))
                                        changed_condensed_bits++;
                        }

                        summary->a_candidate_repair_count++;
                        summary->a_target_zero_repair_count++;
                        have_candidates = true;
                        candidate_mask_flips = changed_condensed_bits;

                        itty_feed_model_layer_state_snapshot_t *candidate_snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                                           final_layer);
                        itty_feed_model_train_stats_t repair_stats = { 0 };
                        itty_feed_model_decoder_objective_t a_after_candidate = { 0 };
                        itty_feed_model_decoder_objective_t a_after_candidate_forced = { 0 };
                        itty_feed_model_decoder_objective_t a_after_candidate_selected = { 0 };
                        itty_feed_model_decoder_objective_t b_after_candidate = { 0 };
                        itty_bit_string_list_t *after_final_outputs = NULL;
                        itty_feed_model_output_evaluation_t after_evaluation = { 0 };

                        itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][evaluation.selected_index],
                                                              final_layer_input,
                                                              candidate_condensed,
                                                              options,
                                                              &repair_stats);
                        bool measured = itty_feed_model_measure_decoder_objective (model,
                                                                                   first_input,
                                                                                   first_target,
                                                                                   &a_after_candidate) &&
                                        itty_feed_model_measure_decoder_objective (model,
                                                                                   second_input,
                                                                                   second_target,
                                                                                   &b_after_candidate) &&
                                        (after_final_outputs = itty_feed_model_run_layer (model,
                                                                                           final_layer,
                                                                                           final_layer_input)) &&
                                        itty_feed_model_evaluate_output (model,
                                                                         after_final_outputs,
                                                                         first_target,
                                                                         &after_evaluation) &&
                                        itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                                              after_final_outputs,
                                                                                              first_target,
                                                                                              evaluation.selected_index,
                                                                                              &a_after_candidate_forced) &&
                                        itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                                              after_final_outputs,
                                                                                              first_target,
                                                                                              after_evaluation.selected_index,
                                                                                              &a_after_candidate_selected) &&
                                        true;

                        bool useful = measured &&
                                      (a_after_candidate_forced.selected_distance < first_selected_before.selected_distance ||
                                       a_after_candidate_forced.false_positive_vote_excess < first_selected_before.false_positive_vote_excess);
                        bool b_safe = !preserve_second ||
                                      (measured &&
                                       b_after_candidate.selected_distance <= second_after_b.selected_distance &&
                                       b_after_candidate.false_negative_vote_deficit <= second_after_b.false_negative_vote_deficit &&
                                       b_after_candidate.false_positive_vote_excess <= second_after_b.false_positive_vote_excess);

                        if (useful) {
                                useful_candidates++;
                                summary->a_useful_repair_count++;
                                have_useful_repairs = true;
                        }
                        if (b_safe) {
                                safe_candidates++;
                                have_b_safe_repairs = true;
                        }
                        if (repair_stats.flips > 0) {
                                have_flipping_repairs = true;
                                trace.min_final_layer_mask_flips_needed = repair_stats.flips;
                        }
                        if (useful && b_safe && repair_stats.flips > 0)
                                accepted = true;

                        if (measured) {
                                size_t threshold = segments_per_bit / 2 + 1;
                                itty_bit_string_t *after_selected_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                                                    model->masks_by_layer_node[final_layer][evaluation.selected_index]);
                                size_t final_condensed_before = itty_feed_model_count_selected_condensed_votes (model,
                                                                                                                final_outputs,
                                                                                                                final_layer,
                                                                                                                evaluation.selected_index,
                                                                                                                target_bit_index,
                                                                                                                target_bit_capacity);
                                size_t final_condensed_after = itty_feed_model_count_selected_condensed_votes (model,
                                                                                                               after_final_outputs,
                                                                                                               final_layer,
                                                                                                               after_evaluation.selected_index,
                                                                                                               target_bit_index,
                                                                                                               target_bit_capacity);
                                size_t final_segment_before = itty_feed_model_count_selected_segment_votes (final_outputs,
                                                                                                            evaluation.selected_index,
                                                                                                            target_bit_index,
                                                                                                            target_bit_capacity);
                                size_t final_segment_after = itty_feed_model_count_selected_segment_votes (after_final_outputs,
                                                                                                           after_evaluation.selected_index,
                                                                                                           target_bit_index,
                                                                                                           target_bit_capacity);
                                bool decoded_before = itty_bit_string_get_bit (evaluation.folded_activation,
                                                                               target_bit_index);
                                bool decoded_after = itty_bit_string_get_bit (after_evaluation.folded_activation,
                                                                              target_bit_index);

                                trace.final_selected_node_before = evaluation.selected_index;
                                trace.final_selected_node_after = after_evaluation.selected_index;
                                trace.actual_final_condensed_ones_before = final_condensed_before;
                                trace.actual_final_condensed_ones_after = final_condensed_after;
                                trace.actual_final_segment_ones_before = final_segment_before;
                                trace.actual_final_segment_ones_after = final_segment_after;
                                trace.forced_node_false_negative_deficit_before = first_selected_before.false_negative_vote_deficit;
                                trace.forced_node_distance_before = first_selected_before.selected_distance;
                                trace.forced_node_false_negative_deficit_after = a_after_candidate_forced.false_negative_vote_deficit;
                                trace.forced_node_distance_after = a_after_candidate_forced.selected_distance;
                                trace.forced_node_false_positive_excess_before = first_selected_before.false_positive_vote_excess;
                                trace.forced_node_false_positive_excess_after = a_after_candidate_forced.false_positive_vote_excess;
                                trace.selected_node_false_negative_deficit_after = a_after_candidate_selected.false_negative_vote_deficit;
                                trace.selected_node_distance_after = a_after_candidate_selected.selected_distance;
                                trace.selected_node_false_positive_excess_after = a_after_candidate_selected.false_positive_vote_excess;
                                trace.replay_example_distance_after = b_after_candidate.selected_distance;
                                trace.replay_example_false_negative_deficit_after = b_after_candidate.false_negative_vote_deficit;
                                trace.replay_example_false_positive_excess_after = b_after_candidate.false_positive_vote_excess;
                                trace.decoded_before = decoded_before;
                                trace.decoded_after = decoded_after;

                                if (after_evaluation.selected_index != evaluation.selected_index)
                                        trace.propagation_failure = ITTY_FEED_MODEL_RESTORE_PROPAGATION_SELECTED_NODE_MISMATCH;
                                else if (trace.projected_condensed_bits < trace.clearable_segment_votes)
                                        trace.propagation_failure = ITTY_FEED_MODEL_RESTORE_PROPAGATION_DUPLICATE_CONDENSED_MAPPING;
                                else if (final_condensed_before == final_condensed_after)
                                        trace.propagation_failure = ITTY_FEED_MODEL_RESTORE_PROPAGATION_NO_MAJORITY_CROSSING;
                                else if (final_segment_before == final_segment_after)
                                        trace.propagation_failure = ITTY_FEED_MODEL_RESTORE_PROPAGATION_ROTATION_OR_EXPANSION_MISMATCH;
                                else if (final_segment_after > final_segment_before)
                                        trace.propagation_failure = ITTY_FEED_MODEL_RESTORE_PROPAGATION_WRONG_POLARITY;
                                else if (decoded_before == decoded_after)
                                        trace.propagation_failure = ITTY_FEED_MODEL_RESTORE_PROPAGATION_SEGMENT_CHANGED_DECODED_NOT_CROSSED;
                                else if (!useful)
                                        trace.propagation_failure = ITTY_FEED_MODEL_RESTORE_PROPAGATION_DECODED_CROSSED_OFFSET_ELSEWHERE;
                                else
                                        trace.propagation_failure = ITTY_FEED_MODEL_RESTORE_PROPAGATION_NONE;

                                for (size_t output_bit_index = target_bit_index;
                                     output_bit_index < activation_bit_capacity &&
                                     summary->clear_trace_count < ITTY_FEED_MODEL_RESTORE_CLEAR_TRACE_LIMIT;
                                     output_bit_index += target_bit_capacity) {
                                        bool before_raw = itty_bit_string_get_bit (selected_activation,
                                                                                   output_bit_index);
                                        if (!before_raw)
                                                continue;
                                        size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                                                    final_layer,
                                                                                                                    output_bit_index);
                                        size_t segment_index = output_bit_index / target_bit_capacity;
                                        bool after_raw = false;
                                        if (after_evaluation.selected_index < after_final_outputs->count) {
                                                itty_bit_string_t *after_selected = itty_bit_string_list_fetch (after_final_outputs,
                                                                                                                after_evaluation.selected_index);
                                                after_raw = itty_bit_string_get_bit (after_selected,
                                                                                    output_bit_index);
                                        }
                                        summary->clear_traces[summary->clear_trace_count++] =
                                                (itty_feed_model_restore_clear_vote_trace_t) {
                                                        .decoded_bit = target_bit_index,
                                                        .segment_index = segment_index,
                                                        .final_output_bit = output_bit_index,
                                                        .raw_final_segment_before = before_raw,
                                                        .desired_final_segment_value = false,
                                                        .raw_final_segment_after = after_raw,
                                                        .mapped_condensed_bit = condensed_bit_index,
                                                        .condensed_before = itty_bit_string_get_bit (selected_condensed,
                                                                                                     condensed_bit_index),
                                                        .desired_condensed_bit = itty_bit_string_get_bit (candidate_condensed,
                                                                                                          condensed_bit_index),
                                                        .condensed_after = after_selected_condensed ?
                                                                          itty_bit_string_get_bit (after_selected_condensed,
                                                                                                   condensed_bit_index) :
                                                                          itty_bit_string_get_bit (candidate_condensed,
                                                                                                   condensed_bit_index),
                                                        .mask_flip_count = repair_stats.flips,
                                                        .majority_ones_before = final_segment_before,
                                                        .majority_ones_after = final_segment_after,
                                                        .majority_threshold = threshold,
                                                        .condensed_bit_changed = itty_bit_string_get_bit (selected_condensed,
                                                                                                          condensed_bit_index) !=
                                                                                itty_bit_string_get_bit (candidate_condensed,
                                                                                                          condensed_bit_index),
                                                        .decoded_ones_before = final_segment_before,
                                                        .decoded_ones_after = final_segment_after,
                                                        .decoded_bit_before = decoded_before,
                                                        .decoded_bit_after = decoded_after,
                                                        .cleared = before_raw && !after_raw
                                                };
                                }
                                if (after_selected_condensed)
                                        itty_bit_string_free (after_selected_condensed);
                        }

                        if (after_final_outputs) {
                                itty_bit_string_free (after_evaluation.folded_activation);
                                itty_bit_string_list_free (after_final_outputs);
                        }
                        itty_feed_model_restore_layer_state_snapshot (model,
                                                                      final_layer,
                                                                      candidate_snapshot);
                }
                itty_bit_string_free (candidate_condensed);

                trace.projected_condensed_bits = projected_condensed_bits;
                trace.direct_candidate_changed = changed;
                trace.candidate_mask_flips = candidate_mask_flips;
                trace.replay_safe_candidates = safe_candidates;
                trace.accepted = accepted;
                if (trace.min_final_layer_mask_flips_needed == (size_t) -1)
                        trace.min_final_layer_mask_flips_needed = 0;

                if (accepted)
                        trace.rejection_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NONE;
                else if (trace.clearable_segment_votes == 0)
                        trace.rejection_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NO_CLEARABLE_SEGMENT_VOTES;
                else if (candidate_mask_flips == 0)
                        trace.rejection_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NO_MASK_PROJECTION;
                else if (useful_candidates == 0)
                        trace.rejection_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NO_USEFUL_REPAIRS;
                else if (safe_candidates == 0)
                        trace.rejection_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NO_B_SAFE_REPAIRS;
                else
                        trace.rejection_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NO_FLIPS_ACCEPTED;

                summary->traces[summary->trace_count++] = trace;
        }

        summary->a_rejected_repair_count = summary->a_candidate_repair_count - summary->a_useful_repair_count;
        if (!have_candidates)
                summary->no_flip_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NO_CANDIDATE_REPAIRS;
        else if (!have_useful_repairs)
                summary->no_flip_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NO_USEFUL_REPAIRS;
        else if (!have_b_safe_repairs)
                summary->no_flip_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NO_B_SAFE_REPAIRS;
        else if (!have_flipping_repairs)
                summary->no_flip_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NO_FLIPS_ACCEPTED;
        else
                summary->no_flip_reason = ITTY_FEED_MODEL_RESTORE_REJECTION_NONE;

        itty_bit_string_free (selected_condensed);
        itty_bit_string_free (evaluation.folded_activation);
        itty_bit_string_list_free (final_outputs);
        if (final_layer_input != first_input)
                itty_bit_string_list_free (final_layer_input);
        itty_feed_model_restore_layer_state_snapshot (model,
                                                      final_layer,
                                                      snapshot);
        return true;
}

bool
itty_feed_model_measure_final_layer_contender_restore (itty_feed_model_t                            *model,
                                                       itty_bit_string_list_t                       *first_input,
                                                       itty_bit_string_t                            *first_target,
                                                       itty_bit_string_list_t                       *second_input,
                                                       itty_bit_string_t                            *second_target,
                                                       itty_feed_model_train_options_t const       *options,
                                                       size_t                                        max_second_steps,
                                                       bool                                          preserve_second,
                                                       itty_feed_model_contender_restore_summary_t *summary)
{
        if (summary)
                *summary = (itty_feed_model_contender_restore_summary_t) { 0 };

        if (!summary ||
            !model ||
            !first_input ||
            !first_target ||
            !second_input ||
            !second_target ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (first_input) != model->inputs_per_node ||
            itty_bit_string_list_get_length (second_input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (first_target) > model->vocabulary_words ||
            itty_bit_string_get_number_of_words (second_target) > model->vocabulary_words ||
            !itty_feed_model_can_train_with_options (options))
                return false;

        size_t final_layer = model->number_of_layers - 1;
        itty_feed_model_layer_state_snapshot_t *snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                 final_layer);
        itty_feed_model_decoder_objective_t first_before = { 0 };
        itty_feed_model_decoder_objective_t first_after_b = { 0 };
        itty_feed_model_decoder_objective_t second_before = { 0 };
        itty_feed_model_decoder_objective_t second_after_b = { 0 };

        if (!itty_feed_model_measure_decoder_objective (model,
                                                        first_input,
                                                        first_target,
                                                        &first_before) ||
            !itty_feed_model_measure_decoder_objective (model,
                                                        second_input,
                                                        second_target,
                                                        &second_before)) {
                itty_feed_model_restore_layer_state_snapshot (model, final_layer, snapshot);
                return false;
        }

        first_after_b = first_before;
        second_after_b = second_before;
        for (size_t step = 0; step < max_second_steps; step++) {
                itty_feed_model_train_stats_t step_stats = { 0 };

                if (!itty_feed_model_train_final_layer_with_suffix_oracle (model,
                                                                           second_input,
                                                                           second_target,
                                                                           options,
                                                                           &step_stats))
                        break;

                if (!itty_feed_model_measure_decoder_objective (model,
                                                                first_input,
                                                                first_target,
                                                                &first_after_b) ||
                    !itty_feed_model_measure_decoder_objective (model,
                                                                second_input,
                                                                second_target,
                                                                &second_after_b)) {
                        itty_feed_model_restore_layer_state_snapshot (model, final_layer, snapshot);
                        return false;
                }

                if (first_after_b.selected_distance > first_before.selected_distance &&
                    (second_after_b.selected_distance < second_before.selected_distance ||
                     second_after_b.false_negative_vote_deficit < second_before.false_negative_vote_deficit))
                        break;
        }

        summary->a_distance_after_b = first_after_b.selected_distance;
        summary->a_false_positive_excess_after_b = first_after_b.false_positive_vote_excess;

        if (first_after_b.false_positive_vote_excess == 0) {
                itty_feed_model_restore_layer_state_snapshot (model, final_layer, snapshot);
                return true;
        }

        itty_feed_model_layer_state_snapshot_t *post_b_snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                        final_layer);
        itty_bit_string_list_t *final_layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                        first_input,
                                                                                        final_layer);
        itty_bit_string_list_t *final_outputs = itty_feed_model_run_layer (model,
                                                                           final_layer,
                                                                           final_layer_input);
        itty_feed_model_output_evaluation_t evaluation = { 0 };
        bool ok = final_outputs &&
                  itty_feed_model_evaluate_output (model,
                                                  final_outputs,
                                                  first_target,
                                                  &evaluation);
        if (!ok) {
            if (final_outputs)
                    itty_bit_string_list_free (final_outputs);
            if (final_layer_input != first_input)
                    itty_bit_string_list_free (final_layer_input);
            itty_feed_model_restore_layer_state_snapshot (model, final_layer, snapshot);
            return false;
        }

        size_t target_bit_capacity = itty_bit_string_get_number_of_words (first_target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        itty_bit_string_t *selected_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                    model->masks_by_layer_node[final_layer][evaluation.selected_index]);
        itty_feed_model_decoder_objective_t selected_before = { 0 };
        if (!selected_condensed ||
            !itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                  final_outputs,
                                                                  first_target,
                                                                  evaluation.selected_index,
                                                                  &selected_before)) {
                if (selected_condensed)
                        itty_bit_string_free (selected_condensed);
                itty_bit_string_free (evaluation.folded_activation);
                itty_bit_string_list_free (final_outputs);
                if (final_layer_input != first_input)
                        itty_bit_string_list_free (final_layer_input);
                itty_feed_model_restore_layer_state_snapshot (model, final_layer, snapshot);
                return false;
        }

        size_t target_bit_index = (size_t) -1;
        for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                bool target_bit = itty_bit_string_get_bit (first_target, bit_index);
                bool folded_bit = itty_bit_string_get_bit (evaluation.folded_activation, bit_index);
                if (!target_bit && folded_bit) {
                        target_bit_index = bit_index;
                        break;
                }
        }

        if (target_bit_index == (size_t) -1) {
                itty_bit_string_free (selected_condensed);
                itty_bit_string_free (evaluation.folded_activation);
                itty_bit_string_list_free (final_outputs);
                if (final_layer_input != first_input)
                        itty_bit_string_list_free (final_layer_input);
                itty_feed_model_restore_layer_state_snapshot (model, final_layer, snapshot);
                return true;
        }

        summary->decoded_bit = target_bit_index;
        summary->selected_node_before = evaluation.selected_index;
        summary->forced_node_distance_before = selected_before.selected_distance;
        summary->forced_node_false_positive_excess_before = selected_before.false_positive_vote_excess;

        itty_bit_string_t *primary_condensed = itty_feed_model_bit_string_clone_to_words (selected_condensed,
                                                                                           itty_bit_string_get_number_of_words (selected_condensed));
        bool primary_changed = primary_condensed &&
                               itty_feed_model_final_layer_apply_decoder_block (model,
                                                                                 target_bit_index,
                                                                                 false,
                                                                                 primary_condensed);
        if (primary_changed) {
                itty_feed_model_train_stats_t primary_stats = { 0 };
                itty_feed_model_decoder_objective_t a_after_primary = { 0 };
                itty_feed_model_decoder_objective_t a_after_primary_forced = { 0 };
                itty_feed_model_decoder_objective_t b_after_primary = { 0 };
                itty_bit_string_list_t *after_primary_outputs = NULL;
                itty_feed_model_output_evaluation_t after_primary_eval = { 0 };

                itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][evaluation.selected_index],
                                                      final_layer_input,
                                                      primary_condensed,
                                                      options,
                                                      &primary_stats);
                bool primary_measured = itty_feed_model_measure_decoder_objective (model,
                                                                                   first_input,
                                                                                   first_target,
                                                                                   &a_after_primary) &&
                                      itty_feed_model_measure_decoder_objective (model,
                                                                                 second_input,
                                                                                 second_target,
                                                                                 &b_after_primary) &&
                                      (after_primary_outputs = itty_feed_model_run_layer (model,
                                                                                           final_layer,
                                                                                           final_layer_input)) &&
                                      itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                                            after_primary_outputs,
                                                                                            first_target,
                                                                                            evaluation.selected_index,
                                                                                            &a_after_primary_forced) &&
                                      itty_feed_model_evaluate_output (model,
                                                                       after_primary_outputs,
                                                                       first_target,
                                                                       &after_primary_eval);

                if (primary_measured) {
                        summary->selected_node_after_first = after_primary_eval.selected_index;
                        summary->forced_node_distance_after_first = a_after_primary_forced.selected_distance;
                        summary->forced_node_false_positive_excess_after_first = a_after_primary_forced.false_positive_vote_excess;
                        summary->forced_useful = a_after_primary_forced.selected_distance < selected_before.selected_distance ||
                                                a_after_primary_forced.false_positive_vote_excess < selected_before.false_positive_vote_excess;

                        if (after_primary_eval.selected_index != evaluation.selected_index &&
                            after_primary_eval.selected_index < model->nodes_per_layer) {
                                itty_feed_model_restore_layer_state_snapshot (model, final_layer, post_b_snapshot);
                                post_b_snapshot = NULL;

                                itty_bit_string_t *contender_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                                              model->masks_by_layer_node[final_layer][after_primary_eval.selected_index]);
                                size_t contender_clear_output_bits[ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_VOTE_LIMIT] = { 0 };
                                size_t contender_clear_segments[ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_VOTE_LIMIT] = { 0 };
                                size_t contender_clear_count = 0;
                                size_t contender_votes_needed = 0;

                                if (contender_condensed &&
                                    itty_feed_model_get_segment_condense_vote_need (model,
                                                                                    target_bit_index,
                                                                                    false,
                                                                                    contender_condensed,
                                                                                    &contender_votes_needed)) {
                                        size_t condensed_bit_capacity = itty_bit_string_get_length (contender_condensed);
                                        size_t output_bit_capacity = condensed_bit_capacity * 2;
                                        size_t target_bit_capacity = model->vocabulary_words *
                                                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

                                        for (size_t output_bit_index = target_bit_index;
                                             output_bit_index < output_bit_capacity;
                                             output_bit_index += target_bit_capacity) {
                                                size_t condensed_bit_index = itty_feed_model_trace_output_bit_to_condensed (model,
                                                                                                                            final_layer,
                                                                                                                            output_bit_index);
                                                if (!itty_bit_string_get_bit (contender_condensed,
                                                                              condensed_bit_index))
                                                        continue;
                                                if (contender_clear_count < ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_VOTE_LIMIT) {
                                                        contender_clear_output_bits[contender_clear_count] = output_bit_index;
                                                        contender_clear_segments[contender_clear_count] = output_bit_index / target_bit_capacity;
                                                        contender_clear_count++;
                                                }
                                        }
                                }

                                if (contender_condensed &&
                                    contender_votes_needed > 0 &&
                                    contender_clear_count >= contender_votes_needed) {
                                        size_t indices[ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_VOTE_LIMIT] = { 0 };
                                        for (size_t index = 0; index < contender_votes_needed; index++)
                                                indices[index] = index;

                                        bool done = false;
                                        itty_feed_model_contender_clear_set_trace_t best_trace = { 0 };
                                        bool have_best_trace = false;

                                        while (!done) {
                                                itty_feed_model_layer_state_snapshot_t *candidate_snapshot =
                                                        itty_feed_model_snapshot_layer_state (model,
                                                                                             final_layer);
                                                itty_bit_string_t *primary_again =
                                                        itty_feed_model_bit_string_clone_to_words (selected_condensed,
                                                                                                   itty_bit_string_get_number_of_words (selected_condensed));
                                                itty_bit_string_t *contender_candidate =
                                                        itty_feed_model_bit_string_clone_to_words (contender_condensed,
                                                                                                   itty_bit_string_get_number_of_words (contender_condensed));
                                                itty_feed_model_contender_clear_set_trace_t trace = {
                                                        .clear_vote_count = contender_votes_needed,
                                                };

                                                for (size_t vote_index = 0; vote_index < contender_votes_needed; vote_index++) {
                                                        trace.segment_indices[vote_index] = contender_clear_segments[indices[vote_index]];
                                                        trace.final_output_bits[vote_index] = contender_clear_output_bits[indices[vote_index]];
                                                }

                                                if (primary_again &&
                                                    contender_candidate &&
                                                    itty_feed_model_final_layer_apply_decoder_block (model,
                                                                                                      target_bit_index,
                                                                                                      false,
                                                                                                      primary_again) &&
                                                    itty_feed_model_apply_final_output_clear_set (model,
                                                                                                  final_layer,
                                                                                                  contender_candidate,
                                                                                                  trace.final_output_bits,
                                                                                                  trace.clear_vote_count)) {
                                                        itty_feed_model_train_stats_t primary_again_stats = { 0 };
                                                        itty_feed_model_train_stats_t contender_stats = { 0 };
                                                        itty_feed_model_decoder_objective_t a_after_contender = { 0 };
                                                        itty_feed_model_decoder_objective_t b_after_contender = { 0 };
                                                        itty_feed_model_segment_node_selection_summary_t a_selection = { 0 };

                                                        itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][evaluation.selected_index],
                                                                                              final_layer_input,
                                                                                              primary_again,
                                                                                              options,
                                                                                              &primary_again_stats);
                                                        itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][after_primary_eval.selected_index],
                                                                                              final_layer_input,
                                                                                              contender_candidate,
                                                                                              options,
                                                                                              &contender_stats);

                                                        if (itty_feed_model_measure_decoder_objective (model,
                                                                                                       first_input,
                                                                                                       first_target,
                                                                                                       &a_after_contender) &&
                                                            itty_feed_model_measure_decoder_objective (model,
                                                                                                       second_input,
                                                                                                       second_target,
                                                                                                       &b_after_contender) &&
                                                            itty_feed_model_measure_segment_node_selection (model,
                                                                                                            first_input,
                                                                                                            first_target,
                                                                                                            &a_selection)) {
                                                                trace.a_distance_after_restore = a_after_contender.selected_distance;
                                                                trace.a_false_positive_excess_after_restore = a_after_contender.false_positive_vote_excess;
                                                                trace.b_distance_after_restore = b_after_contender.selected_distance;
                                                                trace.b_false_negative_deficit_after_restore = b_after_contender.false_negative_vote_deficit;
                                                                trace.b_false_positive_excess_after_restore = b_after_contender.false_positive_vote_excess;
                                                                trace.total_flips = primary_again_stats.flips + contender_stats.flips;
                                                                trace.selected_node_after_restore = a_after_contender.selected_node;
                                                                trace.selection_margin_after_restore = a_selection.popcount_gap;
                                                                trace.a_restored = a_after_contender.selected_distance <= first_before.selected_distance &&
                                                                                   a_after_contender.false_positive_vote_excess <= first_before.false_positive_vote_excess;
                                                                trace.strict_preserved =
                                                                        b_after_contender.selected_distance <= second_after_b.selected_distance &&
                                                                        b_after_contender.false_negative_vote_deficit <= second_after_b.false_negative_vote_deficit;
                                                                trace.distance_preserved =
                                                                        b_after_contender.selected_distance <= second_after_b.selected_distance;
                                                                trace.progress_preserved =
                                                                        b_after_contender.selected_distance <= second_before.selected_distance &&
                                                                        b_after_contender.false_negative_vote_deficit < second_before.false_negative_vote_deficit;
                                                                trace.no_regression =
                                                                        b_after_contender.selected_distance <= second_before.selected_distance &&
                                                                        b_after_contender.false_negative_vote_deficit <= second_before.false_negative_vote_deficit;

                                                                if (summary->clear_set_trace_count < ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_TRACE_LIMIT)
                                                                        summary->clear_set_traces[summary->clear_set_trace_count++] = trace;
                                                                if (!have_best_trace ||
                                                                    itty_feed_model_contender_clear_set_trace_is_better (&trace,
                                                                                                                                 &best_trace)) {
                                                                        best_trace = trace;
                                                                        have_best_trace = true;
                                                                }
                                                        }
                                                }

                                                if (primary_again)
                                                        itty_bit_string_free (primary_again);
                                                if (contender_candidate)
                                                        itty_bit_string_free (contender_candidate);
                                                itty_feed_model_restore_layer_state_snapshot (model,
                                                                                              final_layer,
                                                                                              candidate_snapshot);

                                                size_t advance = contender_votes_needed;
                                                while (advance > 0) {
                                                        advance--;
                                                        if (indices[advance] != contender_clear_count - contender_votes_needed + advance) {
                                                                indices[advance]++;
                                                                for (size_t reset = advance + 1; reset < contender_votes_needed; reset++)
                                                                        indices[reset] = indices[reset - 1] + 1;
                                                                break;
                                                        }
                                                }
                                                if (advance == 0 &&
                                                    indices[0] == contender_clear_count - contender_votes_needed)
                                                        done = true;
                                        }

                                        if (have_best_trace) {
                                                for (size_t trace_index = 0; trace_index < summary->clear_set_trace_count; trace_index++) {
                                                        itty_feed_model_contender_clear_set_trace_t *stored =
                                                                &summary->clear_set_traces[trace_index];
                                                        bool same_set = stored->clear_vote_count == best_trace.clear_vote_count;
                                                        for (size_t vote_index = 0; same_set && vote_index < stored->clear_vote_count; vote_index++)
                                                                same_set = stored->segment_indices[vote_index] == best_trace.segment_indices[vote_index];
                                                        if (same_set)
                                                                stored->chosen_best = true;
                                                }

                                                summary->contender_node = after_primary_eval.selected_index;
                                                summary->a_distance_after_contender = best_trace.a_distance_after_restore;
                                                summary->a_false_positive_excess_after_contender = best_trace.a_false_positive_excess_after_restore;
                                                summary->b_distance_after_contender = best_trace.b_distance_after_restore;
                                                summary->b_false_negative_deficit_after_contender = best_trace.b_false_negative_deficit_after_restore;
                                                summary->total_flips = best_trace.total_flips;
                                                summary->contender_useful = best_trace.a_restored;
                                                summary->b_strict_preserved = best_trace.strict_preserved;
                                                summary->b_distance_preserved = best_trace.distance_preserved;
                                                summary->b_progress_preserved = best_trace.progress_preserved;
                                                summary->b_no_regression = best_trace.no_regression;
                                                summary->contender_b_safe = !preserve_second ||
                                                                           (best_trace.b_distance_after_restore <= second_after_b.selected_distance &&
                                                                            best_trace.b_false_negative_deficit_after_restore <= second_after_b.false_negative_vote_deficit &&
                                                                            best_trace.b_false_positive_excess_after_restore <= second_after_b.false_positive_vote_excess);
                                                summary->contender_accepted = summary->contender_useful &&
                                                                              summary->contender_b_safe &&
                                                                              summary->total_flips > 0;
                                        }
                                }
                                if (contender_condensed)
                                        itty_bit_string_free (contender_condensed);
                        }
                }

                if (after_primary_outputs) {
                        itty_bit_string_free (after_primary_eval.folded_activation);
                        itty_bit_string_list_free (after_primary_outputs);
                }
                if (post_b_snapshot) {
                        itty_feed_model_restore_layer_state_snapshot (model, final_layer, post_b_snapshot);
                        post_b_snapshot = NULL;
                }
        }

        if (primary_condensed)
                itty_bit_string_free (primary_condensed);
        itty_bit_string_free (selected_condensed);
        itty_bit_string_free (evaluation.folded_activation);
        itty_bit_string_list_free (final_outputs);
        if (final_layer_input != first_input)
                itty_bit_string_list_free (final_layer_input);
        itty_feed_model_restore_layer_state_snapshot (model, final_layer, snapshot);
        return true;
}

static bool
itty_feed_model_apply_contender_restore_current_state (itty_feed_model_t                           *model,
                                                       itty_bit_string_list_t                      *first_input,
                                                       itty_bit_string_t                           *first_target,
                                                       itty_bit_string_list_t                      *second_input,
                                                       itty_bit_string_t                           *second_target,
                                                       itty_feed_model_train_options_t const      *options,
                                                       bool                                         preserve_second,
                                                       itty_feed_model_contender_restore_summary_t *summary)
{
        if (summary)
                *summary = (itty_feed_model_contender_restore_summary_t) { 0 };

        size_t final_layer = model->number_of_layers - 1;
        itty_feed_model_decoder_objective_t first_before = { 0 };
        itty_feed_model_decoder_objective_t second_before = { 0 };

        if (!itty_feed_model_measure_decoder_objective (model, first_input, first_target, &first_before) ||
            !itty_feed_model_measure_decoder_objective (model, second_input, second_target, &second_before))
                return false;

        if (summary) {
                summary->a_distance_after_b = first_before.selected_distance;
                summary->a_false_positive_excess_after_b = first_before.false_positive_vote_excess;
        }

        if (first_before.false_positive_vote_excess == 0)
                return true;

        itty_feed_model_layer_state_snapshot_t *post_b_snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                        final_layer);
        itty_bit_string_list_t *final_layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                        first_input,
                                                                                        final_layer);
        itty_bit_string_list_t *final_outputs = itty_feed_model_run_layer (model,
                                                                           final_layer,
                                                                           final_layer_input);
        itty_feed_model_output_evaluation_t evaluation = { 0 };
        bool ok = final_outputs &&
                  itty_feed_model_evaluate_output (model, final_outputs, first_target, &evaluation);
        if (!ok) {
                if (final_outputs)
                        itty_bit_string_list_free (final_outputs);
                if (final_layer_input != first_input)
                        itty_bit_string_list_free (final_layer_input);
                itty_feed_model_restore_layer_state_snapshot (model, final_layer, post_b_snapshot);
                return false;
        }

        size_t target_bit_capacity = itty_bit_string_get_number_of_words (first_target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        itty_bit_string_t *selected_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                    model->masks_by_layer_node[final_layer][evaluation.selected_index]);
        itty_feed_model_decoder_objective_t selected_before = { 0 };
        if (!selected_condensed ||
            !itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                  final_outputs,
                                                                  first_target,
                                                                  evaluation.selected_index,
                                                                  &selected_before)) {
                if (selected_condensed)
                        itty_bit_string_free (selected_condensed);
                itty_bit_string_free (evaluation.folded_activation);
                itty_bit_string_list_free (final_outputs);
                if (final_layer_input != first_input)
                        itty_bit_string_list_free (final_layer_input);
                itty_feed_model_restore_layer_state_snapshot (model, final_layer, post_b_snapshot);
                return false;
        }

        size_t target_bit_index = (size_t) -1;
        for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                bool target_bit = itty_bit_string_get_bit (first_target, bit_index);
                bool folded_bit = itty_bit_string_get_bit (evaluation.folded_activation, bit_index);
                if (!target_bit && folded_bit) {
                        target_bit_index = bit_index;
                        break;
                }
        }

        if (target_bit_index == (size_t) -1) {
                itty_bit_string_free (selected_condensed);
                itty_bit_string_free (evaluation.folded_activation);
                itty_bit_string_list_free (final_outputs);
                if (final_layer_input != first_input)
                        itty_bit_string_list_free (final_layer_input);
                itty_feed_model_restore_layer_state_snapshot (model, final_layer, post_b_snapshot);
                return true;
        }

        if (summary) {
                summary->decoded_bit = target_bit_index;
                summary->selected_node_before = evaluation.selected_index;
                summary->forced_node_distance_before = selected_before.selected_distance;
                summary->forced_node_false_positive_excess_before = selected_before.false_positive_vote_excess;
        }

        itty_bit_string_t *primary_condensed = itty_feed_model_bit_string_clone_to_words (selected_condensed,
                                                                                           itty_bit_string_get_number_of_words (selected_condensed));
        bool primary_changed = primary_condensed &&
                               itty_feed_model_final_layer_apply_decoder_block (model,
                                                                                 target_bit_index,
                                                                                 false,
                                                                                 primary_condensed);
        bool accepted = false;

        if (primary_changed) {
                itty_feed_model_train_stats_t primary_stats = { 0 };
                itty_feed_model_decoder_objective_t a_after_primary_forced = { 0 };
                itty_feed_model_decoder_objective_t b_after_primary = { 0 };
                itty_bit_string_list_t *after_primary_outputs = NULL;
                itty_feed_model_output_evaluation_t after_primary_eval = { 0 };

                itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][evaluation.selected_index],
                                                      final_layer_input,
                                                      primary_condensed,
                                                      options,
                                                      &primary_stats);
                bool primary_measured =
                        itty_feed_model_measure_decoder_objective (model, second_input, second_target, &b_after_primary) &&
                        (after_primary_outputs = itty_feed_model_run_layer (model, final_layer, final_layer_input)) &&
                        itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                             after_primary_outputs,
                                                                             first_target,
                                                                             evaluation.selected_index,
                                                                             &a_after_primary_forced) &&
                        itty_feed_model_evaluate_output (model, after_primary_outputs, first_target, &after_primary_eval);

                if (primary_measured && summary) {
                        summary->selected_node_after_first = after_primary_eval.selected_index;
                        summary->forced_node_distance_after_first = a_after_primary_forced.selected_distance;
                        summary->forced_node_false_positive_excess_after_first = a_after_primary_forced.false_positive_vote_excess;
                        summary->forced_useful = a_after_primary_forced.selected_distance < selected_before.selected_distance ||
                                                a_after_primary_forced.false_positive_vote_excess < selected_before.false_positive_vote_excess;
                }

                if (primary_measured &&
                    after_primary_eval.selected_index != evaluation.selected_index &&
                    after_primary_eval.selected_index < model->nodes_per_layer) {
                        itty_feed_model_restore_layer_state_snapshot (model, final_layer, post_b_snapshot);
                        post_b_snapshot = NULL;

                        itty_feed_model_layer_state_snapshot_t *candidate_snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                                           final_layer);
                        itty_bit_string_t *primary_again = itty_feed_model_bit_string_clone_to_words (selected_condensed,
                                                                                                       itty_bit_string_get_number_of_words (selected_condensed));
                        itty_bit_string_t *contender_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                                      model->masks_by_layer_node[final_layer][after_primary_eval.selected_index]);
                        if (primary_again &&
                            contender_condensed &&
                            itty_feed_model_final_layer_apply_decoder_block (model, target_bit_index, false, primary_again) &&
                            itty_feed_model_final_layer_apply_decoder_block (model, target_bit_index, false, contender_condensed)) {
                                itty_feed_model_train_stats_t primary_again_stats = { 0 };
                                itty_feed_model_train_stats_t contender_stats = { 0 };
                                itty_feed_model_decoder_objective_t a_after_contender = { 0 };
                                itty_feed_model_decoder_objective_t b_after_contender = { 0 };

                                itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][evaluation.selected_index],
                                                                      final_layer_input,
                                                                      primary_again,
                                                                      options,
                                                                      &primary_again_stats);
                                itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][after_primary_eval.selected_index],
                                                                      final_layer_input,
                                                                      contender_condensed,
                                                                      options,
                                                                      &contender_stats);

                                if (itty_feed_model_measure_decoder_objective (model, first_input, first_target, &a_after_contender) &&
                                    itty_feed_model_measure_decoder_objective (model, second_input, second_target, &b_after_contender)) {
                                        if (summary) {
                                                summary->contender_node = after_primary_eval.selected_index;
                                                summary->a_distance_after_contender = a_after_contender.selected_distance;
                                                summary->a_false_positive_excess_after_contender = a_after_contender.false_positive_vote_excess;
                                                summary->b_distance_after_contender = b_after_contender.selected_distance;
                                                summary->b_false_negative_deficit_after_contender = b_after_contender.false_negative_vote_deficit;
                                                summary->total_flips = primary_again_stats.flips + contender_stats.flips;
                                                summary->contender_useful = a_after_contender.selected_distance <= first_before.selected_distance &&
                                                                           a_after_contender.false_positive_vote_excess <= first_before.false_positive_vote_excess;
                                                summary->contender_b_safe = !preserve_second ||
                                                                           (b_after_contender.selected_distance <= second_before.selected_distance &&
                                                                            b_after_contender.false_negative_vote_deficit < second_before.false_negative_vote_deficit);
                                                summary->b_strict_preserved =
                                                        b_after_contender.selected_distance <= second_before.selected_distance &&
                                                        b_after_contender.false_negative_vote_deficit <= second_before.false_negative_vote_deficit;
                                                summary->b_distance_preserved =
                                                        b_after_contender.selected_distance <= second_before.selected_distance;
                                                summary->b_progress_preserved =
                                                        b_after_contender.selected_distance <= second_before.selected_distance &&
                                                        b_after_contender.false_negative_vote_deficit < second_before.false_negative_vote_deficit;
                                                summary->b_no_regression =
                                                        b_after_contender.selected_distance <= second_before.selected_distance &&
                                                        b_after_contender.false_negative_vote_deficit <= second_before.false_negative_vote_deficit;
                                                summary->contender_accepted = summary->contender_useful &&
                                                                              summary->contender_b_safe &&
                                                                              summary->total_flips > 0;
                                                accepted = summary->contender_accepted;
                                        }
                                }
                        }

                        if (!accepted)
                                itty_feed_model_restore_layer_state_snapshot (model, final_layer, candidate_snapshot);
                        else
                                itty_feed_model_free_layer_state_snapshot (model, candidate_snapshot);

                        if (primary_again)
                                itty_bit_string_free (primary_again);
                        if (contender_condensed)
                                itty_bit_string_free (contender_condensed);
                } else {
                        itty_feed_model_restore_layer_state_snapshot (model, final_layer, post_b_snapshot);
                        post_b_snapshot = NULL;
                }

                if (after_primary_outputs) {
                        itty_bit_string_free (after_primary_eval.folded_activation);
                        itty_bit_string_list_free (after_primary_outputs);
                }
        }

        if (post_b_snapshot)
                itty_feed_model_restore_layer_state_snapshot (model, final_layer, post_b_snapshot);
        if (primary_condensed)
                itty_bit_string_free (primary_condensed);
        itty_bit_string_free (selected_condensed);
        itty_bit_string_free (evaluation.folded_activation);
        itty_bit_string_list_free (final_outputs);
        if (final_layer_input != first_input)
                itty_bit_string_list_free (final_layer_input);
        return accepted;
}

bool
itty_feed_model_train_final_layer_transaction_scaffold (itty_feed_model_t                               *model,
                                                        itty_bit_string_list_t                          *first_input,
                                                        itty_bit_string_t                               *first_target,
                                                        itty_bit_string_list_t                          *second_input,
                                                        itty_bit_string_t                               *second_target,
                                                        itty_feed_model_train_options_t const          *options,
                                                        size_t                                           max_rounds,
                                                        itty_feed_model_transaction_scaffold_round_t   *trajectory,
                                                        size_t                                           trajectory_count,
                                                        itty_feed_model_transaction_scaffold_summary_t *summary)
{
        size_t frontier_top_k = 4;
        size_t finish_nearest_deficit_one_threshold = 4;

        if (summary)
                *summary = (itty_feed_model_transaction_scaffold_summary_t) { 0 };

        if (!summary ||
            !model ||
            !first_input ||
            !first_target ||
            !second_input ||
            !second_target ||
            model->number_of_layers == 0 ||
            !itty_feed_model_can_train_with_options (options))
                return false;

        size_t final_layer = model->number_of_layers - 1;
        itty_feed_model_measure_decoder_objective (model, first_input, first_target, &(itty_feed_model_decoder_objective_t){0});
        itty_feed_model_decoder_objective_t a_before = { 0 };
        itty_feed_model_decoder_objective_t b_before = { 0 };
        itty_feed_model_completion_frontier_t b_frontier_before = { 0 };
        if (!itty_feed_model_measure_decoder_objective (model, first_input, first_target, &a_before) ||
            !itty_feed_model_measure_decoder_objective (model, second_input, second_target, &b_before))
                return false;
        itty_feed_model_measure_completion_frontier (&b_before,
                                                     frontier_top_k,
                                                     &b_frontier_before);

        summary->a_distance_before = a_before.selected_distance;
        summary->a_false_positive_excess_before = a_before.false_positive_vote_excess;
        summary->b_distance_before = b_before.selected_distance;
        summary->b_false_negative_deficit_before = b_before.false_negative_vote_deficit;
        summary->finish_margin_required = options->finish_margin;
        summary->b_min_deficit_before = b_frontier_before.min_deficit;
        summary->b_deficit_one_bits_before = b_frontier_before.deficit_one_bits;
        summary->b_deficit_two_bits_before = b_frontier_before.deficit_two_bits;
        summary->b_cheapest_completion_cost_before = b_frontier_before.cheapest_completion_cost;
        summary->b_top_k_completion_cost_before = b_frontier_before.top_k_completion_cost;

        for (size_t round = 0; round < max_rounds; round++) {
                itty_feed_model_layer_state_snapshot_t *round_snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                               final_layer);
                itty_feed_model_decoder_objective_t a_round_before = { 0 };
                itty_feed_model_decoder_objective_t b_round_before = { 0 };
                itty_feed_model_completion_frontier_t b_round_frontier_before = { 0 };
                itty_feed_model_transaction_scaffold_round_t round_trace = {
                        .round_index = round,
                };

                if (!itty_feed_model_measure_decoder_objective (model, first_input, first_target, &a_round_before) ||
                    !itty_feed_model_measure_decoder_objective (model, second_input, second_target, &b_round_before)) {
                        itty_feed_model_restore_layer_state_snapshot (model, final_layer, round_snapshot);
                        return false;
                }
                itty_feed_model_measure_completion_frontier (&b_round_before,
                                                             frontier_top_k,
                                                             &b_round_frontier_before);
                round_trace.a_distance_before = a_round_before.selected_distance;
                round_trace.a_false_positive_excess_before = a_round_before.false_positive_vote_excess;
                round_trace.finish_nearest_threshold = finish_nearest_deficit_one_threshold;
                round_trace.b_distance_before = b_round_before.selected_distance;
                round_trace.b_false_negative_deficit_before = b_round_before.false_negative_vote_deficit;
                round_trace.b_min_deficit_before = b_round_frontier_before.min_deficit;
                round_trace.b_deficit_one_bits_before = b_round_frontier_before.deficit_one_bits;
                round_trace.b_deficit_two_bits_before = b_round_frontier_before.deficit_two_bits;
                round_trace.b_cheapest_completion_cost_before = b_round_frontier_before.cheapest_completion_cost;
                round_trace.b_top_k_completion_cost_before = b_round_frontier_before.top_k_completion_cost;

                summary->rounds_attempted++;
                bool attempted_finish_nearest_bit = b_round_frontier_before.deficit_one_bits >=
                                                    finish_nearest_deficit_one_threshold;
                round_trace.used_finish_nearest_bit = attempted_finish_nearest_bit;
                bool accepted = false;
                size_t restore_flips = 0;
                bool frontier_improved = false;
                size_t accepted_b_flips = 0;

                if (attempted_finish_nearest_bit) {
                        itty_feed_model_finish_candidate_trace_t finish_trace = { 0 };
                        accepted = itty_feed_model_try_best_transaction_completion_candidate (model,
                                                                                              first_input,
                                                                                              first_target,
                                                                                              second_input,
                                                                                              second_target,
                                                                                              options,
                                                                                              &a_round_before,
                                                                                              &b_round_before,
                                                                                              &b_round_frontier_before,
                                                                                              &round_trace,
                                                                                              summary,
                                                                                              &finish_trace);
                        if (accepted) {
                                restore_flips = round_trace.restore_flips;
                                frontier_improved = round_trace.frontier_improved;
                                accepted_b_flips = round_trace.b_flips;
                        } else {
                                round_trace.used_finish_nearest_bit = true;
                                itty_feed_model_restore_layer_state_snapshot (model, final_layer, round_snapshot);
                                round_snapshot = itty_feed_model_snapshot_layer_state (model, final_layer);
                                attempted_finish_nearest_bit = false;
                        }
                }
                for (size_t phase = 0; phase < 1 && !accepted; phase++) {
                        itty_feed_model_train_stats_t b_step = { 0 };
                        itty_feed_model_contender_restore_summary_t contender = { 0 };
                        itty_feed_model_decoder_objective_t a_round_after = { 0 };
                        itty_feed_model_decoder_objective_t b_round_after = { 0 };
                        itty_feed_model_completion_frontier_t b_round_frontier_after = { 0 };
                        bool trained_b = itty_feed_model_train_final_layer_with_suffix_oracle (model,
                                                                                                 second_input,
                                                                                                 second_target,
                                                                                                 options,
                                                                                                 &b_step);

                        if (!trained_b) {
                                continue;
                        }

                        if (!itty_feed_model_measure_decoder_objective (model, first_input, first_target, &a_round_after) ||
                            !itty_feed_model_measure_decoder_objective (model, second_input, second_target, &b_round_after)) {
                                itty_feed_model_restore_layer_state_snapshot (model, final_layer, round_snapshot);
                                return false;
                        }

                        bool phase_accepted = false;
                        size_t phase_restore_flips = 0;
                        bool phase_frontier_improved = false;
                        bool phase_used_restore = false;
                        bool phase_strict = false;
                        bool phase_dist = false;
                        bool phase_progress = false;
                        bool phase_no_reg = false;

                        if (a_round_after.selected_distance <= a_round_before.selected_distance &&
                            a_round_after.false_positive_vote_excess <= a_round_before.false_positive_vote_excess) {
                                itty_feed_model_measure_completion_frontier (&b_round_after,
                                                                             frontier_top_k,
                                                                             &b_round_frontier_after);
                                phase_frontier_improved =
                                        itty_feed_model_completion_frontier_is_better (&b_round_frontier_after,
                                                                                      &b_round_frontier_before);
                                phase_accepted = b_round_after.selected_distance <= b_round_before.selected_distance &&
                                                 (b_round_after.selected_distance < b_round_before.selected_distance ||
                                                  b_round_after.false_negative_vote_deficit < b_round_before.false_negative_vote_deficit ||
                                                  phase_frontier_improved);
                                phase_used_restore = false;
                                phase_strict = true;
                                phase_dist = true;
                                phase_progress = true;
                                phase_no_reg = true;
                        } else {
                                itty_feed_model_layer_state_snapshot_t *post_b_snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                                                final_layer);
                                phase_accepted = itty_feed_model_apply_contender_restore_current_state (model,
                                                                                                        first_input,
                                                                                                        first_target,
                                                                                                        second_input,
                                                                                                        second_target,
                                                                                                        options,
                                                                                                        false,
                                                                                                        &contender);
                                phase_used_restore = true;
                                if (!phase_accepted) {
                                        itty_feed_model_restore_layer_state_snapshot (model, final_layer, post_b_snapshot);
                                } else {
                                        itty_feed_model_decoder_objective_t b_after_restore = { 0 };
                                        if (!itty_feed_model_measure_decoder_objective (model,
                                                                                        second_input,
                                                                                        second_target,
                                                                                        &b_after_restore)) {
                                                itty_feed_model_free_layer_state_snapshot (model, post_b_snapshot);
                                                itty_feed_model_restore_layer_state_snapshot (model, final_layer, round_snapshot);
                                                return false;
                                        }
                                        itty_feed_model_measure_completion_frontier (&b_after_restore,
                                                                                     frontier_top_k,
                                                                                     &b_round_frontier_after);
                                        phase_frontier_improved =
                                                itty_feed_model_completion_frontier_is_better (&b_round_frontier_after,
                                                                                              &b_round_frontier_before);
                                        phase_accepted = contender.contender_accepted &&
                                                         contender.b_no_regression &&
                                                         (b_after_restore.selected_distance < b_round_before.selected_distance ||
                                                          b_after_restore.false_negative_vote_deficit < b_round_before.false_negative_vote_deficit ||
                                                          phase_frontier_improved);
                                        phase_restore_flips = contender.total_flips;
                                        phase_strict = contender.b_strict_preserved;
                                        phase_dist = contender.b_distance_preserved;
                                        phase_progress = contender.b_progress_preserved;
                                        phase_no_reg = contender.b_no_regression;
                                        if (!phase_accepted)
                                                itty_feed_model_restore_layer_state_snapshot (model, final_layer, post_b_snapshot);
                                        else
                                                itty_feed_model_free_layer_state_snapshot (model, post_b_snapshot);
                                }
                        }

                        if (phase_accepted) {
                                itty_feed_model_decoder_objective_t kept_a = { 0 };
                                itty_feed_model_decoder_objective_t kept_b = { 0 };
                                itty_feed_model_completion_frontier_t kept_frontier = { 0 };
                                if (!itty_feed_model_measure_decoder_objective (model,
                                                                                first_input,
                                                                                first_target,
                                                                                &kept_a) ||
                                    !itty_feed_model_measure_decoder_objective (model,
                                                                                second_input,
                                                                                second_target,
                                                                                &kept_b)) {
                                        itty_feed_model_restore_layer_state_snapshot (model, final_layer, round_snapshot);
                                        return false;
                                }
                                itty_feed_model_measure_completion_frontier (&kept_b,
                                                                             frontier_top_k,
                                                                             &kept_frontier);
                                accepted = true;
                                restore_flips = phase_restore_flips;
                                accepted_b_flips = b_step.flips;
                                frontier_improved = phase_frontier_improved;
                                round_trace.used_restore = phase_used_restore;
                                round_trace.used_finish_nearest_bit = false;
                                round_trace.strict_preserved = phase_strict;
                                round_trace.distance_preserved = phase_dist;
                                round_trace.progress_preserved = phase_progress;
                                round_trace.no_regression = phase_no_reg;
                                round_trace.accepted = true;
                                round_trace.a_distance_after = kept_a.selected_distance;
                                round_trace.a_false_positive_excess_after = kept_a.false_positive_vote_excess;
                                round_trace.b_distance_after = kept_b.selected_distance;
                                round_trace.b_false_negative_deficit_after = kept_b.false_negative_vote_deficit;
                                round_trace.b_min_deficit_after = kept_frontier.min_deficit;
                                round_trace.b_deficit_one_bits_after = kept_frontier.deficit_one_bits;
                                round_trace.b_deficit_two_bits_after = kept_frontier.deficit_two_bits;
                                round_trace.b_cheapest_completion_cost_after = kept_frontier.cheapest_completion_cost;
                                round_trace.b_top_k_completion_cost_after = kept_frontier.top_k_completion_cost;
                                round_trace.b_flips = b_step.flips;
                                round_trace.restore_flips = restore_flips;
                                round_trace.frontier_improved = frontier_improved;
                        }
                }

                if (!accepted) {
                        round_trace.accepted = false;
                        round_trace.a_distance_after = a_round_before.selected_distance;
                        round_trace.a_false_positive_excess_after = a_round_before.false_positive_vote_excess;
                        round_trace.b_distance_after = b_round_before.selected_distance;
                        round_trace.b_false_negative_deficit_after = b_round_before.false_negative_vote_deficit;
                        round_trace.b_min_deficit_after = b_round_frontier_before.min_deficit;
                        round_trace.b_deficit_one_bits_after = b_round_frontier_before.deficit_one_bits;
                        round_trace.b_deficit_two_bits_after = b_round_frontier_before.deficit_two_bits;
                        round_trace.b_cheapest_completion_cost_after = b_round_frontier_before.cheapest_completion_cost;
                        round_trace.b_top_k_completion_cost_after = b_round_frontier_before.top_k_completion_cost;
                        round_trace.b_flips = 0;
                        round_trace.restore_flips = restore_flips;
                        round_trace.frontier_improved = false;
                        itty_feed_model_restore_layer_state_snapshot (model, final_layer, round_snapshot);
                        if (round < trajectory_count)
                                trajectory[round] = round_trace;
                        break;
                }

                {
                        itty_feed_model_decoder_objective_t a_round_kept = { 0 };
                        itty_feed_model_decoder_objective_t b_round_kept = { 0 };
                        itty_feed_model_completion_frontier_t b_round_frontier_kept = { 0 };
                        if (!itty_feed_model_measure_decoder_objective (model, first_input, first_target, &a_round_kept) ||
                            !itty_feed_model_measure_decoder_objective (model, second_input, second_target, &b_round_kept)) {
                                itty_feed_model_restore_layer_state_snapshot (model, final_layer, round_snapshot);
                                return false;
                        }
                        itty_feed_model_measure_completion_frontier (&b_round_kept,
                                                                     frontier_top_k,
                                                                     &b_round_frontier_kept);
                        round_trace.accepted = true;
                        round_trace.a_distance_after = a_round_kept.selected_distance;
                        round_trace.a_false_positive_excess_after = a_round_kept.false_positive_vote_excess;
                        round_trace.b_distance_after = b_round_kept.selected_distance;
                        round_trace.b_false_negative_deficit_after = b_round_kept.false_negative_vote_deficit;
                        round_trace.b_min_deficit_after = b_round_frontier_kept.min_deficit;
                        round_trace.b_deficit_one_bits_after = b_round_frontier_kept.deficit_one_bits;
                        round_trace.b_deficit_two_bits_after = b_round_frontier_kept.deficit_two_bits;
                        round_trace.b_cheapest_completion_cost_after = b_round_frontier_kept.cheapest_completion_cost;
                        round_trace.b_top_k_completion_cost_after = b_round_frontier_kept.top_k_completion_cost;
                        round_trace.b_flips = accepted_b_flips;
                        round_trace.restore_flips = restore_flips;
                        round_trace.frontier_improved = frontier_improved;
                }
                itty_feed_model_free_layer_state_snapshot (model, round_snapshot);
                summary->rounds_accepted++;
                summary->total_b_flips += round_trace.b_flips;
                summary->total_restore_flips += round_trace.restore_flips;
                if (round_trace.frontier_improved)
                        summary->frontier_improved_rounds++;
                if (round_trace.strict_preserved)
                        summary->strict_preserved_rounds++;
                if (round_trace.distance_preserved)
                        summary->distance_preserved_rounds++;
                if (round_trace.progress_preserved)
                        summary->progress_preserved_rounds++;
                if (round_trace.no_regression)
                        summary->no_regression_rounds++;
                if (round < trajectory_count)
                        trajectory[round] = round_trace;
        }

        {
                itty_feed_model_decoder_objective_t a_after = { 0 };
                itty_feed_model_decoder_objective_t b_after = { 0 };
                if (!itty_feed_model_measure_decoder_objective (model, first_input, first_target, &a_after) ||
                    !itty_feed_model_measure_decoder_objective (model, second_input, second_target, &b_after))
                        return false;

                summary->a_distance_after = a_after.selected_distance;
                summary->a_false_positive_excess_after = a_after.false_positive_vote_excess;
                summary->b_distance_after = b_after.selected_distance;
                summary->b_false_negative_deficit_after = b_after.false_negative_vote_deficit;
                {
                        itty_feed_model_completion_frontier_t b_frontier_after = { 0 };
                        itty_feed_model_measure_completion_frontier (&b_after,
                                                                     frontier_top_k,
                                                                     &b_frontier_after);
                        summary->b_min_deficit_after = b_frontier_after.min_deficit;
                        summary->b_deficit_one_bits_after = b_frontier_after.deficit_one_bits;
                        summary->b_deficit_two_bits_after = b_frontier_after.deficit_two_bits;
                        summary->b_cheapest_completion_cost_after = b_frontier_after.cheapest_completion_cost;
                        summary->b_top_k_completion_cost_after = b_frontier_after.top_k_completion_cost;
                        summary->b_frontier_improved =
                                itty_feed_model_completion_frontier_is_better (&b_frontier_after,
                                                                              &b_frontier_before);
                }
                summary->b_distance_improved = b_after.selected_distance < b_before.selected_distance;
                summary->b_deficit_improved = b_after.false_negative_vote_deficit < b_before.false_negative_vote_deficit;
                summary->a_remains_solved = a_after.selected_distance <= a_before.selected_distance &&
                                            a_after.false_positive_vote_excess <= a_before.false_positive_vote_excess;
        }

        return true;
}

itty_network_t *
itty_feed_model_build_network (itty_feed_model_t *model)
{
        itty_network_t *network = itty_network_new ();

        for (size_t layer_index = 0; layer_index < model->number_of_layers; layer_index++) {
                itty_network_layer_t *layer = itty_network_layer_new ();

                for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                        itty_bit_string_list_t *model_masks = model->masks_by_layer_node[layer_index][node_index];
                        itty_bit_string_list_t *network_masks = itty_bit_string_list_new ();

                        for (size_t input_index = 0; input_index < model_masks->count; input_index++)
                                itty_bit_string_list_append (network_masks,
                                                             itty_feed_model_bit_string_clone (model_masks->bit_strings[input_index]));

                        itty_network_layer_append (layer,
                                                   itty_network_feed_node_new_with_rotation (network_masks,
                                                                                            model->rotations_by_layer[layer_index]));
                }

                itty_network_append (network, layer);
        }

        return network;
}

bool
itty_feed_model_train_one (itty_feed_model_t      *model,
                           itty_bit_string_list_t *input,
                           itty_bit_string_t      *target)
{
        return itty_feed_model_train_one_with_stats (model, input, target, NULL, NULL);
}

bool
itty_feed_model_train_one_with_options (itty_feed_model_t                     *model,
                                        itty_bit_string_list_t                *input,
                                        itty_bit_string_t                     *target,
                                        itty_feed_model_train_options_t const *options)
{
        return itty_feed_model_train_one_with_stats (model, input, target, options, NULL);
}

bool
itty_feed_model_train_one_with_stats (itty_feed_model_t                     *model,
                                      itty_bit_string_list_t                *input,
                                      itty_bit_string_t                     *target,
                                      itty_feed_model_train_options_t const *options,
                                      itty_feed_model_train_stats_t         *stats)
{
        if (stats)
                *stats = (itty_feed_model_train_stats_t) { 0 };

        if (model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;
        if (!itty_feed_model_can_train_with_options (options))
                return false;

        size_t final_layer = model->number_of_layers - 1;
        itty_bit_string_list_t *final_layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                        input,
                                                                                        final_layer);
        itty_bit_string_t *expanded_target = itty_feed_model_expand_target_for_layer (target,
                                                                                      final_layer);
        bool trained = true;

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_feed_model_train_stats_t current_node_stats = { 0 };

                if (!itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][node_index],
                                                           final_layer_input,
                                                           expanded_target,
                                                           options,
                                                           &current_node_stats)) {
                        trained = false;
                        break;
                }
                itty_feed_model_accumulate_train_stats (stats, &current_node_stats);
        }

        if (final_layer_input != input)
                itty_bit_string_list_free (final_layer_input);
        itty_bit_string_free (expanded_target);

        return trained;
}

bool
itty_feed_model_train_final_layer_with_suffix_oracle (itty_feed_model_t                     *model,
                                                      itty_bit_string_list_t                *input,
                                                      itty_bit_string_t                     *target,
                                                      itty_feed_model_train_options_t const *options,
                                                      itty_feed_model_train_stats_t         *stats)
{
        if (stats)
                *stats = (itty_feed_model_train_stats_t) { 0 };

        if (model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;
        if (!itty_feed_model_can_train_with_options (options))
                return false;

        size_t final_layer = model->number_of_layers - 1;
        itty_bit_string_list_t *final_layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                        input,
                                                                                        final_layer);
        itty_feed_model_final_repair_list_t repairs = { 0 };
        if (!itty_feed_model_collect_final_repairs (model,
                                                   final_layer_input,
                                                   target,
                                                   &repairs)) {
                if (final_layer_input != input)
                        itty_bit_string_list_free (final_layer_input);
                return false;
        }

        itty_bit_string_t *expanded_target = itty_feed_model_expand_target_for_layer (target,
                                                                                      final_layer);
        bool trained = true;

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_t *current_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                           model->masks_by_layer_node[final_layer][node_index]);
                itty_bit_string_t *oracle_target = itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                                              itty_bit_string_get_number_of_words (expanded_target));
                size_t accepted_bits = 0;

                for (size_t repair_index = 0; repair_index < repairs.count; repair_index++) {
                        itty_feed_model_final_repair_t *repair = &repairs.items[repair_index];

                        if (repair->final_node != node_index)
                                continue;

                        itty_bit_string_set_bit (oracle_target,
                                                 repair->condensed_bit,
                                                 repair->value);
                        accepted_bits++;
                }

                oracle_target->pop_count_computed = false;

                if (accepted_bits > 0) {
                        itty_feed_model_train_stats_t current_node_stats = { 0 };

                        if (!itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][node_index],
                                                                   final_layer_input,
                                                                   oracle_target,
                                                                   options,
                                                                   &current_node_stats)) {
                                        trained = false;
                        }
                        itty_feed_model_accumulate_train_stats (stats,
                                                                &current_node_stats);
                }

                itty_bit_string_free (oracle_target);
                itty_bit_string_free (current_condensed);

                if (!trained)
                        break;
        }

        itty_bit_string_free (expanded_target);
        itty_feed_model_final_repair_list_clear (&repairs);
        if (final_layer_input != input)
                itty_bit_string_list_free (final_layer_input);

        return trained;
}

static bool
itty_feed_model_train_final_layer_with_suffix_oracle_for_node_mode_internal (itty_feed_model_t                           *model,
                                                                             itty_bit_string_list_t                      *input,
                                                                             itty_bit_string_t                           *target,
                                                                             size_t                                       node_index,
                                                                             itty_feed_model_train_options_t const       *options,
                                                                             itty_feed_model_final_repair_mode_t          mode,
                                                                             itty_feed_model_train_stats_t               *stats,
                                                                             itty_feed_model_final_repair_mode_summary_t *summary)
{
        if (stats)
                *stats = (itty_feed_model_train_stats_t) { 0 };
        if (summary)
                *summary = (itty_feed_model_final_repair_mode_summary_t) { 0 };

        if (model->number_of_layers == 0 ||
            node_index >= model->nodes_per_layer ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;
        if (!itty_feed_model_can_train_with_options (options))
                return false;

        size_t final_layer = model->number_of_layers - 1;
        itty_bit_string_list_t *final_layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                        input,
                                                                                        final_layer);
        itty_feed_model_final_repair_list_t repairs = { 0 };
        if (!itty_feed_model_collect_final_repairs (model,
                                                    final_layer_input,
                                                    target,
                                                    &repairs)) {
                if (final_layer_input != input)
                        itty_bit_string_list_free (final_layer_input);
                return false;
        }

        itty_bit_string_t *current_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                   model->masks_by_layer_node[final_layer][node_index]);
        itty_bit_string_t *oracle_target = itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                                      itty_bit_string_get_number_of_words (current_condensed));
        size_t accepted_bits = 0;

        for (size_t repair_index = 0; repair_index < repairs.count; repair_index++) {
                itty_feed_model_final_repair_t *repair = &repairs.items[repair_index];

                if (repair->final_node != node_index)
                        continue;
                if ((mode == ITTY_FEED_MODEL_FINAL_REPAIR_MODE_POSITIVE_ONLY && !repair->value) ||
                    (mode == ITTY_FEED_MODEL_FINAL_REPAIR_MODE_ZERO_ONLY && repair->value))
                        continue;
                if (!itty_feed_model_train_options_bit_allowed (options,
                                                                repair->condensed_bit,
                                                                itty_bit_string_get_length (oracle_target),
                                                                false))
                        continue;
                if (summary)
                        summary->candidate_count++;

                itty_bit_string_set_bit (oracle_target,
                                         repair->condensed_bit,
                                         repair->value);
                accepted_bits++;
        }

        if (summary)
                summary->chosen_mode = mode;
        if (summary)
                summary->accepted_candidates = accepted_bits;

        oracle_target->pop_count_computed = false;

        bool trained = true;
        if (accepted_bits > 0) {
                itty_feed_model_train_stats_t current_node_stats = { 0 };

                if (!itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][node_index],
                                                           final_layer_input,
                                                           oracle_target,
                                                           options,
                                                           &current_node_stats))
                        trained = false;
                itty_feed_model_accumulate_train_stats (stats,
                                                        &current_node_stats);
        }

        itty_bit_string_free (oracle_target);
        itty_bit_string_free (current_condensed);
        itty_feed_model_final_repair_list_clear (&repairs);
        if (final_layer_input != input)
                itty_bit_string_list_free (final_layer_input);

        return trained;
}

bool
itty_feed_model_train_final_layer_with_suffix_oracle_for_node_mode (itty_feed_model_t                           *model,
                                                                    itty_bit_string_list_t                      *input,
                                                                    itty_bit_string_t                           *target,
                                                                    size_t                                       node_index,
                                                                    itty_feed_model_train_options_t const       *options,
                                                                    itty_feed_model_final_repair_mode_t          mode,
                                                                    itty_feed_model_train_stats_t               *stats,
                                                                    itty_feed_model_final_repair_mode_summary_t *summary)
{
        if (mode != ITTY_FEED_MODEL_FINAL_REPAIR_MODE_BALANCED)
                return itty_feed_model_train_final_layer_with_suffix_oracle_for_node_mode_internal (model,
                                                                                                    input,
                                                                                                    target,
                                                                                                    node_index,
                                                                                                    options,
                                                                                                    mode,
                                                                                                    stats,
                                                                                                    summary);

        if (stats)
                *stats = (itty_feed_model_train_stats_t) { 0 };
        if (summary)
                *summary = (itty_feed_model_final_repair_mode_summary_t) { 0 };

        itty_feed_model_decoder_objective_t before_objective = { 0 };
        if (!itty_feed_model_measure_decoder_objective_for_node (model,
                                                                 input,
                                                                 target,
                                                                 node_index,
                                                                 &before_objective))
                return false;

        itty_feed_model_layer_state_snapshot_t *best_post_snapshot = NULL;
        itty_feed_model_decoder_objective_t best_objective = before_objective;
        itty_feed_model_train_stats_t best_stats = { 0 };
        itty_feed_model_final_repair_mode_summary_t best_summary = { 0 };
        bool found = false;

        itty_feed_model_final_repair_mode_t const candidate_modes[] = {
                ITTY_FEED_MODEL_FINAL_REPAIR_MODE_POSITIVE_ONLY,
                ITTY_FEED_MODEL_FINAL_REPAIR_MODE_ZERO_ONLY,
                ITTY_FEED_MODEL_FINAL_REPAIR_MODE_BALANCED,
        };

        for (size_t mode_index = 0; mode_index < sizeof (candidate_modes) / sizeof (candidate_modes[0]); mode_index++) {
                itty_feed_model_layer_state_snapshot_t *before_snapshot = itty_feed_model_snapshot_final_layer_state (model);
                itty_feed_model_layer_state_snapshot_t *after_snapshot = NULL;
                itty_feed_model_train_stats_t candidate_stats = { 0 };
                itty_feed_model_final_repair_mode_summary_t candidate_summary = { 0 };
                itty_feed_model_decoder_objective_t candidate_objective = { 0 };
                bool trained_ok;

                if (!before_snapshot)
                        break;

                trained_ok = itty_feed_model_train_final_layer_with_suffix_oracle_for_node_mode_internal (model,
                                                                                                          input,
                                                                                                          target,
                                                                                                          node_index,
                                                                                                          options,
                                                                                                          candidate_modes[mode_index],
                                                                                                          &candidate_stats,
                                                                                                          &candidate_summary) &&
                             itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                 input,
                                                                                 target,
                                                                                 node_index,
                                                                                 &candidate_objective);
                if (trained_ok)
                        after_snapshot = itty_feed_model_snapshot_final_layer_state (model);

                if (trained_ok &&
                    after_snapshot &&
                    (!found ||
                     itty_feed_model_route_local_balanced_objective_is_better (&candidate_objective, &best_objective))) {
                        if (best_post_snapshot)
                                itty_feed_model_restore_final_layer_state_snapshot (model, best_post_snapshot);
                        best_post_snapshot = after_snapshot;
                        best_objective = candidate_objective;
                        best_stats = candidate_stats;
                        best_summary = candidate_summary;
                        best_summary.chosen_mode = candidate_modes[mode_index];
                        found = true;
                }

                itty_feed_model_restore_final_layer_state_snapshot (model, before_snapshot);
                if (trained_ok && after_snapshot && after_snapshot != best_post_snapshot)
                        itty_feed_model_restore_final_layer_state_snapshot (model, after_snapshot);
        }

        if (!found) {
            return false;
        }

        itty_feed_model_restore_final_layer_state_snapshot (model, best_post_snapshot);

        if (stats)
                *stats = best_stats;
        if (summary)
                *summary = best_summary;
        return true;
}

bool
itty_feed_model_train_final_layer_with_suffix_oracle_for_node (itty_feed_model_t                     *model,
                                                               itty_bit_string_list_t                *input,
                                                               itty_bit_string_t                     *target,
                                                               size_t                                 node_index,
                                                               itty_feed_model_train_options_t const *options,
                                                               itty_feed_model_train_stats_t         *stats)
{
        return itty_feed_model_train_final_layer_with_suffix_oracle_for_node_mode (model,
                                                                                   input,
                                                                                   target,
                                                                                   node_index,
                                                                                   options,
                                                                                   ITTY_FEED_MODEL_FINAL_REPAIR_MODE_BALANCED,
                                                                                   stats,
                                                                                   NULL);
}

bool
itty_feed_model_measure_best_single_final_layer_flip_for_node (itty_feed_model_t                    *model,
                                                               itty_bit_string_list_t               *input,
                                                               itty_bit_string_t                    *target,
                                                               size_t                                node_index,
                                                               bool                                 *found,
                                                               itty_feed_model_mask_flip_trace_t    *trace,
                                                               itty_feed_model_decoder_objective_t  *objective)
{
        size_t final_layer;
        itty_bit_string_list_t *baseline_outputs = NULL;
        itty_feed_model_decoder_objective_t current = { 0 };
        itty_feed_model_decoder_objective_t best = { 0 };
        bool have_best = false;

        if (found)
                *found = false;
        if (trace)
                *trace = (itty_feed_model_mask_flip_trace_t) { 0 };
        if (objective)
                *objective = (itty_feed_model_decoder_objective_t) { 0 };
        if (!model || !input || !target || node_index >= model->nodes_per_layer)
                return false;

        final_layer = model->number_of_layers - 1;
        baseline_outputs = itty_feed_model_run_outputs (model, input);
        if (!baseline_outputs ||
            !itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                  baseline_outputs,
                                                                  target,
                                                                  node_index,
                                                                  &current)) {
                if (baseline_outputs)
                        itty_bit_string_list_free (baseline_outputs);
                return false;
        }
        itty_bit_string_list_free (baseline_outputs);

        itty_bit_string_list_t *masks = model->masks_by_layer_node[final_layer][node_index];
        for (size_t input_index = 0; input_index < masks->count; input_index++) {
                itty_bit_string_t *mask = masks->bit_strings[input_index];
                size_t bit_capacity = itty_bit_string_get_length (mask);

                for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                        itty_bit_string_list_t *outputs = NULL;
                        itty_feed_model_decoder_objective_t candidate = { 0 };

                        itty_feed_model_flip_mask_bit (mask, bit_index);
                        outputs = itty_feed_model_run_outputs (model, input);
                        if (outputs &&
                            itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                                 outputs,
                                                                                 target,
                                                                                 node_index,
                                                                                 &candidate) &&
                            itty_feed_model_decoder_objective_is_better (&candidate, &current) &&
                            (!have_best ||
                             itty_feed_model_decoder_objective_is_better (&candidate, &best))) {
                                have_best = true;
                                best = candidate;
                                if (trace) {
                                        *trace = (itty_feed_model_mask_flip_trace_t) {
                                                .node_index = node_index,
                                                .input_index = input_index,
                                                .bit_index = bit_index,
                                                .value_after = itty_bit_string_get_bit (mask, bit_index),
                                        };
                                }
                        }
                        if (outputs)
                                itty_bit_string_list_free (outputs);
                        itty_feed_model_flip_mask_bit (mask, bit_index);
                }
        }

        if (found)
                *found = have_best;
        if (objective && have_best)
                *objective = best;
        return true;
}

bool
itty_feed_model_measure_best_single_penultimate_layer_flip_for_node (itty_feed_model_t                    *model,
                                                                     itty_bit_string_list_t               *input,
                                                                     itty_bit_string_t                    *target,
                                                                     size_t                                node_index,
                                                                     bool                                 *found,
                                                                     itty_feed_model_mask_flip_trace_t    *trace,
                                                                     itty_feed_model_decoder_objective_t  *objective)
{
        size_t penultimate_layer;
        itty_bit_string_list_t *baseline_outputs = NULL;
        itty_feed_model_decoder_objective_t current = { 0 };
        itty_feed_model_decoder_objective_t best = { 0 };
        bool have_best = false;

        if (found)
                *found = false;
        if (trace)
                *trace = (itty_feed_model_mask_flip_trace_t) { 0 };
        if (objective)
                *objective = (itty_feed_model_decoder_objective_t) { 0 };
        if (!model || !input || !target || model->number_of_layers < 2 || node_index >= model->nodes_per_layer)
                return false;

        penultimate_layer = model->number_of_layers - 2;
        baseline_outputs = itty_feed_model_run_outputs (model, input);
        if (!baseline_outputs ||
            !itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                  baseline_outputs,
                                                                  target,
                                                                  node_index,
                                                                  &current)) {
                if (baseline_outputs)
                        itty_bit_string_list_free (baseline_outputs);
                return false;
        }
        itty_bit_string_list_free (baseline_outputs);

        itty_bit_string_list_t *masks = model->masks_by_layer_node[penultimate_layer][node_index];
        for (size_t input_index = 0; input_index < masks->count; input_index++) {
                itty_bit_string_t *mask = masks->bit_strings[input_index];
                size_t bit_capacity = itty_bit_string_get_length (mask);

                for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                        itty_bit_string_list_t *outputs = NULL;
                        itty_feed_model_decoder_objective_t candidate = { 0 };

                        itty_feed_model_flip_mask_bit (mask, bit_index);
                        outputs = itty_feed_model_run_outputs (model, input);
                        if (outputs &&
                            itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                                 outputs,
                                                                                 target,
                                                                                 node_index,
                                                                                 &candidate) &&
                            itty_feed_model_decoder_objective_is_better (&candidate, &current) &&
                            (!have_best ||
                             itty_feed_model_decoder_objective_is_better (&candidate, &best))) {
                                have_best = true;
                                best = candidate;
                                if (trace) {
                                        *trace = (itty_feed_model_mask_flip_trace_t) {
                                                .node_index = node_index,
                                                .input_index = input_index,
                                                .bit_index = bit_index,
                                                .value_after = itty_bit_string_get_bit (mask, bit_index),
                                        };
                                }
                        }
                        if (outputs)
                                itty_bit_string_list_free (outputs);
                        itty_feed_model_flip_mask_bit (mask, bit_index);
                }
        }

        if (found)
                *found = have_best;
        if (objective && have_best)
                *objective = best;
        return true;
}

bool
itty_feed_model_apply_penultimate_layer_mask_flip_trace (itty_feed_model_t                       *model,
                                                         itty_feed_model_mask_flip_trace_t const *trace)
{
        size_t penultimate_layer;
        itty_bit_string_list_t *masks;
        itty_bit_string_t *mask;

        if (!model || !trace || model->number_of_layers < 2 || trace->node_index >= model->nodes_per_layer)
                return false;

        penultimate_layer = model->number_of_layers - 2;
        masks = model->masks_by_layer_node[penultimate_layer][trace->node_index];
        if (!masks || trace->input_index >= masks->count)
                return false;

        mask = masks->bit_strings[trace->input_index];
        if (!mask || trace->bit_index >= itty_bit_string_get_length (mask))
                return false;

        itty_feed_model_flip_mask_bit (mask, trace->bit_index);
        return true;
}

bool
itty_feed_model_measure_best_penultimate_same_bit_bundle_for_node (itty_feed_model_t                    *model,
                                                                   itty_bit_string_list_t               *input,
                                                                   itty_bit_string_t                    *target,
                                                                   size_t                                node_index,
                                                                   size_t                                min_bundle_size,
                                                                   size_t                                max_bundle_size,
                                                                   bool                                 *found,
                                                                   itty_feed_model_mask_flip_trace_t    *traces,
                                                                   size_t                                trace_capacity,
                                                                   size_t                               *trace_count,
                                                                   itty_feed_model_decoder_objective_t  *objective)
{
        return itty_feed_model_measure_best_penultimate_same_bit_bundle_for_node_excluding_bits (model,
                                                                                                  input,
                                                                                                  target,
                                                                                                  node_index,
                                                                                                  min_bundle_size,
                                                                                                  max_bundle_size,
                                                                                                  NULL,
                                                                                                  found,
                                                                                                  traces,
                                                                                                  trace_capacity,
                                                                                                  trace_count,
                                                                                                  objective);
}

bool
itty_feed_model_measure_best_penultimate_same_bit_bundle_for_node_excluding_bits (itty_feed_model_t                    *model,
                                                                                   itty_bit_string_list_t               *input,
                                                                                   itty_bit_string_t                    *target,
                                                                                   size_t                                node_index,
                                                                                   size_t                                min_bundle_size,
                                                                                   size_t                                max_bundle_size,
                                                                                   itty_bit_string_t                    *excluded_bits,
                                                                                   bool                                 *found,
                                                                                   itty_feed_model_mask_flip_trace_t    *traces,
                                                                                   size_t                                trace_capacity,
                                                                                   size_t                               *trace_count,
                                                                                   itty_feed_model_decoder_objective_t  *objective)
{
        size_t penultimate_layer;
        itty_bit_string_list_t *baseline_outputs = NULL;
        itty_feed_model_decoder_objective_t current = { 0 };
        itty_feed_model_decoder_objective_t best = { 0 };
        bool have_best = false;
        size_t best_trace_count = 0;

        if (found)
                *found = false;
        if (trace_count)
                *trace_count = 0;
        if (objective)
                *objective = (itty_feed_model_decoder_objective_t) { 0 };
        if (traces && trace_capacity > 0)
                memset (traces, 0, trace_capacity * sizeof (*traces));
        if (!model || !input || !target || model->number_of_layers < 2 || node_index >= model->nodes_per_layer)
                return false;

        penultimate_layer = model->number_of_layers - 2;
        if (model->nodes_per_layer == 0 || model->nodes_per_layer > 63)
                return false;
        if (min_bundle_size == 0)
                min_bundle_size = 1;
        if (max_bundle_size == 0 || max_bundle_size > model->nodes_per_layer)
                max_bundle_size = model->nodes_per_layer;
        if (min_bundle_size > max_bundle_size || trace_capacity < max_bundle_size)
                return false;

        baseline_outputs = itty_feed_model_run_outputs (model, input);
        if (!baseline_outputs ||
            !itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                  baseline_outputs,
                                                                  target,
                                                                  node_index,
                                                                  &current)) {
                if (baseline_outputs)
                        itty_bit_string_list_free (baseline_outputs);
                return false;
        }
        itty_bit_string_list_free (baseline_outputs);

        size_t input_count = model->masks_by_layer_node[penultimate_layer][0]->count;
        for (size_t input_index = 0; input_index < input_count; input_index++) {
                size_t bit_capacity =
                        itty_bit_string_get_length (model->masks_by_layer_node[penultimate_layer][0]->bit_strings[input_index]);
                uint64_t subset_limit = UINT64_C(1) << model->nodes_per_layer;

                for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                        if (excluded_bits &&
                            bit_index < itty_bit_string_get_length (excluded_bits) &&
                            itty_bit_string_get_bit (excluded_bits, bit_index))
                                continue;

                        for (uint64_t subset = 1; subset < subset_limit; subset++) {
                                size_t bundle_size = (size_t) __builtin_popcountll ((unsigned long long) subset);
                                itty_bit_string_list_t *outputs = NULL;
                                itty_feed_model_decoder_objective_t candidate = { 0 };

                                if (bundle_size < min_bundle_size || bundle_size > max_bundle_size)
                                        continue;

                                for (size_t bundle_node = 0; bundle_node < model->nodes_per_layer; bundle_node++) {
                                        if ((subset & (UINT64_C(1) << bundle_node)) == 0)
                                                continue;
                                        itty_feed_model_flip_mask_bit (model->masks_by_layer_node[penultimate_layer][bundle_node]->bit_strings[input_index],
                                                                       bit_index);
                                }

                                outputs = itty_feed_model_run_outputs (model, input);
                                if (outputs &&
                                    itty_feed_model_evaluate_decoder_objective_for_node (model,
                                                                                         outputs,
                                                                                         target,
                                                                                         node_index,
                                                                                         &candidate) &&
                                    itty_feed_model_decoder_objective_is_better (&candidate, &current) &&
                                    (!have_best ||
                                     itty_feed_model_decoder_objective_is_better (&candidate, &best))) {
                                        size_t trace_index = 0;

                                        have_best = true;
                                        best = candidate;
                                        best_trace_count = bundle_size;
                                        if (traces) {
                                                memset (traces, 0, trace_capacity * sizeof (*traces));
                                                for (size_t bundle_node = 0; bundle_node < model->nodes_per_layer; bundle_node++) {
                                                        if ((subset & (UINT64_C(1) << bundle_node)) == 0)
                                                                continue;
                                                        traces[trace_index++] = (itty_feed_model_mask_flip_trace_t) {
                                                                .node_index = bundle_node,
                                                                .input_index = input_index,
                                                                .bit_index = bit_index,
                                                                .value_after = itty_bit_string_get_bit (model->masks_by_layer_node[penultimate_layer][bundle_node]->bit_strings[input_index],
                                                                                                       bit_index),
                                                        };
                                                }
                                        }
                                }
                                if (outputs)
                                        itty_bit_string_list_free (outputs);

                                for (size_t bundle_node = 0; bundle_node < model->nodes_per_layer; bundle_node++) {
                                        if ((subset & (UINT64_C(1) << bundle_node)) == 0)
                                                continue;
                                        itty_feed_model_flip_mask_bit (model->masks_by_layer_node[penultimate_layer][bundle_node]->bit_strings[input_index],
                                                                       bit_index);
                                }
                        }
                }
        }

        if (found)
                *found = have_best;
        if (trace_count)
                *trace_count = best_trace_count;
        if (objective && have_best)
                *objective = best;
        return true;
}

static bool
itty_feed_model_train_final_layer_selector_protection_for_node_internal (itty_feed_model_t                             *model,
                                                                         itty_bit_string_list_t                        *input,
                                                                         itty_bit_string_t                             *target,
                                                                         size_t                                         owner_route,
                                                                         itty_bit_string_list_t                        *guard_input,
                                                                         itty_bit_string_t                             *guard_target,
                                                                         size_t                                         guard_route,
                                                                         bool                                           require_decode_safety,
                                                                         bool                                           apply_best_snapshot,
                                                                         itty_feed_model_train_options_t const         *options,
                                                                         itty_feed_model_selector_protection_summary_t *summary)
{
        if (summary)
                *summary = (itty_feed_model_selector_protection_summary_t) { 0 };

        if (model == NULL ||
            input == NULL ||
            target == NULL ||
            model->number_of_layers == 0 ||
            owner_route >= model->nodes_per_layer ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words ||
            (guard_input && itty_bit_string_list_get_length (guard_input) != model->inputs_per_node) ||
            (guard_target && itty_bit_string_get_number_of_words (guard_target) > model->vocabulary_words) ||
            (guard_input && guard_route >= model->nodes_per_layer) ||
            !itty_feed_model_can_train_with_options (options))
                return false;

        size_t final_layer = model->number_of_layers - 1;
        itty_bit_string_list_t *final_layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                        input,
                                                                                        final_layer);
        itty_feed_model_decoder_objective_t owner_before = { 0 };
        itty_feed_model_decoder_objective_t global_before = { 0 };
        ptrdiff_t margin_before = 0;
        size_t competitor_route = owner_route;
        itty_feed_model_decoder_objective_t guard_before = { 0 };
        bool measured_before = itty_feed_model_measure_route_margin_state_with_options (model,
                                                                                        input,
                                                                                        target,
                                                                                        owner_route,
                                                                                        options,
                                                                                        &owner_before,
                                                                                        &global_before,
                                                                                        &margin_before,
                                                                                        &competitor_route);
        if (measured_before && guard_input && guard_target)
                measured_before = itty_feed_model_train_options_has_lane_split (options) ?
                                  itty_feed_model_measure_decoder_objective_for_node_with_lane_split (model,
                                                                                                     guard_input,
                                                                                                     guard_target,
                                                                                                     guard_route,
                                                                                                     options->selector_lane_bit_offset,
                                                                                                     options->selector_lane_bit_count,
                                                                                                     options->decoder_lane_bit_offset,
                                                                                                     options->decoder_lane_bit_count,
                                                                                                     &guard_before) :
                                  itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                      guard_input,
                                                                                      guard_target,
                                                                                      guard_route,
                                                                                      &guard_before);

        if (!measured_before) {
                if (final_layer_input != input)
                        itty_bit_string_list_free (final_layer_input);
                return false;
        }

        itty_feed_model_layer_state_snapshot_t *best_snapshot = NULL;
        itty_feed_model_selector_protection_summary_t best_summary = {
                .owner_route = owner_route,
                .competitor_route = competitor_route,
                .margin_before = margin_before,
                .margin_after = margin_before,
                .owner_distance_before = owner_before.selected_distance,
                .owner_distance_after = owner_before.selected_distance,
                .global_distance_before = global_before.selected_distance,
                .global_distance_after = global_before.selected_distance,
                .guard_route = guard_route,
                .guard_distance_before = guard_before.selected_distance,
                .guard_distance_after = guard_before.selected_distance,
                .guard_deficit_before = guard_before.false_negative_vote_deficit,
                .guard_deficit_after = guard_before.false_negative_vote_deficit,
        };

        for (size_t candidate_kind = ITTY_FEED_MODEL_SELECTOR_PROTECTION_OWNER_LIFT;
             candidate_kind <= ITTY_FEED_MODEL_SELECTOR_PROTECTION_COMPETITOR_SUPPRESS;
             candidate_kind++) {
                size_t candidate_route = candidate_kind == ITTY_FEED_MODEL_SELECTOR_PROTECTION_OWNER_LIFT ?
                                         owner_route :
                                         competitor_route;
                if (candidate_route >= model->nodes_per_layer)
                        continue;

                if (summary) {
                        if (candidate_kind == ITTY_FEED_MODEL_SELECTOR_PROTECTION_OWNER_LIFT)
                                summary->owner_route = owner_route;
                        else
                                summary->competitor_route = competitor_route;
                }

                itty_bit_string_t *current_condensed = itty_feed_model_run_node_condensed (final_layer_input,
                                                                                           model->masks_by_layer_node[final_layer][candidate_route]);
                size_t bit_capacity = itty_bit_string_get_length (current_condensed);

                for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                        if (!itty_feed_model_train_options_bit_allowed (options,
                                                                        bit_index,
                                                                        bit_capacity,
                                                                        true))
                                continue;
                        bool current_value = itty_bit_string_get_bit (current_condensed, bit_index);
                        bool desired_value = candidate_kind == ITTY_FEED_MODEL_SELECTOR_PROTECTION_OWNER_LIFT;

                        if (current_value == desired_value)
                                continue;

                        if (summary) {
                                if (candidate_kind == ITTY_FEED_MODEL_SELECTOR_PROTECTION_OWNER_LIFT)
                                        summary->owner_candidate_bits++;
                                else
                                        summary->competitor_candidate_bits++;
                        }

                        itty_feed_model_layer_state_snapshot_t *candidate_snapshot =
                                itty_feed_model_snapshot_layer_state (model, final_layer);
                        itty_bit_string_t *oracle_target =
                                itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                           itty_bit_string_get_number_of_words (current_condensed));
                        itty_feed_model_train_stats_t candidate_stats = { 0 };
                        itty_feed_model_decoder_objective_t owner_after = { 0 };
                        itty_feed_model_decoder_objective_t global_after = { 0 };
                        itty_feed_model_decoder_objective_t guard_after = { 0 };
                        ptrdiff_t margin_after = margin_before;
                        size_t competitor_after = competitor_route;

                        itty_bit_string_set_bit (oracle_target, bit_index, desired_value);
                        oracle_target->pop_count_computed = false;

                        bool trained =
                                itty_feed_model_train_layer_one_node (model->masks_by_layer_node[final_layer][candidate_route],
                                                                      final_layer_input,
                                                                      oracle_target,
                                                                      options,
                                                                      &candidate_stats) &&
                                candidate_stats.flips > 0 &&
                                itty_feed_model_measure_route_margin_state_with_options (model,
                                                                                         input,
                                                                                         target,
                                                                                         owner_route,
                                                                                         options,
                                                                                         &owner_after,
                                                                                         &global_after,
                                                                                         &margin_after,
                                                                                         &competitor_after);
                        if (trained && guard_input && guard_target)
                                trained = itty_feed_model_train_options_has_lane_split (options) ?
                                          itty_feed_model_measure_decoder_objective_for_node_with_lane_split (model,
                                                                                                             guard_input,
                                                                                                             guard_target,
                                                                                                             guard_route,
                                                                                                             options->selector_lane_bit_offset,
                                                                                                             options->selector_lane_bit_count,
                                                                                                             options->decoder_lane_bit_offset,
                                                                                                             options->decoder_lane_bit_count,
                                                                                                             &guard_after) :
                                          itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                              guard_input,
                                                                                              guard_target,
                                                                                              guard_route,
                                                                                              &guard_after);

                        bool a_safe_after = trained &&
                                            owner_after.selected_distance == 0 &&
                                            global_after.selected_distance == 0;
                        bool effective_after = trained && margin_after > margin_before;
                        if (summary && a_safe_after) {
                                if (candidate_kind == ITTY_FEED_MODEL_SELECTOR_PROTECTION_OWNER_LIFT)
                                        summary->owner_safe_candidates++;
                                else
                                        summary->competitor_safe_candidates++;
                        }
                        if (summary && effective_after) {
                                if (candidate_kind == ITTY_FEED_MODEL_SELECTOR_PROTECTION_OWNER_LIFT)
                                        summary->owner_effective_candidates++;
                                else
                                        summary->competitor_effective_candidates++;
                        }

                        if (trained &&
                            (!require_decode_safety || a_safe_after) &&
                            effective_after &&
                            (best_snapshot == NULL ||
                             margin_after > best_summary.margin_after ||
                             (margin_after == best_summary.margin_after &&
                              guard_after.selected_distance < best_summary.guard_distance_after) ||
                             (margin_after == best_summary.margin_after &&
                              guard_after.selected_distance == best_summary.guard_distance_after &&
                              guard_after.false_negative_vote_deficit < best_summary.guard_deficit_after) ||
                             (margin_after == best_summary.margin_after &&
                              guard_after.selected_distance == best_summary.guard_distance_after &&
                              guard_after.false_negative_vote_deficit == best_summary.guard_deficit_after &&
                              candidate_stats.flips < best_summary.flips) ||
                             (margin_after == best_summary.margin_after &&
                              guard_after.selected_distance == best_summary.guard_distance_after &&
                              guard_after.false_negative_vote_deficit == best_summary.guard_deficit_after &&
                              candidate_stats.flips == best_summary.flips &&
                              candidate_kind < best_summary.kind))) {
                                if (best_snapshot)
                                        itty_feed_model_free_layer_state_snapshot (model, best_snapshot);
                                best_snapshot = itty_feed_model_snapshot_layer_state (model, final_layer);
                                best_summary = (itty_feed_model_selector_protection_summary_t) {
                                        .accepted = true,
                                        .kind = (itty_feed_model_selector_protection_kind_t) candidate_kind,
                                        .owner_route = owner_route,
                                        .competitor_route = competitor_after,
                                        .margin_before = margin_before,
                                        .margin_after = margin_after,
                                        .owner_distance_before = owner_before.selected_distance,
                                        .owner_distance_after = owner_after.selected_distance,
                                        .global_distance_before = global_before.selected_distance,
                                        .global_distance_after = global_after.selected_distance,
                                        .guard_route = guard_route,
                                        .guard_distance_before = guard_before.selected_distance,
                                        .guard_distance_after = guard_after.selected_distance,
                                        .guard_deficit_before = guard_before.false_negative_vote_deficit,
                                        .guard_deficit_after = guard_after.false_negative_vote_deficit,
                                        .flips = candidate_stats.flips,
                                };
                        }

                        itty_feed_model_restore_layer_state_snapshot (model, final_layer, candidate_snapshot);
                        itty_bit_string_free (oracle_target);
                }

                itty_bit_string_free (current_condensed);
        }

        if (summary) {
                best_summary.owner_candidate_bits = summary->owner_candidate_bits;
                best_summary.competitor_candidate_bits = summary->competitor_candidate_bits;
                best_summary.owner_safe_candidates = summary->owner_safe_candidates;
                best_summary.competitor_safe_candidates = summary->competitor_safe_candidates;
                best_summary.owner_effective_candidates = summary->owner_effective_candidates;
                best_summary.competitor_effective_candidates = summary->competitor_effective_candidates;
                best_summary.mixed_candidate_pairs = summary->owner_safe_candidates *
                                                     summary->competitor_safe_candidates;
        }

        if (apply_best_snapshot && best_snapshot) {
                itty_feed_model_restore_layer_state_snapshot (model, final_layer, best_snapshot);
                best_snapshot = NULL;
        }
        if (!apply_best_snapshot && best_snapshot)
                itty_feed_model_free_layer_state_snapshot (model, best_snapshot);

        if (summary)
                *summary = best_summary;

        if (final_layer_input != input)
                itty_bit_string_list_free (final_layer_input);

        return true;
}

bool
itty_feed_model_train_final_layer_selector_protection_for_node (itty_feed_model_t                             *model,
                                                                itty_bit_string_list_t                        *input,
                                                                itty_bit_string_t                             *target,
                                                                size_t                                         owner_route,
                                                                itty_feed_model_train_options_t const         *options,
                                                                itty_feed_model_selector_protection_summary_t *summary)
{
        return itty_feed_model_train_final_layer_selector_protection_for_node_internal (model,
                                                                                        input,
                                                                                        target,
                                                                                        owner_route,
                                                                                        NULL,
                                                                                        NULL,
                                                                                        0,
                                                                                        true,
                                                                                        true,
                                                                                        options,
                                                                                        summary);
}

bool
itty_feed_model_train_final_layer_selector_protection_for_node_with_guard (itty_feed_model_t                             *model,
                                                                           itty_bit_string_list_t                        *input,
                                                                           itty_bit_string_t                             *target,
                                                                           size_t                                         owner_route,
                                                                           itty_bit_string_list_t                        *guard_input,
                                                                           itty_bit_string_t                             *guard_target,
                                                                           size_t                                         guard_route,
                                                                           itty_feed_model_train_options_t const         *options,
                                                                           itty_feed_model_selector_protection_summary_t *summary)
{
        return itty_feed_model_train_final_layer_selector_protection_for_node_internal (model,
                                                                                        input,
                                                                                        target,
                                                                                        owner_route,
                                                                                        guard_input,
                                                                                        guard_target,
                                                                                        guard_route,
                                                                                        true,
                                                                                        true,
                                                                                        options,
                                                                                        summary);
}

bool
itty_feed_model_measure_final_layer_selector_protection_for_node_with_guard (itty_feed_model_t                             *model,
                                                                             itty_bit_string_list_t                        *input,
                                                                             itty_bit_string_t                             *target,
                                                                             size_t                                         owner_route,
                                                                             itty_bit_string_list_t                        *guard_input,
                                                                             itty_bit_string_t                             *guard_target,
                                                                             size_t                                         guard_route,
                                                                             itty_feed_model_train_options_t const         *options,
                                                                             itty_feed_model_selector_protection_summary_t *summary)
{
        return itty_feed_model_train_final_layer_selector_protection_for_node_internal (model,
                                                                                        input,
                                                                                        target,
                                                                                        owner_route,
                                                                                        guard_input,
                                                                                        guard_target,
                                                                                        guard_route,
                                                                                        true,
                                                                                        false,
                                                                                        options,
                                                                                        summary);
}

bool
itty_feed_model_train_final_layer_selector_margin_for_node (itty_feed_model_t                             *model,
                                                            itty_bit_string_list_t                        *input,
                                                            itty_bit_string_t                             *target,
                                                            size_t                                         owner_route,
                                                            itty_feed_model_train_options_t const         *options,
                                                            itty_feed_model_selector_protection_summary_t *summary)
{
        return itty_feed_model_train_final_layer_selector_protection_for_node_internal (model,
                                                                                        input,
                                                                                        target,
                                                                                        owner_route,
                                                                                        NULL,
                                                                                        NULL,
                                                                                        0,
                                                                                        false,
                                                                                        true,
                                                                                        options,
                                                                                        summary);
}

bool
itty_feed_model_measure_final_layer_selector_margin_for_node (itty_feed_model_t                             *model,
                                                              itty_bit_string_list_t                        *input,
                                                              itty_bit_string_t                             *target,
                                                              size_t                                         owner_route,
                                                              itty_feed_model_train_options_t const         *options,
                                                              itty_feed_model_selector_protection_summary_t *summary)
{
        return itty_feed_model_train_final_layer_selector_protection_for_node_internal (model,
                                                                                        input,
                                                                                        target,
                                                                                        owner_route,
                                                                                        NULL,
                                                                                        NULL,
                                                                                        0,
                                                                                        false,
                                                                                        false,
                                                                                        options,
                                                                                        summary);
}

itty_feed_model_layer_state_snapshot_t *
itty_feed_model_snapshot_final_layer_state (itty_feed_model_t *model)
{
        if (!model || model->number_of_layers == 0)
                return NULL;

        return itty_feed_model_snapshot_layer_state (model, model->number_of_layers - 1);
}

void
itty_feed_model_restore_final_layer_state_snapshot (itty_feed_model_t                      *model,
                                                    itty_feed_model_layer_state_snapshot_t *snapshot)
{
        if (!model || !snapshot || model->number_of_layers == 0)
                return;

        itty_feed_model_restore_layer_state_snapshot (model, model->number_of_layers - 1, snapshot);
}

itty_feed_model_layer_state_snapshot_t *
itty_feed_model_snapshot_penultimate_layer_state (itty_feed_model_t *model)
{
        if (!model || model->number_of_layers < 2)
                return NULL;

        return itty_feed_model_snapshot_layer_state (model, model->number_of_layers - 2);
}

void
itty_feed_model_restore_penultimate_layer_state_snapshot (itty_feed_model_t                      *model,
                                                          itty_feed_model_layer_state_snapshot_t *snapshot)
{
        if (!model || !snapshot || model->number_of_layers < 2)
                return;

        itty_feed_model_restore_layer_state_snapshot (model, model->number_of_layers - 2, snapshot);
}

void
itty_feed_model_free_penultimate_layer_state_snapshot (itty_feed_model_t                      *model,
                                                       itty_feed_model_layer_state_snapshot_t *snapshot)
{
        if (!model || !snapshot || model->number_of_layers < 2)
                return;

        itty_feed_model_free_layer_state_snapshot (model, snapshot);
}

void
itty_feed_model_free_final_layer_state_snapshot (itty_feed_model_t                      *model,
                                                 itty_feed_model_layer_state_snapshot_t *snapshot)
{
        if (!model || !snapshot)
                return;

        itty_feed_model_free_layer_state_snapshot (model, snapshot);
}

bool
itty_feed_model_train_penultimate_layer_with_final_repairs (itty_feed_model_t                                *model,
                                                            itty_bit_string_list_t                           *input,
                                                            itty_bit_string_t                                *target,
                                                            itty_feed_model_projected_repair_options_t const *options,
                                                            itty_feed_model_projected_repair_stats_t         *stats)
{
        if (stats)
                *stats = (itty_feed_model_projected_repair_stats_t) { 0 };

        if (!model ||
            !input ||
            !target ||
            model->number_of_layers < 2 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        size_t penultimate_layer = model->number_of_layers - 2;
        size_t final_layer = model->number_of_layers - 1;
        size_t max_projected_blocks = options ? options->max_projected_blocks : 0;
        size_t max_layer_flips = options ? options->max_layer_flips : 0;
        size_t max_blocks_per_final_node = options ? options->max_blocks_per_final_node : 0;
        size_t max_strict_distance_blocks = options ? options->max_strict_distance_blocks : 0;
        size_t max_blocker_blocks = options ? options->max_blocker_blocks : 0;
        bool limit_strict_distance_blocks = options ? options->limit_strict_distance_blocks : false;
        bool limit_blocker_blocks = options ? options->limit_blocker_blocks : false;
        bool use_or_residual_repairs = options ? options->use_or_residual_repairs : false;
        bool prefer_blocker_efficiency = options ? options->prefer_blocker_efficiency : false;
        bool require_residual_blocker_efficiency = options ? options->require_residual_blocker_efficiency : false;
        itty_feed_model_replay_example_t const *replay_examples = options ? options->replay_examples : NULL;
        size_t replay_example_count = options ? options->replay_example_count : 0;
        bool strict_replay_guard = options ? options->strict_replay_guard : false;
        bool replay_safe_quota_complete_only = options ? options->replay_safe_quota_complete_only : false;

        itty_bit_string_list_t *layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                  input,
                                                                                  penultimate_layer);
        itty_bit_string_list_t *layer_outputs = itty_feed_model_run_layer (model,
                                                                           penultimate_layer,
                                                                           layer_input);
        itty_feed_model_decoder_objective_t before_objective = { 0 };
        if (!itty_feed_model_evaluate_suffix_decoder_objective (model,
                                                                layer_outputs,
                                                                penultimate_layer,
                                                                target,
                                                                &before_objective)) {
                if (layer_input != input)
                        itty_bit_string_list_free (layer_input);
                itty_bit_string_list_free (layer_outputs);
                return false;
        }

        if (stats) {
                stats->before_distance = before_objective.selected_distance;
                stats->before_blockers = before_objective.false_negative_blocker_bits;
        }

        itty_feed_model_final_repair_list_t repairs = { 0 };
        bool collected_repairs = model->decoder == ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE ?
                                 itty_feed_model_collect_segment_final_repairs_for_penultimate (model,
                                                                                                layer_input,
                                                                                                layer_outputs,
                                                                                                target,
                                                                                                options,
                                                                                                &repairs,
                                                                                                stats) :
                                 itty_feed_model_collect_final_repairs (model,
                                                                        layer_outputs,
                                                                        target,
                                                                        &repairs);
        if (!collected_repairs) {
                if (layer_input != input)
                        itty_bit_string_list_free (layer_input);
                itty_bit_string_list_free (layer_outputs);
                return false;
        }

        size_t condensed_words = model->vocabulary_words << penultimate_layer;
        size_t output_words = condensed_words * 2;
        size_t output_bit_capacity = output_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t target_bit_capacity = itty_bit_string_get_number_of_words (target) *
                                     ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        itty_bit_string_t **condensed_targets = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));
        itty_bit_string_t **condensed_cares = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));
        itty_bit_string_t **residual_cares = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));
        itty_bit_string_t **output_targets = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));
        itty_bit_string_t **output_cares = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));
        itty_bit_string_t **structural_output_targets = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));
        itty_bit_string_t **structural_output_cares = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));
        size_t *blocks_by_final_node = calloc (model->nodes_per_layer, sizeof (size_t));
        size_t projected_blocks = 0;
        itty_feed_model_projected_repair_candidate_t *candidates = NULL;
        bool *selected_candidates = NULL;
        itty_feed_model_projected_repair_candidate_t best_replay_candidate = { 0 };
        itty_feed_model_bad_flip_frequency_list_t bad_flip_frequencies = { 0 };
        size_t *flips_by_decoded_bit = calloc (target_bit_capacity, sizeof (size_t));
        size_t *selected_safe_votes_by_decoded_bit = calloc (target_bit_capacity, sizeof (size_t));
        size_t *selected_safe_flips_by_decoded_bit = calloc (target_bit_capacity, sizeof (size_t));
        size_t *replay_safe_votes_by_decoded_bit = calloc (target_bit_capacity, sizeof (size_t));
        size_t *replay_unsafe_votes_by_decoded_bit = calloc (target_bit_capacity, sizeof (size_t));
        size_t *replay_quota_by_decoded_bit = calloc (target_bit_capacity, sizeof (size_t));
        itty_bit_string_t **replay_safe_final_votes_by_decoded_bit = calloc (target_bit_capacity, sizeof (itty_bit_string_t *));
        itty_bit_string_t **replay_unsafe_final_votes_by_decoded_bit = calloc (target_bit_capacity, sizeof (itty_bit_string_t *));
        itty_bit_string_t **selected_safe_final_votes_by_decoded_bit = calloc (target_bit_capacity, sizeof (itty_bit_string_t *));
        bool *before_false_negative_bits = calloc (target_bit_capacity, sizeof (bool));
        size_t candidate_count = 0;
        size_t candidate_capacity = 0;
        bool have_best_replay_candidate = false;
        bool best_replay_candidate_safe = true;
        bool projected_all = true;

        if (stats) {
                itty_bit_string_t *before_folded = NULL;
                if (itty_feed_model_fold_suffix_selected_output (model,
                                                                 layer_outputs,
                                                                 penultimate_layer,
                                                                 target,
                                                                 &before_folded)) {
                        for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++)
                                before_false_negative_bits[bit_index] =
                                        itty_bit_string_get_bit (target,
                                                                 bit_index) &&
                                        !itty_bit_string_get_bit (before_folded,
                                                                  bit_index);
                        itty_bit_string_free (before_folded);
                }
        }

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_t *current_condensed = itty_feed_model_run_node_condensed (layer_input,
                                                                                           model->masks_by_layer_node[penultimate_layer][node_index]);
                condensed_targets[node_index] = itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                                           condensed_words);
                condensed_cares[node_index] = itty_feed_model_zero_mask_new (condensed_words);
                residual_cares[node_index] = itty_feed_model_zero_mask_new (condensed_words);
                output_targets[node_index] = itty_feed_model_bit_string_clone_to_words (itty_bit_string_list_fetch (layer_outputs,
                                                                                                                    node_index),
                                                                                        output_words);
                output_cares[node_index] = itty_feed_model_zero_mask_new (output_words);
                structural_output_targets[node_index] = itty_feed_model_bit_string_clone_to_words (itty_bit_string_list_fetch (layer_outputs,
                                                                                                                               node_index),
                                                                                                   output_words);
                structural_output_cares[node_index] = itty_feed_model_zero_mask_new (output_words);
                itty_bit_string_free (current_condensed);
        }
        for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                replay_safe_final_votes_by_decoded_bit[bit_index] = itty_feed_model_zero_mask_new (output_words);
                replay_unsafe_final_votes_by_decoded_bit[bit_index] = itty_feed_model_zero_mask_new (output_words);
                selected_safe_final_votes_by_decoded_bit[bit_index] = itty_feed_model_zero_mask_new (output_words);
        }

        for (size_t repair_index = 0; repair_index < repairs.count; repair_index++) {
                if (max_projected_blocks != 0 &&
                    projected_blocks >= max_projected_blocks)
                        break;

                itty_feed_model_final_repair_t *repair = &repairs.items[repair_index];

                itty_feed_model_layer_assignment_list_t assignments = { 0 };
                if (!itty_feed_model_project_repair_to_previous_layer_outputs (model,
                                                                               final_layer,
                                                                               layer_outputs,
                                                                               repair,
                                                                               &assignments)) {
                        itty_feed_model_layer_assignment_list_clear (&assignments);
                        continue;
                }

                projected_blocks++;
                if (stats)
                        stats->projected_blocks = projected_blocks;

                itty_feed_model_layer_assignment_list_t condensed_assignments = { 0 };
                if (!itty_feed_model_make_condensed_assignments_from_outputs (model,
                                                                              penultimate_layer,
                                                                              &assignments,
                                                                              &condensed_assignments)) {
                        if (stats)
                                stats->conflicts++;
                        itty_feed_model_layer_assignment_list_clear (&condensed_assignments);
                        itty_feed_model_layer_assignment_list_clear (&assignments);
                        continue;
                }

                itty_feed_model_decoder_objective_t candidate_objective = { 0 };
                bool scored = itty_feed_model_score_condensed_realistic_candidate (model,
                                                                                  penultimate_layer,
                                                                                  layer_outputs,
                                                                                  target,
                                                                                  &condensed_assignments,
                                                                                  &candidate_objective);
                bool accepted = scored &&
                                itty_feed_model_decoder_objective_accepts (&before_objective,
                                                                           &candidate_objective);

                if (stats && scored) {
                        stats->condensed_realistic_blocks++;
                        if (candidate_objective.selected_distance < before_objective.selected_distance)
                                stats->condensed_realistic_strict_distance_helpful_blocks++;
                        else if (candidate_objective.selected_distance > before_objective.selected_distance)
                                stats->condensed_realistic_harmful_blocks++;
                        else if (candidate_objective.false_negative_blocker_bits < before_objective.false_negative_blocker_bits)
                                stats->condensed_realistic_blocker_helpful_blocks++;
                        else if (accepted)
                                stats->condensed_realistic_objective_helpful_blocks++;
                        else
                                stats->condensed_realistic_neutral_blocks++;
                }

                if (!accepted) {
                        itty_feed_model_layer_assignment_list_clear (&condensed_assignments);
                        itty_feed_model_layer_assignment_list_clear (&assignments);
                        continue;
                }

                if (candidate_count == candidate_capacity) {
                        candidate_capacity = candidate_capacity == 0 ? 64 : candidate_capacity * 2;
                        candidates = realloc (candidates,
                                              candidate_capacity * sizeof (itty_feed_model_projected_repair_candidate_t));
                }

                itty_feed_model_projected_repair_candidate_t *candidate = &candidates[candidate_count];
                *candidate = (itty_feed_model_projected_repair_candidate_t) {
                        .objective = candidate_objective,
                        .final_node = repair->final_node,
                        .final_output_bit = repair->output_bit,
                        .original_index = repair_index,
                        .decoded_bit = repair->decoded_bit,
                        .quota_size = repair->quota_size,
                        .distance_delta = itty_feed_model_positive_delta (before_objective.selected_distance,
                                                                          candidate_objective.selected_distance),
                        .false_negative_count_delta = itty_feed_model_positive_delta (before_objective.false_negative_count,
                                                                                      candidate_objective.false_negative_count),
                        .blocker_delta = itty_feed_model_positive_delta (before_objective.false_negative_blocker_bits,
                                                                         candidate_objective.false_negative_blocker_bits),
                        .vote_deficit_delta = itty_feed_model_positive_delta (before_objective.false_negative_vote_deficit,
                                                                              candidate_objective.false_negative_vote_deficit),
                        .target_one_margin_delta = itty_feed_model_positive_increase (before_objective.target_one_margin,
                                                                                      candidate_objective.target_one_margin),
                        .prefer_blocker_efficiency = prefer_blocker_efficiency,
                        .rank = candidate_objective.selected_distance < before_objective.selected_distance ?
                                ITTY_FEED_MODEL_PROJECTED_REPAIR_STRICT_DISTANCE :
                                (candidate_objective.false_negative_blocker_bits < before_objective.false_negative_blocker_bits ?
                                 ITTY_FEED_MODEL_PROJECTED_REPAIR_BLOCKER :
                                 ITTY_FEED_MODEL_PROJECTED_REPAIR_OBJECTIVE)
                };

                itty_feed_model_layer_assignment_list_copy (&candidate->output_assignments,
                                                            &assignments);
                itty_feed_model_layer_assignment_list_copy (&candidate->condensed_assignments,
                                                            &condensed_assignments);
                itty_feed_model_measure_condensed_assignment_cost (model,
                                                                  penultimate_layer,
                                                                  layer_input,
                                                                  &candidate->condensed_assignments,
                                                                  &candidate->estimated_flips,
                                                                  &candidate->already_satisfied_bits,
                                                                  &candidate->bits_needing_flips,
                                                                  &candidate->available_flippable_votes);
                if (!itty_feed_model_projected_candidate_has_decoder_effect (candidate,
                                                                             &before_objective,
                                                                             stats)) {
                        itty_feed_model_projected_repair_candidate_clear (candidate);
                        itty_feed_model_layer_assignment_list_clear (&condensed_assignments);
                        itty_feed_model_layer_assignment_list_clear (&assignments);
                        continue;
                }
                if (use_or_residual_repairs) {
                        size_t residual_estimated_flips = 0;
                        size_t residual_enable_flips = 0;
                        size_t residual_mask_flips = 0;

                        if (itty_feed_model_measure_or_residual_assignment_cost (model,
                                                                                 penultimate_layer,
                                                                                 layer_input,
                                                                                 &candidate->condensed_assignments,
                                                                                 &residual_estimated_flips,
                                                                                 &residual_enable_flips,
                                                                                 &residual_mask_flips)) {
                                if (stats)
                                        stats->residual_candidate_blocks++;
                                if (residual_estimated_flips < candidate->estimated_flips &&
                                    (!require_residual_blocker_efficiency ||
                                     candidate->blocker_delta >= residual_estimated_flips)) {
                                        candidate->estimated_flips = residual_estimated_flips;
                                        candidate->residual_enable_flips = residual_enable_flips;
                                        candidate->residual_mask_flips = residual_mask_flips;
                                        candidate->use_residual = true;
                                }
                        }
                }
                if (replay_example_count > 0) {
                        itty_feed_model_decoder_objective_t *before_replay_objectives = calloc (replay_example_count,
                                                                                                 sizeof (itty_feed_model_decoder_objective_t));
                        itty_bit_string_t **before_replay_folded = calloc (replay_example_count,
                                                                           sizeof (itty_bit_string_t *));

                        if (itty_feed_model_measure_replay_examples (model,
                                                                     replay_examples,
                                                                     replay_example_count,
                                                                     before_replay_objectives,
                                                                     before_replay_folded)) {
                                candidate->replay_collateral_cost =
                                        itty_feed_model_measure_replay_mask_flip_sensitivity (model,
                                                                                              penultimate_layer,
                                                                                              layer_input,
                                                                                              target,
                                                                                              &before_objective,
                                                                                              candidate,
                                                                                              replay_examples,
                                                                                              replay_example_count,
                                                                                              before_replay_objectives,
                                                                                              before_replay_folded,
                                                                                              options ? options->replay_zero_protection_penalty : 0,
                                                                                              options ? options->replay_one_protection_penalty : 0,
                                                                                              &bad_flip_frequencies,
                                                                                              stats);
                        }

                        for (size_t replay_index = 0; replay_index < replay_example_count; replay_index++)
                                if (before_replay_folded[replay_index])
                                        itty_bit_string_free (before_replay_folded[replay_index]);
                        free (before_replay_folded);
                        free (before_replay_objectives);
                }
                itty_feed_model_refreshed_projected_repair_round_t replay_candidate_stats = { 0 };
                bool replay_safe_candidate = itty_feed_model_projected_candidate_is_replay_safe (model,
                                                                                                 penultimate_layer,
                                                                                                 layer_input,
                                                                                                 candidate,
                                                                                                 replay_examples,
                                                                                                 replay_example_count,
                                                                                                 strict_replay_guard,
                                                                                                 &replay_candidate_stats);
                if (replay_example_count > 0 &&
                    (!have_best_replay_candidate ||
                     compare_projected_repair_candidates (candidate,
                                                          &best_replay_candidate) < 0)) {
                        best_replay_candidate = *candidate;
                        best_replay_candidate_safe = replay_safe_candidate;
                        have_best_replay_candidate = true;
                }
                if (!replay_safe_candidate) {
                        if (stats) {
                                stats->replay_unsafe_candidates++;
                                if (replay_candidate_stats.replay_transitions.correct_zero_to_false_positive_bits > 0)
                                        stats->replay_realization_collateral_false_positive_candidates++;
                                if (candidate->decoded_bit < target_bit_capacity) {
                                        replay_unsafe_votes_by_decoded_bit[candidate->decoded_bit]++;
                                        if (candidate->quota_size > replay_quota_by_decoded_bit[candidate->decoded_bit])
                                                replay_quota_by_decoded_bit[candidate->decoded_bit] = candidate->quota_size;
                                        if (candidate->final_output_bit < output_bit_capacity)
                                                itty_bit_string_set_bit (replay_unsafe_final_votes_by_decoded_bit[candidate->decoded_bit],
                                                                         candidate->final_output_bit,
                                                                         true);
                                }
                                itty_feed_model_accumulate_replay_unsafe_candidate_stats (stats,
                                                                                          &replay_candidate_stats);
                        }
                        itty_feed_model_projected_repair_candidate_clear (candidate);
                        itty_feed_model_layer_assignment_list_clear (&condensed_assignments);
                        itty_feed_model_layer_assignment_list_clear (&assignments);
                        continue;
                }
                if (stats && replay_example_count > 0) {
                        stats->replay_safe_candidates++;
                        itty_feed_model_accumulate_replay_safe_candidate_usefulness (stats,
                                                                                     candidate,
                                                                                     &before_objective);
                        if (candidate->decoded_bit < target_bit_capacity) {
                                replay_safe_votes_by_decoded_bit[candidate->decoded_bit]++;
                                if (candidate->quota_size > replay_quota_by_decoded_bit[candidate->decoded_bit])
                                        replay_quota_by_decoded_bit[candidate->decoded_bit] = candidate->quota_size;
                                if (candidate->final_output_bit < output_bit_capacity)
                                        itty_bit_string_set_bit (replay_safe_final_votes_by_decoded_bit[candidate->decoded_bit],
                                                                 candidate->final_output_bit,
                                                                 true);
                        }
                }
                candidate_count++;

                itty_feed_model_layer_assignment_list_clear (&condensed_assignments);
                itty_feed_model_layer_assignment_list_clear (&assignments);
        }
        if (stats &&
            have_best_replay_candidate &&
            !best_replay_candidate_safe)
                stats->replay_best_candidate_unsafe = 1;
        if (stats && replay_example_count > 0) {
                for (size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
                        itty_feed_model_projected_repair_candidate_t *candidate = &candidates[candidate_index];

                        if (candidate->decoded_bit >= target_bit_capacity ||
                            replay_quota_by_decoded_bit[candidate->decoded_bit] == 0)
                                continue;

                        if (itty_bit_string_get_pop_count (replay_safe_final_votes_by_decoded_bit[candidate->decoded_bit]) >=
                            replay_quota_by_decoded_bit[candidate->decoded_bit])
                                stats->replay_safe_quota_complete_candidates++;
                        else
                                stats->replay_safe_quota_incomplete_candidates++;
                }
                for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                        size_t safe_final_votes = itty_bit_string_get_pop_count (replay_safe_final_votes_by_decoded_bit[bit_index]);
                        size_t unsafe_final_votes = itty_bit_string_get_pop_count (replay_unsafe_final_votes_by_decoded_bit[bit_index]);
                        size_t duplicate_safe_votes = replay_safe_votes_by_decoded_bit[bit_index] > safe_final_votes ?
                                                      replay_safe_votes_by_decoded_bit[bit_index] - safe_final_votes : 0;
                        size_t duplicate_unsafe_votes = replay_unsafe_votes_by_decoded_bit[bit_index] > unsafe_final_votes ?
                                                        replay_unsafe_votes_by_decoded_bit[bit_index] - unsafe_final_votes : 0;

                        if (replay_quota_by_decoded_bit[bit_index] == 0)
                                continue;
                        if (stats &&
                            stats->replay_final_surface_feasibility_trace_count < ITTY_FEED_MODEL_FINAL_SURFACE_FEASIBILITY_TRACE_LIMIT) {
                                itty_feed_model_final_surface_block_reason_t blocked_reason = ITTY_FEED_MODEL_FINAL_SURFACE_BLOCK_NONE;
                                size_t shortfall = replay_quota_by_decoded_bit[bit_index] > safe_final_votes ?
                                                   replay_quota_by_decoded_bit[bit_index] - safe_final_votes : 0;

                                if (safe_final_votes < replay_quota_by_decoded_bit[bit_index]) {
                                        if (safe_final_votes + unsafe_final_votes >= replay_quota_by_decoded_bit[bit_index])
                                                blocked_reason = ITTY_FEED_MODEL_FINAL_SURFACE_BLOCK_REPLAY_UNSAFE;
                                        else if (duplicate_safe_votes > 0 || duplicate_unsafe_votes > 0)
                                                blocked_reason = ITTY_FEED_MODEL_FINAL_SURFACE_BLOCK_DUPLICATE_FINAL_VOTE;
                                        else
                                                blocked_reason = ITTY_FEED_MODEL_FINAL_SURFACE_BLOCK_NO_CANDIDATE_PATH;
                                }

                                stats->replay_final_surface_feasibility_traces[stats->replay_final_surface_feasibility_trace_count++] =
                                        (itty_feed_model_final_surface_feasibility_t) {
                                                .decoded_bit = bit_index,
                                                .needed_final_votes = replay_quota_by_decoded_bit[bit_index],
                                                .safe_final_votes_available = safe_final_votes,
                                                .unsafe_final_votes_available = unsafe_final_votes,
                                                .safe_final_votes_shortfall = shortfall,
                                                .duplicate_safe_final_votes = duplicate_safe_votes,
                                                .duplicate_unsafe_final_votes = duplicate_unsafe_votes,
                                                .blocked_reason = blocked_reason,
                                        };
                        }
                        if (itty_bit_string_get_pop_count (replay_safe_final_votes_by_decoded_bit[bit_index]) >=
                            replay_quota_by_decoded_bit[bit_index])
                                stats->replay_safe_quota_feasible_decoded_bits++;
                        else if (replay_unsafe_votes_by_decoded_bit[bit_index] > 0)
                                stats->replay_safe_quota_blocked_decoded_bits++;
                }
        }
        itty_feed_model_bad_flip_frequency_list_finish (&bad_flip_frequencies,
                                                        stats);

        qsort (candidates,
               candidate_count,
               sizeof (itty_feed_model_projected_repair_candidate_t),
               compare_projected_repair_candidates);
        if (stats && candidate_count == 0 && projected_blocks > 0) {
                if (stats->replay_unsafe_candidates == 0) {
                        stats->no_effect_candidates += projected_blocks;
                        stats->no_effect_candidate_irrelevant_segment += projected_blocks;
                }
        }
        selected_candidates = calloc (candidate_count, sizeof (bool));

        itty_feed_model_decoder_objective_t previous_layer_before_objective = { 0 };
        bool have_previous_layer_objective = penultimate_layer > 0 &&
                                             itty_feed_model_evaluate_suffix_decoder_objective (model,
                                                                                                layer_input,
                                                                                                penultimate_layer - 1,
                                                                                                target,
                                                                                                &previous_layer_before_objective);
        size_t remaining_estimated_flips = max_layer_flips;
        size_t selected_estimated_flips = 0;
        size_t selected_normal_estimated_flips = 0;
        for (size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
                itty_feed_model_projected_repair_candidate_t *candidate = &candidates[candidate_index];

                if (replay_safe_quota_complete_only &&
                    replay_example_count > 0 &&
                    candidate->decoded_bit < target_bit_capacity &&
                    replay_quota_by_decoded_bit[candidate->decoded_bit] > 0 &&
                    itty_bit_string_get_pop_count (replay_safe_final_votes_by_decoded_bit[candidate->decoded_bit]) <
                    replay_quota_by_decoded_bit[candidate->decoded_bit])
                        continue;
                if (max_blocks_per_final_node != 0 &&
                    blocks_by_final_node[candidate->final_node] >= max_blocks_per_final_node)
                        continue;
                if (limit_strict_distance_blocks &&
                    candidate->rank == ITTY_FEED_MODEL_PROJECTED_REPAIR_STRICT_DISTANCE &&
                    stats &&
                    stats->accepted_strict_distance_blocks >= max_strict_distance_blocks)
                        continue;
                if (limit_blocker_blocks &&
                    candidate->rank == ITTY_FEED_MODEL_PROJECTED_REPAIR_BLOCKER &&
                    stats &&
                    stats->accepted_blocker_blocks >= max_blocker_blocks)
                        continue;

                bool conflicted = false;
                for (size_t assignment_index = 0; assignment_index < candidate->condensed_assignments.count; assignment_index++) {
                        itty_feed_model_layer_assignment_t *assignment = &candidate->condensed_assignments.items[assignment_index];

                        if (itty_bit_string_get_bit (condensed_cares[assignment->layer_node],
                                                     assignment->bit_index) &&
                            itty_bit_string_get_bit (condensed_targets[assignment->layer_node],
                                                     assignment->bit_index) != assignment->value) {
                                conflicted = true;
                                break;
                        }
                        if (itty_bit_string_get_bit (residual_cares[assignment->layer_node],
                                                     assignment->bit_index) &&
                            itty_bit_string_get_bit (condensed_targets[assignment->layer_node],
                                                     assignment->bit_index) != assignment->value) {
                                conflicted = true;
                                break;
                        }
                }
                if (!conflicted) {
                        for (size_t assignment_index = 0; assignment_index < candidate->output_assignments.count; assignment_index++) {
                                itty_feed_model_layer_assignment_t *assignment = &candidate->output_assignments.items[assignment_index];

                                if (itty_bit_string_get_bit (output_cares[assignment->layer_node],
                                                             assignment->bit_index) &&
                                    itty_bit_string_get_bit (output_targets[assignment->layer_node],
                                                             assignment->bit_index) != assignment->value) {
                                        conflicted = true;
                                        break;
                                }
                        }
                }

                if (conflicted) {
                        if (stats)
                                stats->conflicts++;
                        continue;
                }
                if (candidate->decoded_bit < target_bit_capacity &&
                    candidate->final_output_bit < output_bit_capacity &&
                    itty_bit_string_get_bit (selected_safe_final_votes_by_decoded_bit[candidate->decoded_bit],
                                             candidate->final_output_bit))
                        continue;

                size_t marginal_estimated_flips = candidate->use_residual ?
                                                  candidate->estimated_flips :
                                                  itty_feed_model_measure_candidate_marginal_cost (model,
                                                                                                   penultimate_layer,
                                                                                                   layer_input,
                                                                                                   candidate,
                                                                                                   condensed_cares,
                                                                                                   condensed_targets);
                if (max_layer_flips != 0 &&
                    marginal_estimated_flips > remaining_estimated_flips)
                        continue;

                if (stats) {
                        bool current_candidate_negative =
                                candidate->objective.selected_distance > before_objective.selected_distance ||
                                candidate->objective.false_negative_count > before_objective.false_negative_count ||
                                candidate->objective.false_positive_vote_excess > before_objective.false_positive_vote_excess ||
                                candidate->objective.target_one_margin < before_objective.target_one_margin ||
                                candidate->objective.target_zero_safety_min < before_objective.target_zero_safety_min;
                        bool current_candidate_positive =
                                candidate->objective.selected_distance < before_objective.selected_distance ||
                                candidate->objective.false_negative_count < before_objective.false_negative_count ||
                                candidate->objective.false_positive_vote_excess < before_objective.false_positive_vote_excess ||
                                candidate->objective.target_one_margin > before_objective.target_one_margin ||
                                candidate->objective.target_zero_safety_min > before_objective.target_zero_safety_min;

                        if (current_candidate_negative)
                                stats->current_candidate_net_negative++;
                        else if (current_candidate_positive)
                                stats->current_candidate_net_positive++;
                        else
                                stats->current_candidate_net_zero++;
                }

                selected_candidates[candidate_index] = true;
                blocks_by_final_node[candidate->final_node]++;
                selected_estimated_flips += marginal_estimated_flips;
                if (!candidate->use_residual)
                        selected_normal_estimated_flips += marginal_estimated_flips;
                if (max_layer_flips != 0)
                        remaining_estimated_flips -= marginal_estimated_flips;
                if (candidate->decoded_bit < target_bit_capacity) {
                        flips_by_decoded_bit[candidate->decoded_bit] += marginal_estimated_flips;
                        if (replay_quota_by_decoded_bit[candidate->decoded_bit] > 0 &&
                            itty_bit_string_get_pop_count (replay_safe_final_votes_by_decoded_bit[candidate->decoded_bit]) >=
                            replay_quota_by_decoded_bit[candidate->decoded_bit]) {
                                selected_safe_votes_by_decoded_bit[candidate->decoded_bit]++;
                                selected_safe_flips_by_decoded_bit[candidate->decoded_bit] += marginal_estimated_flips;
                                if (candidate->final_output_bit < output_bit_capacity)
                                        itty_bit_string_set_bit (selected_safe_final_votes_by_decoded_bit[candidate->decoded_bit],
                                                                 candidate->final_output_bit,
                                                                 true);
                        }
                }
                if (stats) {
                        stats->accepted_blocks++;
                        stats->quota_size_total += candidate->quota_size;
                        if (candidate->quota_size > stats->quota_size_max)
                                stats->quota_size_max = candidate->quota_size;
                        if (candidate->rank == ITTY_FEED_MODEL_PROJECTED_REPAIR_STRICT_DISTANCE)
                                stats->accepted_strict_distance_blocks++;
                        else if (candidate->rank == ITTY_FEED_MODEL_PROJECTED_REPAIR_BLOCKER)
                                stats->accepted_blocker_blocks++;
                        stats->estimated_layer_flips += marginal_estimated_flips;
                        stats->already_satisfied_condensed_bits += candidate->already_satisfied_bits;
                        stats->condensed_bits_needing_flips += candidate->bits_needing_flips;
                        stats->available_flippable_votes += candidate->available_flippable_votes;
                        if (candidate->use_residual)
                                stats->residual_accepted_blocks++;
                        if (have_previous_layer_objective)
                                itty_feed_model_measure_previous_layer_projection (model,
                                                                                  penultimate_layer,
                                                                                  layer_input,
                                                                                  target,
                                                                                  &previous_layer_before_objective,
                                                                                  &candidate->condensed_assignments,
                                                                                  stats);
                }

                for (size_t assignment_index = 0; assignment_index < candidate->output_assignments.count; assignment_index++) {
                        itty_feed_model_layer_assignment_t *assignment = &candidate->output_assignments.items[assignment_index];
                        if (!itty_bit_string_get_bit (output_cares[assignment->layer_node],
                                                      assignment->bit_index)) {
                                if (stats)
                                        stats->requested_output_bits++;
                                itty_bit_string_set_bit (output_cares[assignment->layer_node],
                                                         assignment->bit_index,
                                                         true);
                        }
                        itty_bit_string_set_bit (output_targets[assignment->layer_node],
                                                 assignment->bit_index,
                                                 assignment->value);

                        output_cares[assignment->layer_node]->pop_count_computed = false;
                        output_targets[assignment->layer_node]->pop_count_computed = false;
                }

                for (size_t assignment_index = 0; assignment_index < candidate->condensed_assignments.count; assignment_index++) {
                        itty_feed_model_layer_assignment_t *assignment = &candidate->condensed_assignments.items[assignment_index];
                        itty_bit_string_t *care_bits = candidate->use_residual ?
                                                       residual_cares[assignment->layer_node] :
                                                       condensed_cares[assignment->layer_node];

                        if (!itty_bit_string_get_bit (care_bits,
                                                      assignment->bit_index)) {
                                if (stats) {
                                        size_t support_cost = candidate->use_residual ? 1 :
                                                              itty_feed_model_measure_condensed_assignment_flip_cost (model,
                                                                                                                     penultimate_layer,
                                                                                                                     layer_input,
                                                                                                                     assignment);
                                        stats->requested_condensed_bits++;
                                        itty_feed_model_decoder_histogram_increment (stats->quota_vote_support_cost_histogram,
                                                                                     support_cost);
                                }
                                itty_bit_string_set_bit (care_bits,
                                                         assignment->bit_index,
                                                         true);
                        }
                        itty_bit_string_set_bit (condensed_targets[assignment->layer_node],
                                                 assignment->bit_index,
                                                 assignment->value);

                        care_bits->pop_count_computed = false;
                        condensed_targets[assignment->layer_node]->pop_count_computed = false;
                }

                if (candidate->use_residual) {
                        size_t enable_flips = 0;
                        size_t mask_flips = 0;

                        itty_feed_model_apply_or_residual_assignments (model,
                                                                       penultimate_layer,
                                                                       layer_input,
                                                                       &candidate->condensed_assignments,
                                                                       &enable_flips,
                                                                       &mask_flips);
                        if (stats) {
                                stats->residual_enable_flips += enable_flips;
                                stats->residual_mask_flips += mask_flips;
                                stats->layer_flips += enable_flips + mask_flips;
                        }
                }
        }

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_t *candidate_output = itty_feed_model_expand_condensed_output (condensed_targets[node_index],
                                                                                               model->rotations_by_layer[penultimate_layer]);
                itty_bit_string_t *before_output = itty_bit_string_list_fetch (layer_outputs,
                                                                               node_index);
                size_t output_bit_capacity = output_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

                for (size_t bit_index = 0; bit_index < output_bit_capacity; bit_index++) {
                        bool before_bit = itty_bit_string_get_bit (before_output,
                                                                   bit_index);
                        bool structural_bit = itty_bit_string_get_bit (candidate_output,
                                                                       bit_index);

                        if (before_bit == structural_bit)
                                continue;

                        itty_bit_string_set_bit (structural_output_cares[node_index],
                                                 bit_index,
                                                 true);
                        itty_bit_string_set_bit (structural_output_targets[node_index],
                                                 bit_index,
                                                 structural_bit);
                }
                structural_output_cares[node_index]->pop_count_computed = false;
                structural_output_targets[node_index]->pop_count_computed = false;
                itty_bit_string_free (candidate_output);
        }

        size_t remaining_flips = max_layer_flips == 0 ? 0 : selected_normal_estimated_flips;
        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_feed_model_train_stats_t node_stats = { 0 };
                size_t node_max_flips = remaining_flips;

                if (!itty_feed_model_train_layer_one_node_with_care (model->masks_by_layer_node[penultimate_layer][node_index],
                                                                     layer_input,
                                                                     condensed_targets[node_index],
                                                                     condensed_cares[node_index],
                                                                     node_max_flips,
                                                                     &node_stats)) {
                        projected_all = false;
                        break;
                }

                if (stats)
                        stats->layer_flips += node_stats.flips;

                if (max_layer_flips != 0) {
                        if (node_stats.flips >= remaining_flips)
                                remaining_flips = 0;
                        else
                                remaining_flips -= node_stats.flips;

                        if (remaining_flips == 0)
                                break;
                }
        }

        itty_bit_string_list_t *after_layer_outputs = itty_feed_model_run_layer (model,
                                                                                 penultimate_layer,
                                                                                 layer_input);
        if (stats) {
                size_t output_bit_capacity = output_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
                size_t condensed_bit_capacity = condensed_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
                itty_bit_string_t **after_condensed = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));

                for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                        after_condensed[node_index] = itty_feed_model_reduce_desired_output_for_layer (itty_bit_string_list_fetch (after_layer_outputs,
                                                                                                                                  node_index),
                                                                                                       model->rotations_by_layer[penultimate_layer]);
                        stats->residual_active_bits += itty_bit_string_get_pop_count (model->residual_enable_by_layer_node[penultimate_layer][node_index]);

                        for (size_t bit_index = 0; bit_index < condensed_bit_capacity; bit_index++) {
                                if (!itty_bit_string_get_bit (condensed_cares[node_index],
                                                              bit_index) &&
                                    !itty_bit_string_get_bit (residual_cares[node_index],
                                                              bit_index))
                                        continue;

                                if (itty_bit_string_get_bit (after_condensed[node_index],
                                                             bit_index) ==
                                    itty_bit_string_get_bit (condensed_targets[node_index],
                                                             bit_index))
                                        stats->realized_condensed_bits++;
                        }
                }

                for (size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
                        if (!selected_candidates[candidate_index])
                                continue;

                        itty_feed_model_projected_repair_candidate_t *candidate = &candidates[candidate_index];
                        size_t realized_assignments = 0;

                        for (size_t assignment_index = 0; assignment_index < candidate->condensed_assignments.count; assignment_index++) {
                                itty_feed_model_layer_assignment_t *assignment = &candidate->condensed_assignments.items[assignment_index];

                                if (itty_bit_string_get_bit (after_condensed[assignment->layer_node],
                                                             assignment->bit_index) == assignment->value)
                                        realized_assignments++;
                        }

                        if (realized_assignments == candidate->condensed_assignments.count)
                                stats->fully_realized_blocks++;
                        else if (realized_assignments > 0)
                                stats->partially_realized_blocks++;
                        else
                                stats->unrealized_blocks++;
                }

                for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                        itty_bit_string_t *before_output = itty_bit_string_list_fetch (layer_outputs,
                                                                                       node_index);
                        itty_bit_string_t *after_output = itty_bit_string_list_fetch (after_layer_outputs,
                                                                                      node_index);

                        for (size_t bit_index = 0; bit_index < output_bit_capacity; bit_index++) {
                                bool before_bit = itty_bit_string_get_bit (before_output,
                                                                           bit_index);
                                bool after_bit = itty_bit_string_get_bit (after_output,
                                                                          bit_index);
                                bool cared = itty_bit_string_get_bit (output_cares[node_index],
                                                                      bit_index);

                                if (cared) {
                                        if (after_bit == itty_bit_string_get_bit (output_targets[node_index],
                                                                                 bit_index))
                                                stats->realized_output_bits++;
                                        else
                                                stats->lost_output_bits++;
                                } else if (before_bit != after_bit) {
                                        stats->extra_output_bits++;
                                        if (itty_bit_string_get_bit (structural_output_cares[node_index],
                                                                    bit_index) &&
                                            after_bit == itty_bit_string_get_bit (structural_output_targets[node_index],
                                                                                 bit_index))
                                                stats->structural_extra_output_bits++;
                                        else
                                                stats->collateral_extra_output_bits++;
                                }
                        }
                }

                for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++)
                        itty_bit_string_free (after_condensed[node_index]);
                free (after_condensed);

                itty_feed_model_decoder_objective_t after_objective = { 0 };
                if (itty_feed_model_evaluate_suffix_decoder_objective (model,
                                                                       after_layer_outputs,
                                                                       penultimate_layer,
                                                                       target,
                                                                       &after_objective)) {
                        stats->after_distance = after_objective.selected_distance;
                        stats->after_blockers = after_objective.false_negative_blocker_bits;
                }

                itty_bit_string_t *after_folded = NULL;
                if (itty_feed_model_fold_suffix_selected_output (model,
                                                                 after_layer_outputs,
                                                                 penultimate_layer,
                                                                 target,
                                                                 &after_folded)) {
                        for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                                bool target_bit = itty_bit_string_get_bit (target,
                                                                           bit_index);
                                bool after_bit = itty_bit_string_get_bit (after_folded,
                                                                          bit_index);

                                if (!before_false_negative_bits[bit_index] || after_bit != target_bit)
                                        continue;

                                stats->decoded_bits_fixed++;
                                stats->fixed_decoded_bit_flips_total += flips_by_decoded_bit[bit_index];
                                if (stats->decoded_bits_fixed == 1 ||
                                    flips_by_decoded_bit[bit_index] < stats->fixed_decoded_bit_flips_min)
                                        stats->fixed_decoded_bit_flips_min = flips_by_decoded_bit[bit_index];
                                if (flips_by_decoded_bit[bit_index] > stats->fixed_decoded_bit_flips_max)
                                        stats->fixed_decoded_bit_flips_max = flips_by_decoded_bit[bit_index];
                                if (replay_quota_by_decoded_bit[bit_index] > 0 &&
                                    itty_bit_string_get_pop_count (replay_safe_final_votes_by_decoded_bit[bit_index]) >= replay_quota_by_decoded_bit[bit_index] &&
                                    itty_bit_string_get_pop_count (selected_safe_final_votes_by_decoded_bit[bit_index]) >= replay_quota_by_decoded_bit[bit_index])
                                        stats->replay_safe_quota_distance_flip_decoded_bits++;
                        }
                }
                if (after_folded)
                        itty_bit_string_free (after_folded);

                if (replay_example_count > 0) {
                        size_t segment_vote_threshold = output_bit_capacity / target_bit_capacity / 2 + 1;
                        itty_bit_string_list_t *before_final_outputs = NULL;
                        itty_bit_string_list_t *after_final_outputs = NULL;
                        itty_feed_model_output_evaluation_t before_final_evaluation = { 0 };
                        itty_feed_model_output_evaluation_t after_final_evaluation = { 0 };
                        bool have_before_final = itty_feed_model_run_suffix_outputs (model,
                                                                                     layer_outputs,
                                                                                     penultimate_layer,
                                                                                     &before_final_outputs) &&
                                                itty_feed_model_evaluate_output (model,
                                                                                 before_final_outputs,
                                                                                 target,
                                                                                 &before_final_evaluation);
                        bool have_after_final = itty_feed_model_run_suffix_outputs (model,
                                                                                    after_layer_outputs,
                                                                                    penultimate_layer,
                                                                                    &after_final_outputs) &&
                                               itty_feed_model_evaluate_output (model,
                                                                                after_final_outputs,
                                                                                target,
                                                                                &after_final_evaluation);

                        for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                                size_t safe_final_votes = itty_bit_string_get_pop_count (replay_safe_final_votes_by_decoded_bit[bit_index]);
                                size_t selected_safe_final_votes = itty_bit_string_get_pop_count (selected_safe_final_votes_by_decoded_bit[bit_index]);

                                if (replay_quota_by_decoded_bit[bit_index] == 0 ||
                                    safe_final_votes < replay_quota_by_decoded_bit[bit_index])
                                        continue;

                                if (selected_safe_final_votes > 0) {
                                        stats->replay_safe_quota_selected_decoded_bits++;
                                        stats->replay_safe_quota_local_votes_selected += selected_safe_votes_by_decoded_bit[bit_index];
                                        if (selected_safe_flips_by_decoded_bit[bit_index] > 0 ||
                                            selected_safe_final_votes > 0)
                                                stats->replay_safe_quota_accepted_decoded_bits++;
                                }
                                if (selected_safe_final_votes >= replay_quota_by_decoded_bit[bit_index]) {
                                        stats->replay_safe_quota_completed_decoded_bits++;
                                }
                                if (stats->replay_safe_quota_effect_trace_count < ITTY_FEED_MODEL_SAFE_QUOTA_EFFECT_TRACE_LIMIT) {
                                        size_t penultimate_ones_before = itty_feed_model_count_selected_segment_votes (layer_outputs,
                                                                                                                       before_objective.selected_node,
                                                                                                                       bit_index,
                                                                                                                       target_bit_capacity);
                                        size_t penultimate_ones_after = itty_feed_model_count_selected_segment_votes (after_layer_outputs,
                                                                                                                      after_objective.selected_node,
                                                                                                                      bit_index,
                                                                                                                      target_bit_capacity);
                                        size_t final_condensed_ones_before = have_before_final ?
                                                itty_feed_model_count_selected_condensed_votes (model,
                                                                                               before_final_outputs,
                                                                                               final_layer,
                                                                                               before_final_evaluation.selected_index,
                                                                                               bit_index,
                                                                                               target_bit_capacity) : 0;
                                        size_t final_condensed_ones_after = have_after_final ?
                                                itty_feed_model_count_selected_condensed_votes (model,
                                                                                               after_final_outputs,
                                                                                               final_layer,
                                                                                               after_final_evaluation.selected_index,
                                                                                               bit_index,
                                                                                               target_bit_capacity) : 0;
                                        size_t final_segment_ones_before = have_before_final ?
                                                itty_feed_model_count_selected_segment_votes (before_final_outputs,
                                                                                             before_final_evaluation.selected_index,
                                                                                             bit_index,
                                                                                             target_bit_capacity) : 0;
                                        size_t final_segment_ones_after = have_after_final ?
                                                itty_feed_model_count_selected_segment_votes (after_final_outputs,
                                                                                            after_final_evaluation.selected_index,
                                                                                            bit_index,
                                                                                            target_bit_capacity) : 0;
                                        bool final_decoded_before = have_before_final ?
                                                itty_bit_string_get_bit (before_final_evaluation.folded_activation,
                                                                         bit_index) : false;
                                        bool final_decoded_after = have_after_final ?
                                                itty_bit_string_get_bit (after_final_evaluation.folded_activation,
                                                                         bit_index) : false;
                                        itty_feed_model_safe_quota_effect_t *trace =
                                                &stats->replay_safe_quota_effect_traces[stats->replay_safe_quota_effect_trace_count++];
                                        *trace = (itty_feed_model_safe_quota_effect_t) {
                                                .decoded_bit = bit_index,
                                                .penultimate_selected_node_before = before_objective.selected_node,
                                                .penultimate_selected_node_after = after_objective.selected_node,
                                                .target_bit = itty_bit_string_get_bit (target,
                                                                                       bit_index) ? 1 : 0,
                                                .penultimate_ones_before = penultimate_ones_before,
                                                .penultimate_ones_after = penultimate_ones_after,
                                                .penultimate_threshold = segment_vote_threshold,
                                                .needed_before = replay_quota_by_decoded_bit[bit_index],
                                                .safe_votes_available = safe_final_votes,
                                                .safe_votes_selected = selected_safe_final_votes,
                                                .penultimate_votes_changed = penultimate_ones_after > penultimate_ones_before ?
                                                                            penultimate_ones_after - penultimate_ones_before : 0,
                                                .final_selected_node_before = have_before_final ? before_final_evaluation.selected_index : 0,
                                                .final_selected_node_after = have_after_final ? after_final_evaluation.selected_index : 0,
                                                .final_condensed_ones_before = final_condensed_ones_before,
                                                .final_condensed_ones_after = final_condensed_ones_after,
                                                .final_condensed_threshold = segment_vote_threshold,
                                                .final_segment_ones_before = final_segment_ones_before,
                                                .final_segment_ones_after = final_segment_ones_after,
                                                .final_segment_threshold = segment_vote_threshold,
                                                .final_distance_contribution_before = final_decoded_before == itty_bit_string_get_bit (target, bit_index) ? 0 : 1,
                                                .final_distance_contribution_after = final_decoded_after == itty_bit_string_get_bit (target, bit_index) ? 0 : 1,
                                                .quota_realized = selected_safe_final_votes >= replay_quota_by_decoded_bit[bit_index],
                                                .penultimate_threshold_crossed = penultimate_ones_before < segment_vote_threshold &&
                                                                                penultimate_ones_after >= segment_vote_threshold,
                                                .penultimate_decoded_before = penultimate_ones_before >= segment_vote_threshold,
                                                .penultimate_decoded_after = penultimate_ones_after >= segment_vote_threshold,
                                                .final_condensed_changed = final_condensed_ones_before != final_condensed_ones_after,
                                                .final_condensed_crossed = final_condensed_ones_before < segment_vote_threshold &&
                                                                          final_condensed_ones_after >= segment_vote_threshold,
                                                .final_segment_changed = final_segment_ones_before != final_segment_ones_after,
                                                .final_decoded_before = final_decoded_before,
                                                .final_decoded_after = final_decoded_after,
                                                .final_decode_crossed = !final_decoded_before && final_decoded_after,
                                        };
                                        if (trace->penultimate_threshold_crossed) {
                                                stats->replay_safe_quota_local_crossed_decoded_bits++;
                                                if (trace->final_selected_node_before != trace->final_selected_node_after)
                                                        stats->replay_safe_quota_lost_at_unselected_final_node++;
                                                else if (!trace->final_condensed_changed)
                                                        stats->replay_safe_quota_lost_at_final_majority++;
                                                else if (!trace->final_segment_changed)
                                                        stats->replay_safe_quota_lost_at_final_rotation_or_expansion++;
                                                else if (!trace->final_decode_crossed) {
                                                        if (trace->final_segment_ones_after < trace->final_segment_threshold)
                                                                stats->replay_safe_quota_lost_at_segment_vote_margin++;
                                                        else
                                                                stats->replay_safe_quota_lost_due_to_duplicate_final_segment++;
                                                }
                                        }
                                        if (selected_safe_votes_by_decoded_bit[bit_index] >= replay_quota_by_decoded_bit[bit_index])
                                                stats->replay_safe_quota_local_realized_decoded_bits++;
                                        stats->replay_safe_quota_unique_local_votes_changed += trace->penultimate_votes_changed;
                                        if (trace->final_condensed_changed)
                                                stats->replay_safe_quota_unique_final_condensed_bits_affected +=
                                                        trace->final_condensed_ones_after > trace->final_condensed_ones_before ?
                                                        trace->final_condensed_ones_after - trace->final_condensed_ones_before :
                                                        trace->final_condensed_ones_before - trace->final_condensed_ones_after;
                                        if (trace->final_segment_changed)
                                                stats->replay_safe_quota_unique_final_segment_votes_affected +=
                                                        trace->final_segment_ones_after > trace->final_segment_ones_before ?
                                                        trace->final_segment_ones_after - trace->final_segment_ones_before :
                                                        trace->final_segment_ones_before - trace->final_segment_ones_after;
                                        if (trace->final_decoded_before != trace->final_decoded_after)
                                                stats->replay_safe_quota_unique_decoded_bits_affected++;
                                        if (trace->final_condensed_changed)
                                                stats->replay_safe_quota_final_vote_reached++;
                                        if (trace->final_segment_changed)
                                                stats->replay_safe_quota_final_vote_changed++;
                                        if (trace->final_condensed_crossed)
                                                stats->replay_safe_quota_final_vote_crossed++;
                                        if (trace->final_decoded_before != trace->final_decoded_after)
                                                stats->replay_safe_quota_decoded_vote_changed++;
                                        if (trace->final_decode_crossed)
                                                stats->replay_safe_quota_decoded_threshold_crossed++;
                                }
                        }

                        if (before_final_outputs)
                                itty_bit_string_list_free (before_final_outputs);
                        if (after_final_outputs)
                                itty_bit_string_list_free (after_final_outputs);
                        if (have_before_final)
                                itty_bit_string_free (before_final_evaluation.folded_activation);
                        if (have_after_final)
                                itty_bit_string_free (after_final_evaluation.folded_activation);
                }
        }

        itty_bit_string_list_free (after_layer_outputs);

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_free (condensed_targets[node_index]);
                itty_bit_string_free (condensed_cares[node_index]);
                itty_bit_string_free (residual_cares[node_index]);
                itty_bit_string_free (output_targets[node_index]);
                itty_bit_string_free (output_cares[node_index]);
                itty_bit_string_free (structural_output_targets[node_index]);
                itty_bit_string_free (structural_output_cares[node_index]);
        }
        for (size_t bit_index = 0; bit_index < target_bit_capacity; bit_index++) {
                itty_bit_string_free (replay_safe_final_votes_by_decoded_bit[bit_index]);
                itty_bit_string_free (replay_unsafe_final_votes_by_decoded_bit[bit_index]);
                itty_bit_string_free (selected_safe_final_votes_by_decoded_bit[bit_index]);
        }
        for (size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++)
                itty_feed_model_projected_repair_candidate_clear (&candidates[candidate_index]);
        if (stats) {
                size_t residual_flips = stats->residual_enable_flips + stats->residual_mask_flips;
                size_t normal_flips = stats->layer_flips > residual_flips ?
                                      stats->layer_flips - residual_flips :
                                      0;

                stats->direct_quota_vote_flips = stats->requested_condensed_bits < normal_flips ?
                                                 stats->requested_condensed_bits :
                                                 normal_flips;
                stats->majority_threshold_support_flips = normal_flips - stats->direct_quota_vote_flips;
                stats->conflict_resolution_flips = 0;
                stats->collateral_preservation_flips = 0;
                stats->target_zero_safety_preservation_flips = 0;
                stats->selection_preservation_flips = 0;
        }
        free (selected_candidates);
        free (candidates);
        itty_feed_model_bad_flip_frequency_list_clear (&bad_flip_frequencies);
        free (selected_safe_flips_by_decoded_bit);
        free (selected_safe_votes_by_decoded_bit);
        free (selected_safe_final_votes_by_decoded_bit);
        free (flips_by_decoded_bit);
        free (before_false_negative_bits);
        free (blocks_by_final_node);
        free (replay_quota_by_decoded_bit);
        free (replay_unsafe_votes_by_decoded_bit);
        free (replay_safe_votes_by_decoded_bit);
        free (replay_safe_final_votes_by_decoded_bit);
        free (replay_unsafe_final_votes_by_decoded_bit);
        free (structural_output_cares);
        free (structural_output_targets);
        free (output_cares);
        free (output_targets);
        free (residual_cares);
        free (condensed_cares);
        free (condensed_targets);
        itty_feed_model_final_repair_list_clear (&repairs);
        itty_bit_string_list_free (layer_outputs);
        if (layer_input != input)
                itty_bit_string_list_free (layer_input);

        return projected_all;
}

static void
itty_feed_model_accumulate_projected_repair_stats (itty_feed_model_projected_repair_stats_t       *stats,
                                                   itty_feed_model_projected_repair_stats_t const *batch_stats)
{
        size_t before_distance = stats->before_distance;
        size_t before_blockers = stats->before_blockers;

        stats->projected_blocks += batch_stats->projected_blocks;
        stats->accepted_blocks += batch_stats->accepted_blocks;
        stats->fully_realized_blocks += batch_stats->fully_realized_blocks;
        stats->partially_realized_blocks += batch_stats->partially_realized_blocks;
        stats->unrealized_blocks += batch_stats->unrealized_blocks;
        stats->condensed_realistic_blocks += batch_stats->condensed_realistic_blocks;
        stats->condensed_realistic_strict_distance_helpful_blocks += batch_stats->condensed_realistic_strict_distance_helpful_blocks;
        stats->condensed_realistic_blocker_helpful_blocks += batch_stats->condensed_realistic_blocker_helpful_blocks;
        stats->condensed_realistic_objective_helpful_blocks += batch_stats->condensed_realistic_objective_helpful_blocks;
        stats->condensed_realistic_harmful_blocks += batch_stats->condensed_realistic_harmful_blocks;
        stats->condensed_realistic_neutral_blocks += batch_stats->condensed_realistic_neutral_blocks;
        stats->accepted_strict_distance_blocks += batch_stats->accepted_strict_distance_blocks;
        stats->accepted_blocker_blocks += batch_stats->accepted_blocker_blocks;
        stats->strict_distance_layer_flips += batch_stats->strict_distance_layer_flips;
        stats->blocker_layer_flips += batch_stats->blocker_layer_flips;
        stats->residual_enable_flips += batch_stats->residual_enable_flips;
        stats->residual_mask_flips += batch_stats->residual_mask_flips;
        stats->residual_active_bits += batch_stats->residual_active_bits;
        stats->residual_candidate_blocks += batch_stats->residual_candidate_blocks;
        stats->residual_accepted_blocks += batch_stats->residual_accepted_blocks;
        stats->previous_layer_projected_blocks += batch_stats->previous_layer_projected_blocks;
        stats->previous_layer_strict_distance_helpful_blocks += batch_stats->previous_layer_strict_distance_helpful_blocks;
        stats->previous_layer_blocker_helpful_blocks += batch_stats->previous_layer_blocker_helpful_blocks;
        stats->previous_layer_objective_helpful_blocks += batch_stats->previous_layer_objective_helpful_blocks;
        stats->previous_layer_harmful_blocks += batch_stats->previous_layer_harmful_blocks;
        stats->previous_layer_neutral_blocks += batch_stats->previous_layer_neutral_blocks;
        stats->previous_layer_harmful_distance_blocks += batch_stats->previous_layer_harmful_distance_blocks;
        stats->previous_layer_harmful_blocker_blocks += batch_stats->previous_layer_harmful_blocker_blocks;
        stats->previous_layer_harmful_margin_blocks += batch_stats->previous_layer_harmful_margin_blocks;
        stats->previous_layer_harmful_safety_blocks += batch_stats->previous_layer_harmful_safety_blocks;
        stats->previous_layer_harm_correct_zero_to_false_positive_bits += batch_stats->previous_layer_harm_correct_zero_to_false_positive_bits;
        stats->previous_layer_harm_correct_one_to_false_negative_bits += batch_stats->previous_layer_harm_correct_one_to_false_negative_bits;
        stats->previous_layer_harm_false_positive_to_correct_zero_bits += batch_stats->previous_layer_harm_false_positive_to_correct_zero_bits;
        stats->previous_layer_harm_false_negative_to_correct_one_bits += batch_stats->previous_layer_harm_false_negative_to_correct_one_bits;
        stats->previous_layer_harm_false_positive_to_false_negative_bits += batch_stats->previous_layer_harm_false_positive_to_false_negative_bits;
        stats->previous_layer_harm_false_negative_to_false_positive_bits += batch_stats->previous_layer_harm_false_negative_to_false_positive_bits;
        stats->previous_layer_harm_unchanged_wrong_bits += batch_stats->previous_layer_harm_unchanged_wrong_bits;
        stats->previous_layer_harm_unchanged_correct_bits += batch_stats->previous_layer_harm_unchanged_correct_bits;
        stats->previous_layer_pinned_projected_blocks += batch_stats->previous_layer_pinned_projected_blocks;
        stats->previous_layer_pinned_strict_distance_helpful_blocks += batch_stats->previous_layer_pinned_strict_distance_helpful_blocks;
        stats->previous_layer_pinned_blocker_helpful_blocks += batch_stats->previous_layer_pinned_blocker_helpful_blocks;
        stats->previous_layer_pinned_objective_helpful_blocks += batch_stats->previous_layer_pinned_objective_helpful_blocks;
        stats->previous_layer_pinned_harmful_blocks += batch_stats->previous_layer_pinned_harmful_blocks;
        stats->previous_layer_pinned_neutral_blocks += batch_stats->previous_layer_pinned_neutral_blocks;
        stats->conflicts += batch_stats->conflicts;
        stats->estimated_layer_flips += batch_stats->estimated_layer_flips;
        stats->direct_quota_vote_flips += batch_stats->direct_quota_vote_flips;
        stats->majority_threshold_support_flips += batch_stats->majority_threshold_support_flips;
        stats->conflict_resolution_flips += batch_stats->conflict_resolution_flips;
        stats->collateral_preservation_flips += batch_stats->collateral_preservation_flips;
        stats->target_zero_safety_preservation_flips += batch_stats->target_zero_safety_preservation_flips;
        stats->selection_preservation_flips += batch_stats->selection_preservation_flips;
        for (size_t bucket = 0; bucket < ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS; bucket++)
                stats->quota_vote_support_cost_histogram[bucket] += batch_stats->quota_vote_support_cost_histogram[bucket];
        stats->decoded_bits_fixed += batch_stats->decoded_bits_fixed;
        stats->fixed_decoded_bit_flips_total += batch_stats->fixed_decoded_bit_flips_total;
        if (batch_stats->decoded_bits_fixed > 0 &&
            (stats->decoded_bits_fixed == batch_stats->decoded_bits_fixed ||
             batch_stats->fixed_decoded_bit_flips_min < stats->fixed_decoded_bit_flips_min))
                stats->fixed_decoded_bit_flips_min = batch_stats->fixed_decoded_bit_flips_min;
        if (batch_stats->fixed_decoded_bit_flips_max > stats->fixed_decoded_bit_flips_max)
                stats->fixed_decoded_bit_flips_max = batch_stats->fixed_decoded_bit_flips_max;
        stats->quota_size_total += batch_stats->quota_size_total;
        if (batch_stats->quota_size_max > stats->quota_size_max)
                stats->quota_size_max = batch_stats->quota_size_max;
        stats->requested_condensed_bits += batch_stats->requested_condensed_bits;
        stats->realized_condensed_bits += batch_stats->realized_condensed_bits;
        stats->already_satisfied_condensed_bits += batch_stats->already_satisfied_condensed_bits;
        stats->condensed_bits_needing_flips += batch_stats->condensed_bits_needing_flips;
        stats->available_flippable_votes += batch_stats->available_flippable_votes;
        stats->requested_output_bits += batch_stats->requested_output_bits;
        stats->realized_output_bits += batch_stats->realized_output_bits;
        stats->lost_output_bits += batch_stats->lost_output_bits;
        stats->extra_output_bits += batch_stats->extra_output_bits;
        stats->structural_extra_output_bits += batch_stats->structural_extra_output_bits;
        stats->collateral_extra_output_bits += batch_stats->collateral_extra_output_bits;
        stats->no_effect_candidates += batch_stats->no_effect_candidates;
        stats->no_effect_candidate_already_satisfied += batch_stats->no_effect_candidate_already_satisfied;
        stats->no_effect_candidate_no_majority_crossing += batch_stats->no_effect_candidate_no_majority_crossing;
        stats->no_effect_candidate_unselected_node += batch_stats->no_effect_candidate_unselected_node;
        stats->no_effect_candidate_irrelevant_segment += batch_stats->no_effect_candidate_irrelevant_segment;
        stats->no_effect_candidate_vote_tied += batch_stats->no_effect_candidate_vote_tied;
        stats->replay_safe_candidates += batch_stats->replay_safe_candidates;
        stats->replay_safe_strict_distance_candidates += batch_stats->replay_safe_strict_distance_candidates;
        stats->replay_safe_deficit_candidates += batch_stats->replay_safe_deficit_candidates;
        stats->replay_safe_frontier_candidates += batch_stats->replay_safe_frontier_candidates;
        stats->replay_safe_vote_movement_candidates += batch_stats->replay_safe_vote_movement_candidates;
        stats->replay_safe_noop_candidates += batch_stats->replay_safe_noop_candidates;
        stats->replay_safe_irrelevant_candidates += batch_stats->replay_safe_irrelevant_candidates;
        stats->current_candidate_net_positive += batch_stats->current_candidate_net_positive;
        stats->current_candidate_net_zero += batch_stats->current_candidate_net_zero;
        stats->current_candidate_net_negative += batch_stats->current_candidate_net_negative;
        stats->replay_safe_quota_feasible_decoded_bits += batch_stats->replay_safe_quota_feasible_decoded_bits;
        stats->replay_safe_quota_selected_decoded_bits += batch_stats->replay_safe_quota_selected_decoded_bits;
        stats->replay_safe_quota_accepted_decoded_bits += batch_stats->replay_safe_quota_accepted_decoded_bits;
        stats->replay_safe_quota_completed_decoded_bits += batch_stats->replay_safe_quota_completed_decoded_bits;
        stats->replay_safe_quota_local_realized_decoded_bits += batch_stats->replay_safe_quota_local_realized_decoded_bits;
        stats->replay_safe_quota_local_crossed_decoded_bits += batch_stats->replay_safe_quota_local_crossed_decoded_bits;
        stats->replay_safe_quota_net_positive_decoded_bits += batch_stats->replay_safe_quota_net_positive_decoded_bits;
        stats->replay_safe_quota_cancelled_decoded_bits += batch_stats->replay_safe_quota_cancelled_decoded_bits;
        stats->replay_safe_quota_final_vote_reached += batch_stats->replay_safe_quota_final_vote_reached;
        stats->replay_safe_quota_final_vote_changed += batch_stats->replay_safe_quota_final_vote_changed;
        stats->replay_safe_quota_final_vote_crossed += batch_stats->replay_safe_quota_final_vote_crossed;
        stats->replay_safe_quota_decoded_vote_changed += batch_stats->replay_safe_quota_decoded_vote_changed;
        stats->replay_safe_quota_decoded_threshold_crossed += batch_stats->replay_safe_quota_decoded_threshold_crossed;
        stats->replay_safe_quota_lost_at_final_majority += batch_stats->replay_safe_quota_lost_at_final_majority;
        stats->replay_safe_quota_lost_at_final_rotation_or_expansion += batch_stats->replay_safe_quota_lost_at_final_rotation_or_expansion;
        stats->replay_safe_quota_lost_at_unselected_final_node += batch_stats->replay_safe_quota_lost_at_unselected_final_node;
        stats->replay_safe_quota_lost_at_segment_vote_margin += batch_stats->replay_safe_quota_lost_at_segment_vote_margin;
        stats->replay_safe_quota_lost_due_to_duplicate_final_segment += batch_stats->replay_safe_quota_lost_due_to_duplicate_final_segment;
        stats->replay_safe_quota_local_votes_selected += batch_stats->replay_safe_quota_local_votes_selected;
        stats->replay_safe_quota_unique_local_votes_changed += batch_stats->replay_safe_quota_unique_local_votes_changed;
        stats->replay_safe_quota_unique_final_condensed_bits_affected += batch_stats->replay_safe_quota_unique_final_condensed_bits_affected;
        stats->replay_safe_quota_unique_final_segment_votes_affected += batch_stats->replay_safe_quota_unique_final_segment_votes_affected;
        stats->replay_safe_quota_unique_decoded_bits_affected += batch_stats->replay_safe_quota_unique_decoded_bits_affected;
        stats->replay_safe_quota_distance_flip_decoded_bits += batch_stats->replay_safe_quota_distance_flip_decoded_bits;
        stats->replay_safe_quota_rejected_local_only_decoded_bits += batch_stats->replay_safe_quota_rejected_local_only_decoded_bits;
        stats->replay_safe_quota_blocked_decoded_bits += batch_stats->replay_safe_quota_blocked_decoded_bits;
        for (size_t trace_index = 0;
             trace_index < batch_stats->replay_final_surface_feasibility_trace_count &&
             stats->replay_final_surface_feasibility_trace_count < ITTY_FEED_MODEL_FINAL_SURFACE_FEASIBILITY_TRACE_LIMIT;
             trace_index++)
                stats->replay_final_surface_feasibility_traces[stats->replay_final_surface_feasibility_trace_count++] =
                        batch_stats->replay_final_surface_feasibility_traces[trace_index];
        for (size_t trace_index = 0;
             trace_index < batch_stats->replay_safe_quota_effect_trace_count &&
             stats->replay_safe_quota_effect_trace_count < ITTY_FEED_MODEL_SAFE_QUOTA_EFFECT_TRACE_LIMIT;
             trace_index++)
                stats->replay_safe_quota_effect_traces[stats->replay_safe_quota_effect_trace_count++] =
                        batch_stats->replay_safe_quota_effect_traces[trace_index];
        stats->replay_safe_quota_complete_candidates += batch_stats->replay_safe_quota_complete_candidates;
        stats->replay_safe_quota_incomplete_candidates += batch_stats->replay_safe_quota_incomplete_candidates;
        stats->replay_direct_protected_zero_hit_candidates += batch_stats->replay_direct_protected_zero_hit_candidates;
        stats->replay_reserved_zero_votes += batch_stats->replay_reserved_zero_votes;
        stats->replay_realization_collateral_false_positive_candidates += batch_stats->replay_realization_collateral_false_positive_candidates;
        stats->replay_sensitive_mask_flips += batch_stats->replay_sensitive_mask_flips;
        stats->replay_safe_mask_flips += batch_stats->replay_safe_mask_flips;
        stats->replay_false_positive_mask_flips += batch_stats->replay_false_positive_mask_flips;
        stats->replay_false_negative_mask_flips += batch_stats->replay_false_negative_mask_flips;
        stats->replay_margin_or_safety_weakening_mask_flips += batch_stats->replay_margin_or_safety_weakening_mask_flips;
        stats->replay_collateral_cost += batch_stats->replay_collateral_cost;
        stats->replay_decomposed_candidates += batch_stats->replay_decomposed_candidates;
        stats->replay_decomposed_mask_flips += batch_stats->replay_decomposed_mask_flips;
        stats->replay_decomposed_unsafe_mask_flips += batch_stats->replay_decomposed_unsafe_mask_flips;
        stats->replay_one_bad_flip_candidates += batch_stats->replay_one_bad_flip_candidates;
        stats->replay_mostly_unsafe_candidates += batch_stats->replay_mostly_unsafe_candidates;
        stats->replay_alternate_mask_flips += batch_stats->replay_alternate_mask_flips;
        stats->replay_alternate_unsafe_mask_flips += batch_stats->replay_alternate_unsafe_mask_flips;
        stats->replay_alternate_collateral_cost += batch_stats->replay_alternate_collateral_cost;
        stats->replay_alternate_better_candidates += batch_stats->replay_alternate_better_candidates;
        itty_feed_model_accumulate_bad_flip_top (stats,
                                                 batch_stats);
        stats->replay_unsafe_candidates += batch_stats->replay_unsafe_candidates;
        if (batch_stats->replay_best_candidate_unsafe)
                stats->replay_best_candidate_unsafe = 1;
        stats->replay_unsafe_distance_regressions += batch_stats->replay_unsafe_distance_regressions;
        stats->replay_unsafe_false_positive_excess_regressions += batch_stats->replay_unsafe_false_positive_excess_regressions;
        stats->replay_unsafe_target_one_margin_regressions += batch_stats->replay_unsafe_target_one_margin_regressions;
        stats->replay_unsafe_target_zero_safety_regressions += batch_stats->replay_unsafe_target_zero_safety_regressions;
        stats->replay_unsafe_selected_node_switches += batch_stats->replay_unsafe_selected_node_switches;
        stats->replay_unsafe_best_decoded_node_switches += batch_stats->replay_unsafe_best_decoded_node_switches;
        stats->replay_unsafe_transitions.correct_zero_to_false_positive_bits += batch_stats->replay_unsafe_transitions.correct_zero_to_false_positive_bits;
        stats->replay_unsafe_transitions.correct_one_to_false_negative_bits += batch_stats->replay_unsafe_transitions.correct_one_to_false_negative_bits;
        stats->replay_unsafe_transitions.false_positive_to_correct_zero_bits += batch_stats->replay_unsafe_transitions.false_positive_to_correct_zero_bits;
        stats->replay_unsafe_transitions.false_negative_to_correct_one_bits += batch_stats->replay_unsafe_transitions.false_negative_to_correct_one_bits;
        stats->replay_unsafe_transitions.false_positive_to_false_negative_bits += batch_stats->replay_unsafe_transitions.false_positive_to_false_negative_bits;
        stats->replay_unsafe_transitions.false_negative_to_false_positive_bits += batch_stats->replay_unsafe_transitions.false_negative_to_false_positive_bits;
        stats->replay_unsafe_transitions.unchanged_wrong_bits += batch_stats->replay_unsafe_transitions.unchanged_wrong_bits;
        stats->replay_unsafe_transitions.unchanged_correct_bits += batch_stats->replay_unsafe_transitions.unchanged_correct_bits;
        stats->layer_flips += batch_stats->layer_flips;
        stats->before_distance = before_distance;
        stats->before_blockers = before_blockers;
        stats->after_distance = batch_stats->after_distance;
        stats->after_blockers = batch_stats->after_blockers;
}

static void
itty_feed_model_accumulate_no_effect_candidate_stats (itty_feed_model_projected_repair_stats_t       *stats,
                                                     itty_feed_model_projected_repair_stats_t const *batch_stats)
{
        stats->no_effect_candidates += batch_stats->no_effect_candidates;
        stats->no_effect_candidate_already_satisfied += batch_stats->no_effect_candidate_already_satisfied;
        stats->no_effect_candidate_no_majority_crossing += batch_stats->no_effect_candidate_no_majority_crossing;
        stats->no_effect_candidate_unselected_node += batch_stats->no_effect_candidate_unselected_node;
        stats->no_effect_candidate_irrelevant_segment += batch_stats->no_effect_candidate_irrelevant_segment;
        stats->no_effect_candidate_vote_tied += batch_stats->no_effect_candidate_vote_tied;
        stats->replay_safe_candidates += batch_stats->replay_safe_candidates;
        stats->replay_safe_strict_distance_candidates += batch_stats->replay_safe_strict_distance_candidates;
        stats->replay_safe_deficit_candidates += batch_stats->replay_safe_deficit_candidates;
        stats->replay_safe_frontier_candidates += batch_stats->replay_safe_frontier_candidates;
        stats->replay_safe_vote_movement_candidates += batch_stats->replay_safe_vote_movement_candidates;
        stats->replay_safe_noop_candidates += batch_stats->replay_safe_noop_candidates;
        stats->replay_safe_irrelevant_candidates += batch_stats->replay_safe_irrelevant_candidates;
        stats->replay_safe_quota_feasible_decoded_bits += batch_stats->replay_safe_quota_feasible_decoded_bits;
        stats->replay_safe_quota_selected_decoded_bits += batch_stats->replay_safe_quota_selected_decoded_bits;
        stats->replay_safe_quota_accepted_decoded_bits += batch_stats->replay_safe_quota_accepted_decoded_bits;
        stats->replay_safe_quota_completed_decoded_bits += batch_stats->replay_safe_quota_completed_decoded_bits;
        stats->replay_safe_quota_distance_flip_decoded_bits += batch_stats->replay_safe_quota_distance_flip_decoded_bits;
        stats->replay_safe_quota_rejected_local_only_decoded_bits += batch_stats->replay_safe_quota_rejected_local_only_decoded_bits;
        stats->replay_safe_quota_blocked_decoded_bits += batch_stats->replay_safe_quota_blocked_decoded_bits;
        for (size_t trace_index = 0;
             trace_index < batch_stats->replay_final_surface_feasibility_trace_count &&
             stats->replay_final_surface_feasibility_trace_count < ITTY_FEED_MODEL_FINAL_SURFACE_FEASIBILITY_TRACE_LIMIT;
             trace_index++)
                stats->replay_final_surface_feasibility_traces[stats->replay_final_surface_feasibility_trace_count++] =
                        batch_stats->replay_final_surface_feasibility_traces[trace_index];
        for (size_t trace_index = 0;
             trace_index < batch_stats->replay_safe_quota_effect_trace_count &&
             stats->replay_safe_quota_effect_trace_count < ITTY_FEED_MODEL_SAFE_QUOTA_EFFECT_TRACE_LIMIT;
             trace_index++)
                stats->replay_safe_quota_effect_traces[stats->replay_safe_quota_effect_trace_count++] =
                        batch_stats->replay_safe_quota_effect_traces[trace_index];
        stats->replay_safe_quota_complete_candidates += batch_stats->replay_safe_quota_complete_candidates;
        stats->replay_safe_quota_incomplete_candidates += batch_stats->replay_safe_quota_incomplete_candidates;
        stats->replay_direct_protected_zero_hit_candidates += batch_stats->replay_direct_protected_zero_hit_candidates;
        stats->replay_reserved_zero_votes += batch_stats->replay_reserved_zero_votes;
        stats->replay_realization_collateral_false_positive_candidates += batch_stats->replay_realization_collateral_false_positive_candidates;
        stats->replay_sensitive_mask_flips += batch_stats->replay_sensitive_mask_flips;
        stats->replay_safe_mask_flips += batch_stats->replay_safe_mask_flips;
        stats->replay_false_positive_mask_flips += batch_stats->replay_false_positive_mask_flips;
        stats->replay_false_negative_mask_flips += batch_stats->replay_false_negative_mask_flips;
        stats->replay_margin_or_safety_weakening_mask_flips += batch_stats->replay_margin_or_safety_weakening_mask_flips;
        stats->replay_collateral_cost += batch_stats->replay_collateral_cost;
        stats->replay_decomposed_candidates += batch_stats->replay_decomposed_candidates;
        stats->replay_decomposed_mask_flips += batch_stats->replay_decomposed_mask_flips;
        stats->replay_decomposed_unsafe_mask_flips += batch_stats->replay_decomposed_unsafe_mask_flips;
        stats->replay_one_bad_flip_candidates += batch_stats->replay_one_bad_flip_candidates;
        stats->replay_mostly_unsafe_candidates += batch_stats->replay_mostly_unsafe_candidates;
        stats->replay_alternate_mask_flips += batch_stats->replay_alternate_mask_flips;
        stats->replay_alternate_unsafe_mask_flips += batch_stats->replay_alternate_unsafe_mask_flips;
        stats->replay_alternate_collateral_cost += batch_stats->replay_alternate_collateral_cost;
        stats->replay_alternate_better_candidates += batch_stats->replay_alternate_better_candidates;
        itty_feed_model_accumulate_bad_flip_top (stats,
                                                 batch_stats);
        stats->replay_unsafe_candidates += batch_stats->replay_unsafe_candidates;
        if (batch_stats->replay_best_candidate_unsafe)
                stats->replay_best_candidate_unsafe = 1;
        stats->replay_unsafe_distance_regressions += batch_stats->replay_unsafe_distance_regressions;
        stats->replay_unsafe_false_positive_excess_regressions += batch_stats->replay_unsafe_false_positive_excess_regressions;
        stats->replay_unsafe_target_one_margin_regressions += batch_stats->replay_unsafe_target_one_margin_regressions;
        stats->replay_unsafe_target_zero_safety_regressions += batch_stats->replay_unsafe_target_zero_safety_regressions;
        stats->replay_unsafe_selected_node_switches += batch_stats->replay_unsafe_selected_node_switches;
        stats->replay_unsafe_best_decoded_node_switches += batch_stats->replay_unsafe_best_decoded_node_switches;
        stats->replay_unsafe_transitions.correct_zero_to_false_positive_bits += batch_stats->replay_unsafe_transitions.correct_zero_to_false_positive_bits;
        stats->replay_unsafe_transitions.correct_one_to_false_negative_bits += batch_stats->replay_unsafe_transitions.correct_one_to_false_negative_bits;
        stats->replay_unsafe_transitions.false_positive_to_correct_zero_bits += batch_stats->replay_unsafe_transitions.false_positive_to_correct_zero_bits;
        stats->replay_unsafe_transitions.false_negative_to_correct_one_bits += batch_stats->replay_unsafe_transitions.false_negative_to_correct_one_bits;
        stats->replay_unsafe_transitions.false_positive_to_false_negative_bits += batch_stats->replay_unsafe_transitions.false_positive_to_false_negative_bits;
        stats->replay_unsafe_transitions.false_negative_to_false_positive_bits += batch_stats->replay_unsafe_transitions.false_negative_to_false_positive_bits;
        stats->replay_unsafe_transitions.unchanged_wrong_bits += batch_stats->replay_unsafe_transitions.unchanged_wrong_bits;
        stats->replay_unsafe_transitions.unchanged_correct_bits += batch_stats->replay_unsafe_transitions.unchanged_correct_bits;
}

struct itty_feed_model_layer_state_snapshot_t {
        itty_bit_string_list_t **masks;
        itty_bit_string_t      **residual_enables;
        itty_bit_string_t      **residual_masks;
};

static itty_feed_model_layer_state_snapshot_t *
itty_feed_model_snapshot_layer_state (itty_feed_model_t *model,
                                      size_t             layer_index)
{
        itty_feed_model_layer_state_snapshot_t *snapshot = calloc (1, sizeof (itty_feed_model_layer_state_snapshot_t));
        snapshot->masks = calloc (model->nodes_per_layer, sizeof (itty_bit_string_list_t *));
        snapshot->residual_enables = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));
        snapshot->residual_masks = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                snapshot->masks[node_index] = itty_feed_model_bit_string_list_clone (model->masks_by_layer_node[layer_index][node_index]);
                snapshot->residual_enables[node_index] = itty_feed_model_bit_string_clone (model->residual_enable_by_layer_node[layer_index][node_index]);
                snapshot->residual_masks[node_index] = itty_feed_model_bit_string_clone (model->residual_mask_by_layer_node[layer_index][node_index]);
        }

        return snapshot;
}

static void
itty_feed_model_free_layer_state_snapshot (itty_feed_model_t                      *model,
                                           itty_feed_model_layer_state_snapshot_t *snapshot)
{
        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_list_free (snapshot->masks[node_index]);
                itty_bit_string_free (snapshot->residual_enables[node_index]);
                itty_bit_string_free (snapshot->residual_masks[node_index]);
        }
        free (snapshot->masks);
        free (snapshot->residual_enables);
        free (snapshot->residual_masks);
        free (snapshot);
}

static void
itty_feed_model_restore_layer_state_snapshot (itty_feed_model_t                      *model,
                                              size_t                                  layer_index,
                                              itty_feed_model_layer_state_snapshot_t *snapshot)
{
        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_list_free (model->masks_by_layer_node[layer_index][node_index]);
                itty_bit_string_free (model->residual_enable_by_layer_node[layer_index][node_index]);
                itty_bit_string_free (model->residual_mask_by_layer_node[layer_index][node_index]);
                model->masks_by_layer_node[layer_index][node_index] = snapshot->masks[node_index];
                model->residual_enable_by_layer_node[layer_index][node_index] = snapshot->residual_enables[node_index];
                model->residual_mask_by_layer_node[layer_index][node_index] = snapshot->residual_masks[node_index];
        }
        free (snapshot->masks);
        free (snapshot->residual_enables);
        free (snapshot->residual_masks);
        free (snapshot);
}

static void
itty_feed_model_set_mask_trace_value (itty_feed_model_t const                 *model,
                                      itty_bit_string_list_t                  *masks,
                                      itty_feed_model_mask_flip_trace_t const *trace)
{
        itty_bit_string_t *mask;
        size_t *words;
        size_t word_index;
        size_t bit_offset;
        size_t bit_capacity;

        mask = itty_bit_string_list_fetch (masks, trace->input_index);
        words = itty_bit_string_get_words (mask);
        word_index = trace->bit_index / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        bit_offset = trace->bit_index % ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        bit_capacity = itty_bit_string_get_length (mask);

        if (!mask || !words || trace->node_index >= model->nodes_per_layer || trace->bit_index >= bit_capacity)
                return;

        if (trace->value_after)
                words[word_index] |= ((size_t) 1) << bit_offset;
        else
                words[word_index] &= ~(((size_t) 1) << bit_offset);
}

static size_t
itty_feed_model_collect_layer_mask_flip_traces (itty_feed_model_t                            *model,
                                                size_t                                        layer_index,
                                                itty_feed_model_layer_state_snapshot_t const *before_snapshot,
                                                itty_feed_model_mask_flip_trace_t            *traces,
                                                size_t                                        trace_limit)
{
        size_t trace_count = 0;

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_list_t *before_masks = before_snapshot->masks[node_index];
                itty_bit_string_list_t *after_masks = model->masks_by_layer_node[layer_index][node_index];

                for (size_t input_index = 0; input_index < model->inputs_per_node; input_index++) {
                        itty_bit_string_t *before_mask = before_masks->bit_strings[input_index];
                        itty_bit_string_t *after_mask = after_masks->bit_strings[input_index];
                        size_t bit_capacity = itty_bit_string_get_length (after_mask);

                        for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                                bool before_value = itty_bit_string_get_bit (before_mask, bit_index);
                                bool after_value = itty_bit_string_get_bit (after_mask, bit_index);

                                if (before_value == after_value)
                                        continue;
                                if (trace_count < trace_limit)
                                        traces[trace_count] = (itty_feed_model_mask_flip_trace_t) {
                                                .node_index = node_index,
                                                .input_index = input_index,
                                                .bit_index = bit_index,
                                                .value_after = after_value,
                                        };
                                trace_count++;
                        }
                }
        }

        return trace_count;
}

size_t
itty_feed_model_collect_final_layer_mask_flip_traces (itty_feed_model_t                            *model,
                                                      itty_feed_model_layer_state_snapshot_t const *before_snapshot,
                                                      itty_feed_model_mask_flip_trace_t            *traces,
                                                      size_t                                        trace_limit)
{
        if (!model || !before_snapshot || !traces || trace_limit == 0 || model->number_of_layers == 0)
                return 0;

        return itty_feed_model_collect_layer_mask_flip_traces (model,
                                                               model->number_of_layers - 1,
                                                               before_snapshot,
                                                               traces,
                                                               trace_limit);
}

size_t
itty_feed_model_collect_final_layer_toggle_traces_for_nodes (itty_feed_model_t                 *model,
                                                             size_t const                      *node_indices,
                                                             size_t                             node_count,
                                                             itty_feed_model_mask_flip_trace_t *traces,
                                                             size_t                             trace_limit)
{
        size_t trace_count = 0;
        size_t final_layer_index;

        if (!model || !node_indices || !traces || trace_limit == 0 || model->number_of_layers == 0)
                return 0;

        final_layer_index = model->number_of_layers - 1;
        for (size_t node_pos = 0; node_pos < node_count; node_pos++) {
                size_t node_index = node_indices[node_pos];

                if (node_index >= model->nodes_per_layer)
                        continue;

                for (size_t input_index = 0; input_index < model->inputs_per_node; input_index++) {
                        itty_bit_string_t *mask =
                                itty_bit_string_list_fetch (model->masks_by_layer_node[final_layer_index][node_index],
                                                            input_index);
                        size_t *words = mask ? itty_bit_string_get_words (mask) : NULL;
                        size_t bit_capacity = mask ? itty_bit_string_get_length (mask) : 0;

                        if (!mask || !words)
                                continue;

                        for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                                size_t word_index = bit_index / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
                                size_t bit_offset = bit_index % ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
                                bool current_value = (words[word_index] & (((size_t) 1) << bit_offset)) != 0;

                                if (trace_count >= trace_limit)
                                        return trace_count;

                                traces[trace_count++] = (itty_feed_model_mask_flip_trace_t) {
                                        .node_index = node_index,
                                        .input_index = input_index,
                                        .bit_index = bit_index,
                                        .value_after = !current_value,
                                };
                        }
                }
        }

        return trace_count;
}

bool
itty_feed_model_apply_final_layer_mask_flip_traces (itty_feed_model_t                       *model,
                                                    itty_feed_model_mask_flip_trace_t const *traces,
                                                    size_t                                   trace_count)
{
        size_t final_layer_index;

        if (!model || !traces || trace_count == 0 || model->number_of_layers == 0)
                return false;

        final_layer_index = model->number_of_layers - 1;
        for (size_t trace_index = 0; trace_index < trace_count; trace_index++) {
                if (traces[trace_index].node_index >= model->nodes_per_layer ||
                    traces[trace_index].input_index >= model->inputs_per_node)
                        return false;

                itty_feed_model_set_mask_trace_value (model,
                                                      model->masks_by_layer_node[final_layer_index][traces[trace_index].node_index],
                                                      &traces[trace_index]);
        }

        return true;
}

static size_t
itty_feed_model_count_mask_flip_trace_overlap (itty_feed_model_mask_flip_trace_t const *a_traces,
                                               size_t                                   a_count,
                                               itty_feed_model_mask_flip_trace_t const *b_traces,
                                               size_t                                   b_count,
                                               bool                                     require_same_direction)
{
        size_t overlap = 0;

        for (size_t a_index = 0; a_index < a_count; a_index++) {
            for (size_t b_index = 0; b_index < b_count; b_index++) {
                        if (a_traces[a_index].node_index != b_traces[b_index].node_index ||
                            a_traces[a_index].input_index != b_traces[b_index].input_index ||
                            a_traces[a_index].bit_index != b_traces[b_index].bit_index)
                                continue;
                        if (require_same_direction &&
                            a_traces[a_index].value_after != b_traces[b_index].value_after)
                                continue;
                        overlap++;
                        break;
                }
        }

        return overlap;
}

static bool
itty_feed_model_measure_output_popcount_gap (itty_feed_model_t      *model,
                                             itty_bit_string_list_t *input,
                                             size_t                 *selected_node,
                                             size_t                 *selected_popcount,
                                             size_t                 *popcount_gap)
{
        itty_bit_string_list_t *outputs = itty_feed_model_run_to_layer_input (model,
                                                                              input,
                                                                              model->number_of_layers);
        size_t selected_index = 0;

        if (!itty_network_select_output (outputs,
                                         &selected_index)) {
                itty_bit_string_list_free (outputs);
                return false;
        }

        size_t runner_up_popcount = 0;
        *selected_node = selected_index;
        *selected_popcount = itty_bit_string_get_pop_count (itty_bit_string_list_fetch (outputs,
                                                                                        selected_index));

        for (size_t node_index = 0; node_index < outputs->count; node_index++) {
                if (node_index == selected_index)
                        continue;

                size_t popcount = itty_bit_string_get_pop_count (itty_bit_string_list_fetch (outputs,
                                                                                             node_index));
                if (popcount > runner_up_popcount)
                        runner_up_popcount = popcount;
        }

        *popcount_gap = *selected_popcount >= runner_up_popcount ?
                        *selected_popcount - runner_up_popcount :
                        0;
        itty_bit_string_list_free (outputs);
        return true;
}

static size_t
itty_feed_model_count_selected_segment_votes (itty_bit_string_list_t *outputs,
                                              size_t                  selected_node,
                                              size_t                  decoded_bit,
                                              size_t                  target_bit_capacity)
{
        if (!outputs || selected_node >= outputs->count || target_bit_capacity == 0)
                return 0;

        itty_bit_string_t *activation = itty_bit_string_list_fetch (outputs,
                                                                    selected_node);
        size_t output_bit_capacity = itty_bit_string_get_length (activation);
        size_t ones = 0;

        for (size_t output_bit = decoded_bit;
             output_bit < output_bit_capacity;
             output_bit += target_bit_capacity) {
                if (itty_bit_string_get_bit (activation,
                                             output_bit))
                        ones++;
        }

        return ones;
}

bool
itty_feed_model_measure_segment_vote_profile_for_node (itty_feed_model_t                      *model,
                                                       itty_bit_string_list_t                 *input,
                                                       itty_bit_string_t                      *target,
                                                       size_t                                  selected_node,
                                                       itty_feed_model_segment_vote_profile_t *profile)
{
        if (!model ||
            !input ||
            !target ||
            !profile ||
            model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        *profile = (itty_feed_model_segment_vote_profile_t) { 0 };

        itty_bit_string_list_t *outputs = itty_feed_model_run_to_layer_input (model,
                                                                              input,
                                                                              model->number_of_layers);
        if (!outputs || selected_node >= outputs->count) {
                if (outputs)
                        itty_bit_string_list_free (outputs);
                return false;
        }

        itty_bit_string_t *activation = itty_bit_string_list_fetch (outputs, selected_node);
        size_t output_bit_capacity = itty_bit_string_get_length (activation);
        size_t target_bit_capacity = itty_bit_string_get_length (target);
        size_t segments_per_bit = target_bit_capacity == 0 ? 0 : output_bit_capacity / target_bit_capacity;
        size_t threshold = segments_per_bit / 2 + 1;

        profile->node_index = selected_node;
        profile->target_bit_capacity = target_bit_capacity;
        profile->segments_per_bit = segments_per_bit;
        profile->threshold = threshold;

        for (size_t decoded_bit = 0; decoded_bit < target_bit_capacity; decoded_bit++) {
                size_t ones = itty_feed_model_count_selected_segment_votes (outputs,
                                                                            selected_node,
                                                                            decoded_bit,
                                                                            target_bit_capacity);
                size_t bucket = ones < ITTY_FEED_MODEL_SEGMENT_VOTE_PROFILE_LIMIT ?
                                ones :
                                ITTY_FEED_MODEL_SEGMENT_VOTE_PROFILE_LIMIT - 1;
                bool target_bit = itty_bit_string_get_bit (target, decoded_bit);
                bool decoded_one = ones >= threshold;
                size_t deficit = target_bit && !decoded_one ? threshold - ones : 0;
                size_t excess = !target_bit && decoded_one ? ones - (threshold - 1) : 0;
                size_t aligned_levels = target_bit ? ones : (segments_per_bit >= ones ? segments_per_bit - ones : 0);
                size_t opposed_levels = target_bit ? (segments_per_bit >= ones ? segments_per_bit - ones : 0) : ones;
                ptrdiff_t signed_momentum = (ptrdiff_t) aligned_levels - (ptrdiff_t) opposed_levels;
                ptrdiff_t frontier_margin = target_bit ?
                                            (ptrdiff_t) ones - (ptrdiff_t) threshold + 1 :
                                            (ptrdiff_t) (threshold == 0 ? 0 : threshold - 1) - (ptrdiff_t) ones;

                profile->vote_histogram[bucket]++;
                profile->target_aligned_threshold_crossings += aligned_levels;
                profile->target_opposed_threshold_crossings += opposed_levels;
                profile->net_threshold_momentum += signed_momentum;
                profile->frontier_margin_sum += frontier_margin;

                if (excess > 0) {
                        profile->false_positive_bits++;
                        profile->false_positive_vote_histogram[bucket]++;
                        if (excess == 1)
                                profile->near_threshold_false_positive_bits++;
                        else
                                profile->confident_false_positive_bits++;
                }
                if (deficit > 0) {
                        profile->false_negative_bits++;
                        profile->false_negative_vote_histogram[bucket]++;
                        if (deficit == 1)
                                profile->near_threshold_false_negative_bits++;
                        else
                                profile->confident_false_negative_bits++;
                }
                if (deficit > 0 || excess > 0) {
                        profile->wrong_bit_net_threshold_momentum += signed_momentum;
                        profile->wrong_bit_frontier_margin_sum += frontier_margin;
                }
                if ((deficit > 0 || excess > 0) &&
                    profile->trace_count < ITTY_FEED_MODEL_SEGMENT_VOTE_TRACE_LIMIT) {
                        profile->traces[profile->trace_count++] =
                                (itty_feed_model_segment_vote_trace_t) {
                                        .decoded_bit = decoded_bit,
                                        .target_bit = target_bit,
                                        .ones = ones,
                                        .threshold = threshold,
                                        .max_zero = threshold == 0 ? 0 : threshold - 1,
                                        .deficit = deficit,
                                        .excess = excess,
                                        .target_aligned_levels = aligned_levels,
                                        .target_opposed_levels = opposed_levels,
                                        .signed_momentum = signed_momentum,
                                        .frontier_margin = frontier_margin,
                                };
                }
        }

        itty_bit_string_list_free (outputs);
        return true;
}

bool
itty_feed_model_measure_node_condense_vote_histogram (itty_feed_model_t      *model,
                                                      itty_bit_string_list_t *input,
                                                      size_t                  layer_index,
                                                      size_t                  node_index,
                                                      size_t                 *histogram,
                                                      size_t                  histogram_count,
                                                      size_t                 *bit_count)
{
        if (!model ||
            !input ||
            !histogram ||
            histogram_count == 0 ||
            layer_index >= model->number_of_layers ||
            node_index >= model->nodes_per_layer ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node)
                return false;

        memset (histogram, 0, histogram_count * sizeof *histogram);
        if (bit_count)
                *bit_count = 0;

        itty_bit_string_list_t *layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                  input,
                                                                                  layer_index);
        if (!layer_input)
                return false;

        itty_bit_string_list_t *modulated_inputs =
                itty_bit_string_list_exclusive_or (layer_input,
                                                  model->masks_by_layer_node[layer_index][node_index]);
        if (layer_input != input)
                itty_bit_string_list_free (layer_input);
        if (!modulated_inputs)
                return false;

        size_t bit_length = itty_bit_string_list_get_bit_length (modulated_inputs);
        size_t vote_count = modulated_inputs->count;

        for (size_t bit_index = 0; bit_index < bit_length; bit_index++) {
                size_t one_votes = 0;
                size_t bucket = 0;

                for (size_t input_index = 0; input_index < vote_count; input_index++) {
                        if (itty_bit_string_get_bit (modulated_inputs->bit_strings[input_index], bit_index))
                                one_votes++;
                }

                bucket = one_votes < histogram_count ? one_votes : histogram_count - 1;
                histogram[bucket]++;
                if (bit_count)
                        (*bit_count)++;
        }

        itty_bit_string_list_free (modulated_inputs);
        return true;
}

bool
itty_feed_model_measure_node_condensed_output (itty_feed_model_t      *model,
                                               itty_bit_string_list_t *input,
                                               size_t                  layer_index,
                                               size_t                  node_index,
                                               itty_bit_string_t     **condensed_output)
{
        if (!model ||
            !input ||
            !condensed_output ||
            layer_index >= model->number_of_layers ||
            node_index >= model->nodes_per_layer ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node)
                return false;

        *condensed_output = NULL;

        itty_bit_string_list_t *layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                  input,
                                                                                  layer_index);
        if (!layer_input)
                return false;

        itty_bit_string_t *condensed =
                itty_feed_model_run_node_condensed_with_threshold (layer_input,
                                                                   model->masks_by_layer_node[layer_index][node_index],
                                                                   model->within_node_condense_threshold_override);
        if (layer_input != input)
                itty_bit_string_list_free (layer_input);
        if (!condensed)
                return false;

        *condensed_output = condensed;
        return true;
}

bool
itty_feed_model_measure_node_activation_output (itty_feed_model_t      *model,
                                                itty_bit_string_list_t *input,
                                                size_t                  node_index,
                                                itty_bit_string_t     **activation_output)
{
        if (!model ||
            !input ||
            !activation_output ||
            node_index >= model->nodes_per_layer ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node)
                return false;

        *activation_output = NULL;

        itty_bit_string_list_t *outputs = itty_feed_model_run_outputs (model, input);
        if (!outputs)
                return false;

        itty_bit_string_t *activation = itty_bit_string_clone (itty_bit_string_list_fetch (outputs,
                                                                                           node_index));
        itty_bit_string_list_free (outputs);
        if (!activation)
                return false;

        *activation_output = activation;
        return true;
}

bool
itty_feed_model_measure_selected_folded_output (itty_feed_model_t      *model,
                                                itty_bit_string_list_t *input,
                                                itty_bit_string_t      *target,
                                                size_t                 *selected_node,
                                                itty_bit_string_t     **folded_output)
{
        if (!model ||
            !input ||
            !target ||
            !selected_node ||
            !folded_output ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        *folded_output = NULL;
        *selected_node = 0;

        itty_bit_string_list_t *outputs = itty_feed_model_run_to_layer_input (model,
                                                                              input,
                                                                              model->number_of_layers);
        if (!outputs)
                return false;

        itty_feed_model_output_evaluation_t evaluation = { 0 };
        bool ok = itty_feed_model_evaluate_output (model,
                                                   outputs,
                                                   target,
                                                   &evaluation);
        itty_bit_string_list_free (outputs);
        if (!ok)
                return false;

        *selected_node = evaluation.selected_index;
        *folded_output = evaluation.folded_activation;
        return true;
}

bool
itty_feed_model_measure_node_folded_output (itty_feed_model_t      *model,
                                            itty_bit_string_list_t *input,
                                            itty_bit_string_t      *target,
                                            size_t                  node_index,
                                            itty_bit_string_t     **folded_output)
{
        if (!model ||
            !input ||
            !target ||
            !folded_output ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        *folded_output = NULL;

        itty_bit_string_list_t *outputs = itty_feed_model_run_to_layer_input (model,
                                                                              input,
                                                                              model->number_of_layers);
        if (!outputs || node_index >= outputs->count) {
                if (outputs)
                        itty_bit_string_list_free (outputs);
                return false;
        }

        itty_bit_string_t *activation = itty_bit_string_list_fetch (outputs, node_index);
        itty_bit_string_t *folded = NULL;
        bool ok = itty_feed_model_fold_activation_to_target_width (model,
                                                                   activation,
                                                                   target,
                                                                   node_index,
                                                                   &folded);
        itty_bit_string_list_free (outputs);
        if (!ok)
                return false;

        *folded_output = folded == activation ?
                         itty_feed_model_bit_string_clone (folded) :
                         folded;
        return *folded_output != NULL;
}

static size_t
itty_feed_model_count_selected_condensed_votes (itty_feed_model_t      *model,
                                                itty_bit_string_list_t *outputs,
                                                size_t                  layer_index,
                                                size_t                  selected_node,
                                                size_t                  decoded_bit,
                                                size_t                  target_bit_capacity)
{
        if (!model || !outputs || selected_node >= outputs->count || target_bit_capacity == 0)
                return 0;

        itty_bit_string_t *activation = itty_bit_string_list_fetch (outputs,
                                                                    selected_node);
        if (!activation)
                return 0;

        itty_bit_string_t *condensed = itty_feed_model_reduce_desired_output_for_layer (activation,
                                                                                         model->rotations_by_layer[layer_index]);
        if (!condensed)
                return 0;

        size_t output_bit_capacity = itty_bit_string_get_length (activation);
        size_t condensed_bit_capacity = itty_bit_string_get_length (condensed);
        bool *seen = calloc (condensed_bit_capacity == 0 ? 1 : condensed_bit_capacity,
                             sizeof (bool));
        size_t ones = 0;

        if (seen) {
                for (size_t output_bit = decoded_bit;
                     output_bit < output_bit_capacity;
                     output_bit += target_bit_capacity) {
                        size_t source_bit = itty_feed_model_trace_expanded_bit_to_layer (model,
                                                                                         layer_index,
                                                                                         output_bit);
                        if (source_bit >= condensed_bit_capacity || seen[source_bit])
                                continue;
                        seen[source_bit] = true;
                        if (itty_bit_string_get_bit (condensed,
                                                     source_bit))
                                ones++;
                }
                free (seen);
        }

        itty_bit_string_free (condensed);
        return ones;
}

static itty_feed_model_segment_train_stop_reason_t
itty_feed_model_classify_rejected_batch (itty_feed_model_decoder_objective_t const      *before,
                                         itty_feed_model_decoder_objective_t const      *after,
                                         itty_feed_model_projected_repair_stats_t const *batch_stats)
{
        if (after->selected_distance > before->selected_distance)
                return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_DISTANCE_REGRESSION;
        if (after->false_positive_vote_excess > before->false_positive_vote_excess)
                return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_EXCESS_REGRESSION;
        if (after->target_zero_safety_min < before->target_zero_safety_min)
                return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_ZERO_SAFETY_REGRESSION;
        if (after->target_one_margin < before->target_one_margin)
                return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_TARGET_ONE_MARGIN_LOSS;
        if (after->selected_distance == before->selected_distance &&
            after->false_negative_vote_deficit == before->false_negative_vote_deficit &&
            after->false_positive_vote_excess == before->false_positive_vote_excess &&
            after->target_one_margin == before->target_one_margin &&
            after->target_zero_safety_min == before->target_zero_safety_min) {
                if (batch_stats->accepted_blocks == 0 &&
                    batch_stats->replay_safe_quota_blocked_decoded_bits > 0 &&
                    batch_stats->replay_safe_quota_feasible_decoded_bits == 0)
                        return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_CAPACITY_CONFLICT;
                if (batch_stats->accepted_blocks == 0 &&
                    batch_stats->replay_best_candidate_unsafe)
                        return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_SINGLE_CANDIDATE_CONFLICT;
                if (batch_stats->accepted_blocks == 0 &&
                    batch_stats->replay_unsafe_candidates > 0 &&
                    batch_stats->replay_safe_candidates == 0)
                        return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_BLOCKED_ALL_CANDIDATES;
                if (batch_stats->accepted_blocks == 0 &&
                    batch_stats->replay_safe_candidates > 0)
                        return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_SAFE_NO_CURRENT_GAIN;
                if (batch_stats->projected_blocks == 0 &&
                    batch_stats->accepted_blocks == 0)
                        return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_REPAIRS;
                if (batch_stats->accepted_blocks == 0 &&
                    batch_stats->no_effect_candidates > 0)
                        return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_EFFECTIVE_CANDIDATES;
                bool activation_moved = batch_stats->realized_output_bits > 0 ||
                                        batch_stats->extra_output_bits > 0 ||
                                        batch_stats->realized_condensed_bits > 0;
                bool histograms_changed = false;
                size_t before_future_cost = 0;
                size_t after_future_cost = 0;

                for (size_t bucket = 0; bucket < ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS; bucket++) {
                        if (before->false_negative_vote_deficit_histogram[bucket] != after->false_negative_vote_deficit_histogram[bucket] ||
                            before->false_positive_vote_excess_histogram[bucket] != after->false_positive_vote_excess_histogram[bucket] ||
                            before->target_one_margin_histogram[bucket] != after->target_one_margin_histogram[bucket] ||
                            before->target_zero_safety_histogram[bucket] != after->target_zero_safety_histogram[bucket])
                                histograms_changed = true;

                        before_future_cost += bucket * before->false_negative_vote_deficit_histogram[bucket];
                        after_future_cost += bucket * after->false_negative_vote_deficit_histogram[bucket];
                }

                if (!activation_moved)
                        return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_TRUE_NOOP;
                if (after_future_cost < before_future_cost)
                        return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_FUTURE_COST_IMPROVEMENT;
                if (histograms_changed)
                        return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_VOTE_MOVEMENT_TIED;
                return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_LOCAL_ACTIVATION_ONLY;
        }
        if (batch_stats->realized_condensed_bits < batch_stats->requested_condensed_bits ||
            batch_stats->collateral_extra_output_bits > 0)
                return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_REALIZATION_MISMATCH;

        return ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_BATCH_INTERACTION;
}

static void
itty_feed_model_fill_refreshed_round_diagnostics (itty_feed_model_refreshed_projected_repair_round_t *round_stats,
                                                  itty_feed_model_decoder_objective_t const           *before,
                                                  itty_feed_model_decoder_objective_t const           *after,
                                                  itty_feed_model_projected_repair_stats_t const      *batch_stats)
{
        round_stats->selected_node_after = after->selected_node;
        round_stats->best_decoded_node_after = after->best_decoded_node;
        round_stats->best_decoded_distance_after = after->best_decoded_distance;
        round_stats->after_distance = after->selected_distance;
        round_stats->after_blockers = after->false_negative_blocker_bits;
        round_stats->distance_delta = itty_feed_model_positive_delta (before->selected_distance,
                                                                      after->selected_distance);
        round_stats->blocker_delta = itty_feed_model_positive_delta (before->false_negative_blocker_bits,
                                                                     after->false_negative_blocker_bits);
        round_stats->estimated_distance_delta = itty_feed_model_signed_delta (batch_stats->before_distance,
                                                                              batch_stats->after_distance);
        round_stats->actual_distance_delta = itty_feed_model_signed_delta (before->selected_distance,
                                                                           after->selected_distance);
        round_stats->before_false_negative_deficit = before->false_negative_vote_deficit;
        round_stats->after_false_negative_deficit = after->false_negative_vote_deficit;
        round_stats->estimated_false_negative_deficit_delta = 0;
        round_stats->actual_false_negative_deficit_delta = itty_feed_model_signed_delta (before->false_negative_vote_deficit,
                                                                                        after->false_negative_vote_deficit);
        round_stats->before_false_positive_excess = before->false_positive_vote_excess;
        round_stats->after_false_positive_excess = after->false_positive_vote_excess;
        round_stats->estimated_false_positive_excess_delta = 0;
        round_stats->actual_false_positive_excess_delta = itty_feed_model_signed_delta (before->false_positive_vote_excess,
                                                                                       after->false_positive_vote_excess);
        round_stats->before_target_one_margin = before->target_one_margin;
        round_stats->after_target_one_margin = after->target_one_margin;
        round_stats->estimated_target_one_margin_delta = 0;
        round_stats->actual_target_one_margin_delta = itty_feed_model_signed_increase (before->target_one_margin,
                                                                                      after->target_one_margin);
        round_stats->before_target_zero_safety = before->target_zero_safety_min;
        round_stats->after_target_zero_safety = after->target_zero_safety_min;
        round_stats->estimated_target_zero_safety_delta = 0;
        round_stats->actual_target_zero_safety_delta = itty_feed_model_signed_increase (before->target_zero_safety_min,
                                                                                       after->target_zero_safety_min);
        round_stats->selected_activation_changed = batch_stats->realized_output_bits > 0 ||
                                                   batch_stats->extra_output_bits > 0 ||
                                                   batch_stats->realized_condensed_bits > 0;
        for (size_t bucket = 0; bucket < ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS; bucket++) {
                if (before->false_negative_vote_deficit_histogram[bucket] != after->false_negative_vote_deficit_histogram[bucket] ||
                    before->false_positive_vote_excess_histogram[bucket] != after->false_positive_vote_excess_histogram[bucket] ||
                    before->target_one_margin_histogram[bucket] != after->target_one_margin_histogram[bucket] ||
                    before->target_zero_safety_histogram[bucket] != after->target_zero_safety_histogram[bucket])
                        round_stats->segment_votes_changed = true;
        }
        round_stats->candidate_blocks_proposed = batch_stats->projected_blocks;
        round_stats->candidate_blocks_selected = batch_stats->accepted_blocks;
        round_stats->requested_constraints = batch_stats->requested_condensed_bits;
        round_stats->realized_constraints = batch_stats->realized_condensed_bits;
        round_stats->collateral_changed_bits = batch_stats->collateral_extra_output_bits;
        round_stats->accepted_blocks = batch_stats->accepted_blocks;
        round_stats->layer_flips = batch_stats->layer_flips;
}

static void
itty_feed_model_accumulate_replay_transition (itty_feed_model_replay_transition_matrix_t *matrix,
                                              itty_bit_string_t                          *target,
                                              itty_bit_string_t                          *before_folded,
                                              itty_bit_string_t                          *after_folded)
{
        size_t bit_capacity = itty_bit_string_get_number_of_words (target) *
                              ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        for (size_t bit_index = 0; bit_index < bit_capacity; bit_index++) {
                bool target_bit = itty_bit_string_get_bit (target,
                                                           bit_index);
                bool before_bit = itty_bit_string_get_bit (before_folded,
                                                           bit_index);
                bool after_bit = itty_bit_string_get_bit (after_folded,
                                                          bit_index);
                bool before_correct = before_bit == target_bit;
                bool after_correct = after_bit == target_bit;

                if (before_correct && after_correct) {
                        matrix->unchanged_correct_bits++;
                } else if (!before_correct && !after_correct) {
                        if (!target_bit && before_bit && !after_bit)
                                matrix->false_positive_to_false_negative_bits++;
                        else if (target_bit && !before_bit && after_bit)
                                matrix->false_negative_to_false_positive_bits++;
                        else
                                matrix->unchanged_wrong_bits++;
                } else if (before_correct && !after_correct) {
                        if (target_bit)
                                matrix->correct_one_to_false_negative_bits++;
                        else
                                matrix->correct_zero_to_false_positive_bits++;
                } else {
                        if (target_bit)
                                matrix->false_negative_to_correct_one_bits++;
                        else
                                matrix->false_positive_to_correct_zero_bits++;
                }
        }
}

static bool
itty_feed_model_evaluate_replay_example (itty_feed_model_t                    *model,
                                         itty_feed_model_replay_example_t const *example,
                                         itty_feed_model_decoder_objective_t   *objective,
                                         itty_bit_string_t                    **folded_activation)
{
        itty_bit_string_list_t *outputs = itty_feed_model_run_to_layer_input (model,
                                                                              example->input,
                                                                              model->number_of_layers);
        itty_feed_model_output_evaluation_t evaluation = { 0 };

        if (!itty_feed_model_evaluate_output (model,
                                              outputs,
                                              example->target,
                                              &evaluation)) {
                itty_bit_string_list_free (outputs);
                return false;
        }

        if (!itty_feed_model_evaluate_decoder_objective (model,
                                                         example->input,
                                                         example->target,
                                                         objective)) {
                itty_bit_string_free (evaluation.folded_activation);
                itty_bit_string_list_free (outputs);
                return false;
        }

        *folded_activation = evaluation.folded_activation;
        itty_bit_string_list_free (outputs);
        return true;
}

static bool
itty_feed_model_measure_replay_examples (itty_feed_model_t                         *model,
                                         itty_feed_model_replay_example_t const    *examples,
                                         size_t                                     example_count,
                                         itty_feed_model_decoder_objective_t       *objectives,
                                         itty_bit_string_t                        **folded_activations)
{
        for (size_t example_index = 0; example_index < example_count; example_index++) {
                if (!itty_feed_model_evaluate_replay_example (model,
                                                              &examples[example_index],
                                                              &objectives[example_index],
                                                              &folded_activations[example_index]))
                        return false;
        }

        return true;
}

static bool
itty_feed_model_score_replay_after_batch (itty_feed_model_t                                      *model,
                                          itty_feed_model_replay_example_t const                 *examples,
                                          size_t                                                  example_count,
                                          bool                                                    strict_replay_guard,
                                          itty_feed_model_decoder_objective_t const              *before_objectives,
                                          itty_bit_string_t                                     **before_folded_activations,
                                          itty_feed_model_refreshed_projected_repair_round_t     *round_stats)
{
        bool replay_safe = true;

        for (size_t example_index = 0; example_index < example_count; example_index++) {
                itty_feed_model_decoder_objective_t after_objective = { 0 };
                itty_bit_string_t *after_folded = NULL;

                if (!itty_feed_model_evaluate_replay_example (model,
                                                              &examples[example_index],
                                                              &after_objective,
                                                              &after_folded))
                        return false;

                itty_bit_string_t *before_difference = itty_bit_string_exclusive_or (before_folded_activations[example_index],
                                                                                     examples[example_index].target);
                itty_bit_string_t *after_difference = itty_bit_string_exclusive_or (after_folded,
                                                                                    examples[example_index].target);
                size_t before_folded_distance = itty_bit_string_get_pop_count (before_difference);
                size_t after_folded_distance = itty_bit_string_get_pop_count (after_difference);
                itty_bit_string_free (before_difference);
                itty_bit_string_free (after_difference);

                if (before_folded_distance == 0)
                        round_stats->replay_solved_before++;
                if (strict_replay_guard &&
                    before_folded_distance == 0 &&
                    after_folded_distance > 0)
                        replay_safe = false;
                if (before_folded_distance == 0 &&
                    after_folded_distance > 0)
                        round_stats->replay_unsolved_after++;

                itty_feed_model_accumulate_replay_transition (&round_stats->replay_transitions,
                                                              examples[example_index].target,
                                                              before_folded_activations[example_index],
                                                              after_folded);
                round_stats->replay_distance_delta += itty_feed_model_signed_delta (before_objectives[example_index].selected_distance,
                                                                                    after_objective.selected_distance);
                round_stats->replay_false_negative_deficit_delta += itty_feed_model_signed_delta (before_objectives[example_index].false_negative_vote_deficit,
                                                                                                  after_objective.false_negative_vote_deficit);
                round_stats->replay_false_positive_excess_delta += itty_feed_model_signed_delta (before_objectives[example_index].false_positive_vote_excess,
                                                                                                 after_objective.false_positive_vote_excess);
                round_stats->replay_target_one_margin_delta += itty_feed_model_signed_increase (before_objectives[example_index].target_one_margin,
                                                                                                after_objective.target_one_margin);
                round_stats->replay_target_zero_safety_delta += itty_feed_model_signed_increase (before_objectives[example_index].target_zero_safety_min,
                                                                                                 after_objective.target_zero_safety_min);
                if (before_objectives[example_index].selected_node != after_objective.selected_node)
                        round_stats->replay_selected_node_switches++;
                if (before_objectives[example_index].best_decoded_node != after_objective.best_decoded_node)
                        round_stats->replay_best_decoded_node_switches++;
                round_stats->replay_examples_scored++;

                itty_bit_string_free (after_folded);
        }

        return replay_safe;
}

bool
itty_feed_model_train_penultimate_layer_with_refreshed_final_repairs (itty_feed_model_t                                          *model,
                                                                      itty_bit_string_list_t                                     *input,
                                                                      itty_bit_string_t                                          *target,
                                                                      itty_feed_model_refreshed_projected_repair_options_t const *options,
                                                                      itty_feed_model_refreshed_projected_repair_stats_t         *stats)
{
        if (stats)
                *stats = (itty_feed_model_refreshed_projected_repair_stats_t) { 0 };

        if (!model ||
            !input ||
            !target ||
            model->number_of_layers < 2 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        size_t penultimate_layer = model->number_of_layers - 2;
        size_t batch_size = options ? options->batch_size : 0;
        size_t max_rounds = options ? options->max_rounds : 0;
        size_t max_layer_flips_per_batch = options ? options->max_layer_flips_per_batch : 0;
        size_t max_blocks_per_final_node = options ? options->max_blocks_per_final_node : 0;
        bool use_or_residual_repairs = options ? options->use_or_residual_repairs : false;
        bool prefer_blocker_efficiency = options ? options->prefer_blocker_efficiency : false;
        bool require_residual_blocker_efficiency = options ? options->require_residual_blocker_efficiency : false;
        size_t top_k_segment_vote_alternatives = options ? options->top_k_segment_vote_alternatives : 0;
        size_t replay_zero_protection_penalty = options ? options->replay_zero_protection_penalty : 0;
        size_t replay_one_protection_penalty = options ? options->replay_one_protection_penalty : 0;
        size_t replay_taboo_flip_penalty = options ? options->replay_taboo_flip_penalty : 0;
        bool replay_safe_quota_complete_only = options ? options->replay_safe_quota_complete_only : false;
        bool reserve_replay_protected_zero_votes = options ? options->reserve_replay_protected_zero_votes : false;
        itty_feed_model_replay_example_t const *replay_examples = options ? options->replay_examples : NULL;
        size_t replay_example_count = options ? options->replay_example_count : 0;
        bool strict_replay_guard = options ? options->strict_replay_guard : false;
        bool strict_replay_taboo_rejection = options ? options->strict_replay_taboo_rejection : false;
        itty_feed_model_refreshed_projected_repair_round_t *trajectory = options ? options->trajectory : NULL;
        size_t trajectory_count = options ? options->trajectory_count : 0;
        itty_feed_model_decoder_objective_t current_objective = { 0 };
        itty_feed_model_decoder_objective_t *replay_baseline_objectives = NULL;
        itty_bit_string_t **replay_baseline_folded = NULL;

        if (!itty_feed_model_measure_decoder_objective (model,
                                                        input,
                                                        target,
                                                        &current_objective))
                return false;

        if (replay_example_count > 0) {
                replay_baseline_objectives = calloc (replay_example_count,
                                                     sizeof (itty_feed_model_decoder_objective_t));
                replay_baseline_folded = calloc (replay_example_count,
                                                 sizeof (itty_bit_string_t *));
                if (!itty_feed_model_measure_replay_examples (model,
                                                              replay_examples,
                                                              replay_example_count,
                                                              replay_baseline_objectives,
                                                              replay_baseline_folded)) {
                        for (size_t replay_index = 0; replay_index < replay_example_count; replay_index++)
                                if (replay_baseline_folded[replay_index])
                                        itty_bit_string_free (replay_baseline_folded[replay_index]);
                        free (replay_baseline_folded);
                        free (replay_baseline_objectives);
                        return false;
                }
        }

        if (stats) {
                stats->before_distance = current_objective.selected_distance;
                stats->after_distance = current_objective.selected_distance;
                stats->before_blockers = current_objective.false_negative_blocker_bits;
                stats->after_blockers = current_objective.false_negative_blocker_bits;
                stats->projected.before_distance = current_objective.selected_distance;
                stats->projected.before_blockers = current_objective.false_negative_blocker_bits;
        }

        for (size_t round = 0; max_rounds == 0 || round < max_rounds; round++) {
                if (current_objective.selected_distance == 0)
                        break;

                itty_feed_model_replay_example_t current_example = {
                        .input = input,
                        .target = target,
                };
                itty_bit_string_t *current_before_folded = NULL;
                itty_feed_model_decoder_objective_t current_before_objective = { 0 };
                bool have_current_before_folded =
                        itty_feed_model_evaluate_replay_example (model,
                                                                 &current_example,
                                                                 &current_before_objective,
                                                                 &current_before_folded);

                itty_feed_model_refreshed_projected_repair_round_t round_stats = {
                        .round_index = round,
                        .selected_node = current_objective.selected_node,
                        .selected_popcount = current_objective.selected_popcount,
                        .best_decoded_node = current_objective.best_decoded_node,
                        .best_decoded_distance = current_objective.best_decoded_distance,
                        .selected_is_best_decoded = current_objective.selected_node == current_objective.best_decoded_node,
                        .before_distance = current_objective.selected_distance,
                        .before_blockers = current_objective.false_negative_blocker_bits,
                        .before_false_negative_deficit = current_objective.false_negative_vote_deficit,
                        .before_false_positive_excess = current_objective.false_positive_vote_excess,
                        .before_target_one_margin = current_objective.target_one_margin,
                        .before_target_zero_safety = current_objective.target_zero_safety_min
                };
                itty_feed_model_measure_output_popcount_gap (model,
                                                             input,
                                                             &round_stats.selected_node,
                                                             &round_stats.selected_popcount,
                                                             &round_stats.popcount_gap);

                size_t effective_batch_size = batch_size;
                bool round_finished = false;
                while (!round_finished) {
                        itty_feed_model_layer_state_snapshot_t *snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                                 penultimate_layer);
                        itty_feed_model_projected_repair_options_t batch_options = {
                                .max_projected_blocks = effective_batch_size,
                                .max_layer_flips = max_layer_flips_per_batch,
                                .max_blocks_per_final_node = max_blocks_per_final_node,
                                .use_or_residual_repairs = use_or_residual_repairs,
                                .prefer_blocker_efficiency = prefer_blocker_efficiency,
                                .require_residual_blocker_efficiency = require_residual_blocker_efficiency,
                                .top_k_segment_vote_alternatives = top_k_segment_vote_alternatives,
                                .replay_zero_protection_penalty = replay_zero_protection_penalty,
                                .replay_one_protection_penalty = replay_one_protection_penalty,
                                .replay_taboo_flip_penalty = replay_taboo_flip_penalty,
                                .replay_examples = replay_examples,
                                .replay_example_count = replay_example_count,
                                .strict_replay_guard = strict_replay_guard,
                                .strict_replay_taboo_rejection = strict_replay_taboo_rejection,
                                .replay_safe_quota_complete_only = replay_safe_quota_complete_only,
                                .reserve_replay_protected_zero_votes = reserve_replay_protected_zero_votes
                        };
                        itty_feed_model_projected_repair_stats_t batch_stats = { 0 };
                        if (!itty_feed_model_train_penultimate_layer_with_final_repairs (model,
                                                                                         input,
                                                                                         target,
                                                                                         &batch_options,
                                                                                         &batch_stats)) {
                                itty_feed_model_restore_layer_state_snapshot (model,
                                                                              penultimate_layer,
                                                                              snapshot);
                                if (replay_baseline_folded) {
                                        for (size_t replay_index = 0; replay_index < replay_example_count; replay_index++)
                                                itty_bit_string_free (replay_baseline_folded[replay_index]);
                                        free (replay_baseline_folded);
                                        free (replay_baseline_objectives);
                                }
                                return false;
                        }

                        itty_feed_model_decoder_objective_t next_objective = { 0 };
                        if (!itty_feed_model_measure_decoder_objective (model,
                                                                        input,
                                                                        target,
                                                                        &next_objective)) {
                                itty_feed_model_restore_layer_state_snapshot (model,
                                                                              penultimate_layer,
                                                                              snapshot);
                                if (replay_baseline_folded) {
                                        for (size_t replay_index = 0; replay_index < replay_example_count; replay_index++)
                                                itty_bit_string_free (replay_baseline_folded[replay_index]);
                                        free (replay_baseline_folded);
                                        free (replay_baseline_objectives);
                                }
                                return false;
                        }

                        itty_feed_model_refreshed_projected_repair_round_t attempt_stats = round_stats;
                        itty_feed_model_fill_refreshed_round_diagnostics (&attempt_stats,
                                                                          &current_objective,
                                                                          &next_objective,
                                                                          &batch_stats);
                        bool replay_safe = true;
                        if (replay_example_count > 0)
                                replay_safe = itty_feed_model_score_replay_after_batch (model,
                                                                                       replay_examples,
                                                                                       replay_example_count,
                                                                                       strict_replay_guard,
                                                                                       replay_baseline_objectives,
                                                                                       replay_baseline_folded,
                                                                                       &attempt_stats);

                        if (!replay_safe) {
                                itty_feed_model_restore_layer_state_snapshot (model,
                                                                              penultimate_layer,
                                                                              snapshot);
                                if (stats) {
                                        stats->replay_rejected_batches++;
                                        stats->replay_bisections++;
                                }
                                if (effective_batch_size > 1) {
                                        effective_batch_size = effective_batch_size / 2;
                                        if (effective_batch_size == 0)
                                                effective_batch_size = 1;
                                        continue;
                                }

                                attempt_stats.reverted = true;
                                attempt_stats.rejection_reason = ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_REPLAY_GUARD;
                                round_stats = attempt_stats;
                                if (round < trajectory_count)
                                        trajectory[round] = round_stats;
                                if (stats) {
                                stats->rounds++;
                                stats->rejected_rounds++;
                                stats->rejection_reason = round_stats.rejection_reason;
                                stats->rejected_round = round_stats;
                                itty_feed_model_accumulate_no_effect_candidate_stats (&stats->projected,
                                                                                      &batch_stats);
                        }
                                round_finished = true;
                                break;
                        }

                        if (!itty_feed_model_decoder_objective_accepts (&current_objective,
                                                                        &next_objective)) {
                                attempt_stats.rejection_reason = itty_feed_model_classify_rejected_batch (&current_objective,
                                                                                                          &next_objective,
                                                                                                          &batch_stats);
                                if (attempt_stats.rejection_reason == ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_LOCAL_ACTIVATION_ONLY)
                                        batch_stats.replay_safe_quota_rejected_local_only_decoded_bits +=
                                                batch_stats.replay_safe_quota_selected_decoded_bits;
                                itty_feed_model_restore_layer_state_snapshot (model,
                                                                              penultimate_layer,
                                                                              snapshot);
                                attempt_stats.reverted = true;
                                round_stats = attempt_stats;
                                if (round < trajectory_count)
                                        trajectory[round] = round_stats;
                                if (stats) {
                                        stats->rounds++;
                                        stats->rejected_rounds++;
                                        stats->rejection_reason = round_stats.rejection_reason;
                                        stats->rejected_round = round_stats;
                                        itty_feed_model_accumulate_no_effect_candidate_stats (&stats->projected,
                                                                                              &batch_stats);
                                }
                                round_finished = true;
                                break;
                        }

                        itty_feed_model_free_layer_state_snapshot (model,
                                                                   snapshot);
                        round_stats = attempt_stats;
                        round_stats.accepted = true;
                        if (round < trajectory_count)
                                trajectory[round] = round_stats;
                        if (stats) {
                                stats->rounds++;
                                stats->accepted_rounds++;
                                stats->replay_examples_scored += round_stats.replay_examples_scored;
                                stats->replay_transitions.correct_zero_to_false_positive_bits += round_stats.replay_transitions.correct_zero_to_false_positive_bits;
                                stats->replay_transitions.correct_one_to_false_negative_bits += round_stats.replay_transitions.correct_one_to_false_negative_bits;
                                stats->replay_transitions.false_positive_to_correct_zero_bits += round_stats.replay_transitions.false_positive_to_correct_zero_bits;
                                stats->replay_transitions.false_negative_to_correct_one_bits += round_stats.replay_transitions.false_negative_to_correct_one_bits;
                                stats->replay_transitions.false_positive_to_false_negative_bits += round_stats.replay_transitions.false_positive_to_false_negative_bits;
                                stats->replay_transitions.false_negative_to_false_positive_bits += round_stats.replay_transitions.false_negative_to_false_positive_bits;
                                stats->replay_transitions.unchanged_wrong_bits += round_stats.replay_transitions.unchanged_wrong_bits;
                                stats->replay_transitions.unchanged_correct_bits += round_stats.replay_transitions.unchanged_correct_bits;
                                itty_feed_model_accumulate_projected_repair_stats (&stats->projected,
                                                                                   &batch_stats);
                                stats->after_distance = next_objective.selected_distance;
                                stats->after_blockers = next_objective.false_negative_blocker_bits;
                        }
                        if (have_current_before_folded) {
                                itty_bit_string_t *current_after_folded = NULL;
                                itty_feed_model_decoder_objective_t current_after_objective = { 0 };

                                if (itty_feed_model_evaluate_replay_example (model,
                                                                             &current_example,
                                                                             &current_after_objective,
                                                                             &current_after_folded)) {
                                        itty_feed_model_accumulate_replay_transition (&round_stats.current_transitions,
                                                                                      target,
                                                                                      current_before_folded,
                                                                                      current_after_folded);
                                        if (stats) {
                                                size_t cancelled_bits = 0;

                                                stats->current_batch_cancel_target_one_loss +=
                                                        round_stats.current_transitions.correct_one_to_false_negative_bits;
                                                stats->current_batch_cancel_target_zero_loss +=
                                                        round_stats.current_transitions.correct_zero_to_false_positive_bits;
                                                if (current_after_objective.selected_node != current_objective.selected_node)
                                                        stats->current_batch_cancel_selected_node_change++;

                                                for (size_t trace_index = 0;
                                                     trace_index < batch_stats.replay_safe_quota_effect_trace_count;
                                                     trace_index++) {
                                                        itty_feed_model_safe_quota_effect_t *trace =
                                                                &batch_stats.replay_safe_quota_effect_traces[trace_index];
                                                        bool target_bit;
                                                        bool before_bit;
                                                        bool after_bit;

                                                        if (trace->decoded_bit >= itty_bit_string_get_length (target))
                                                                continue;

                                                        target_bit = itty_bit_string_get_bit (target,
                                                                                              trace->decoded_bit);
                                                        before_bit = itty_bit_string_get_bit (current_before_folded,
                                                                                              trace->decoded_bit);
                                                        after_bit = itty_bit_string_get_bit (current_after_folded,
                                                                                             trace->decoded_bit);

                                                        if (!trace->quota_realized)
                                                                continue;

                                                        if (after_bit == target_bit &&
                                                            before_bit != target_bit) {
                                                                batch_stats.replay_safe_quota_net_positive_decoded_bits++;
                                                        } else if (trace->penultimate_threshold_crossed) {
                                                                batch_stats.replay_safe_quota_cancelled_decoded_bits++;
                                                                cancelled_bits++;
                                                        }
                                                }

                                                stats->current_batch_cancel_duplicate_or_overlap_side_effect += cancelled_bits;
                                        }
                                        itty_bit_string_free (current_after_folded);
                                }
                        }
                        if (stats) {
                                stats->current_transitions.correct_zero_to_false_positive_bits += round_stats.current_transitions.correct_zero_to_false_positive_bits;
                                stats->current_transitions.correct_one_to_false_negative_bits += round_stats.current_transitions.correct_one_to_false_negative_bits;
                                stats->current_transitions.false_positive_to_correct_zero_bits += round_stats.current_transitions.false_positive_to_correct_zero_bits;
                                stats->current_transitions.false_negative_to_correct_one_bits += round_stats.current_transitions.false_negative_to_correct_one_bits;
                                stats->current_transitions.false_positive_to_false_negative_bits += round_stats.current_transitions.false_positive_to_false_negative_bits;
                                stats->current_transitions.false_negative_to_false_positive_bits += round_stats.current_transitions.false_negative_to_false_positive_bits;
                                stats->current_transitions.unchanged_wrong_bits += round_stats.current_transitions.unchanged_wrong_bits;
                                stats->current_transitions.unchanged_correct_bits += round_stats.current_transitions.unchanged_correct_bits;
                                stats->current_candidate_net_positive += batch_stats.current_candidate_net_positive;
                                stats->current_candidate_net_zero += batch_stats.current_candidate_net_zero;
                                stats->current_candidate_net_negative += batch_stats.current_candidate_net_negative;
                                stats->current_helpful_decoded_bits +=
                                        round_stats.current_transitions.false_negative_to_correct_one_bits +
                                        round_stats.current_transitions.false_positive_to_correct_zero_bits;
                                stats->current_harmed_decoded_bits +=
                                        round_stats.current_transitions.correct_one_to_false_negative_bits +
                                        round_stats.current_transitions.correct_zero_to_false_positive_bits;
                                stats->current_neutral_decoded_bits +=
                                        round_stats.current_transitions.false_positive_to_false_negative_bits +
                                        round_stats.current_transitions.false_negative_to_false_positive_bits +
                                        round_stats.current_transitions.unchanged_wrong_bits +
                                        round_stats.current_transitions.unchanged_correct_bits;
                        }
                        current_objective = next_objective;
                        round_finished = true;
                }

                if (current_before_folded)
                        itty_bit_string_free (current_before_folded);
                if (round_stats.reverted)
                        break;
        }

        if (replay_baseline_folded) {
                for (size_t replay_index = 0; replay_index < replay_example_count; replay_index++)
                        itty_bit_string_free (replay_baseline_folded[replay_index]);
                free (replay_baseline_folded);
                free (replay_baseline_objectives);
        }

        return true;
}

bool
itty_feed_model_train_segment_condense_quota_repair_projection (itty_feed_model_t                                          *model,
                                                                itty_bit_string_list_t                                     *input,
                                                                itty_bit_string_t                                          *target,
                                                                itty_feed_model_refreshed_projected_repair_options_t const *options,
                                                                itty_feed_model_segment_training_summary_t                 *summary)
{
        if (summary)
                *summary = (itty_feed_model_segment_training_summary_t) { 0 };

        if (!model || !input || !target)
                return false;

        itty_feed_model_set_decoder (model,
                                     ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);

        itty_feed_model_decoder_objective_t before_objective = { 0 };
        itty_feed_model_decoder_objective_t after_objective = { 0 };
        itty_model_metrics_bit_summary_t masks_before = { 0 };
        itty_model_metrics_bit_summary_t masks_after = { 0 };
        itty_feed_model_refreshed_projected_repair_stats_t training_stats = { 0 };

        if (!itty_feed_model_measure_decoder_objective (model,
                                                        input,
                                                        target,
                                                        &before_objective))
                return false;

        itty_feed_model_measure_masks (model,
                                       &masks_before);

        if (!itty_feed_model_train_penultimate_layer_with_refreshed_final_repairs (model,
                                                                                   input,
                                                                                   target,
                                                                                   options,
                                                                                   &training_stats))
                return false;

        if (!itty_feed_model_measure_decoder_objective (model,
                                                        input,
                                                        target,
                                                        &after_objective))
                return false;

        itty_feed_model_measure_masks (model,
                                       &masks_after);

        if (summary) {
                size_t fixed_bits = before_objective.selected_distance > after_objective.selected_distance ?
                                    before_objective.selected_distance - after_objective.selected_distance :
                                    0;
                size_t deficit_reduced = before_objective.false_negative_vote_deficit > after_objective.false_negative_vote_deficit ?
                                         before_objective.false_negative_vote_deficit - after_objective.false_negative_vote_deficit :
                                         0;
                size_t target_one_bits = itty_bit_string_get_pop_count (target);
                size_t correct_target_one_bits = target_one_bits > after_objective.false_negative_count ?
                                                 target_one_bits - after_objective.false_negative_count :
                                                 0;
                size_t decoded_bits_fixed = before_objective.false_negative_count > after_objective.false_negative_count ?
                                            before_objective.false_negative_count - after_objective.false_negative_count :
                                            0;

                summary->initial_distance = before_objective.selected_distance;
                summary->final_distance = after_objective.selected_distance;
                if (after_objective.selected_distance == 0)
                        summary->stop_reason = ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_CONVERGED;
                else if (training_stats.rejected_rounds > 0)
                        summary->stop_reason = training_stats.rejection_reason;
                else if (options &&
                         options->max_rounds != 0 &&
                         training_stats.rounds >= options->max_rounds)
                        summary->stop_reason = ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_MAX_ROUNDS;
                else if (training_stats.projected.accepted_blocks == 0) {
                        if (training_stats.projected.replay_safe_quota_blocked_decoded_bits > 0 &&
                            training_stats.projected.replay_safe_quota_feasible_decoded_bits == 0)
                                summary->stop_reason = ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_CAPACITY_CONFLICT;
                        else if (training_stats.projected.replay_best_candidate_unsafe)
                                summary->stop_reason = ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_SINGLE_CANDIDATE_CONFLICT;
                        else if (training_stats.projected.replay_unsafe_candidates > 0 &&
                                 training_stats.projected.replay_safe_candidates == 0)
                                summary->stop_reason = ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_BLOCKED_ALL_CANDIDATES;
                        else if (options &&
                                 options->replay_example_count > 0 &&
                                 (training_stats.projected.replay_safe_candidates > 0 ||
                                  training_stats.projected.no_effect_candidates > 0))
                                summary->stop_reason = ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_SAFE_NO_CURRENT_GAIN;
                        else if (training_stats.projected.no_effect_candidates > 0)
                                summary->stop_reason = ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_EFFECTIVE_CANDIDATES;
                        else
                                summary->stop_reason = ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_REPAIRS;
                }
                else
                        summary->stop_reason = ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_PROGRESS;
                summary->layer6_flips = training_stats.projected.layer_flips;
                summary->rounds = training_stats.rounds;
                summary->accepted_blocks = training_stats.projected.accepted_blocks;
                summary->quota_size_total = training_stats.projected.quota_size_total;
                summary->quota_size_max = training_stats.projected.quota_size_max;
                summary->average_quota_size = training_stats.projected.accepted_blocks == 0 ? 0.0 :
                                              (double) training_stats.projected.quota_size_total /
                                              (double) training_stats.projected.accepted_blocks;
                summary->false_negative_vote_deficit_before = before_objective.false_negative_vote_deficit;
                summary->false_negative_vote_deficit_after = after_objective.false_negative_vote_deficit;
                summary->false_positive_vote_excess_before = before_objective.false_positive_vote_excess;
                summary->false_positive_vote_excess_after = after_objective.false_positive_vote_excess;
                summary->target_zero_safety_minimum = before_objective.target_zero_safety_min < after_objective.target_zero_safety_min ?
                                                      before_objective.target_zero_safety_min :
                                                      after_objective.target_zero_safety_min;
                summary->mask_entropy_before = masks_before.entropy;
                summary->mask_entropy_after = masks_after.entropy;
                summary->quota_completion_efficiency = training_stats.projected.layer_flips == 0 ? 0.0 :
                                                       (double) fixed_bits /
                                                       (double) training_stats.projected.layer_flips;
                summary->vote_efficiency = training_stats.projected.layer_flips == 0 ? 0.0 :
                                           (double) deficit_reduced /
                                           (double) training_stats.projected.layer_flips;
                summary->direct_quota_vote_flips = training_stats.projected.direct_quota_vote_flips;
                summary->majority_threshold_support_flips = training_stats.projected.majority_threshold_support_flips;
                summary->conflict_resolution_flips = training_stats.projected.conflict_resolution_flips;
                summary->collateral_preservation_flips = training_stats.projected.collateral_preservation_flips;
                summary->target_zero_safety_preservation_flips = training_stats.projected.target_zero_safety_preservation_flips;
                summary->selection_preservation_flips = training_stats.projected.selection_preservation_flips;
                for (size_t bucket = 0; bucket < ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS; bucket++)
                        summary->quota_vote_support_cost_histogram[bucket] =
                                training_stats.projected.quota_vote_support_cost_histogram[bucket];
                summary->decoded_bits_fixed = decoded_bits_fixed;
                summary->average_flips_per_fixed_decoded_bit = decoded_bits_fixed == 0 ? 0.0 :
                                                               (double) training_stats.projected.layer_flips /
                                                               (double) decoded_bits_fixed;
                summary->min_flips_per_fixed_decoded_bit = training_stats.projected.fixed_decoded_bit_flips_min;
                summary->max_flips_per_fixed_decoded_bit = training_stats.projected.fixed_decoded_bit_flips_max;
                summary->average_final_target_one_margin = correct_target_one_bits == 0 ? 0.0 :
                                                           (double) after_objective.target_one_margin /
                                                           (double) correct_target_one_bits;
                summary->training = training_stats;
        }

        return true;
}

bool
itty_feed_model_train_segment_condense_with_summary (itty_feed_model_t                                          *model,
                                                     itty_bit_string_list_t                                     *input,
                                                     itty_bit_string_t                                          *target,
                                                     itty_feed_model_refreshed_projected_repair_options_t const *options,
                                                     itty_feed_model_segment_training_summary_t                 *summary)
{
        return itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                               input,
                                                                               target,
                                                                               options,
                                                                               summary);
}

bool
itty_feed_model_train_antepenultimate_layer_with_projected_repairs (itty_feed_model_t                                *model,
                                                                    itty_bit_string_list_t                           *input,
                                                                    itty_bit_string_t                                *target,
                                                                    itty_feed_model_projected_repair_options_t const *options,
                                                                    itty_feed_model_projected_repair_stats_t         *stats)
{
        if (stats)
                *stats = (itty_feed_model_projected_repair_stats_t) { 0 };

        if (!model ||
            !input ||
            !target ||
            model->number_of_layers < 3 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;

        size_t layer_index = model->number_of_layers - 3;
        size_t next_layer = layer_index + 1;
        size_t final_layer = model->number_of_layers - 1;
        size_t max_projected_blocks = options ? options->max_projected_blocks : 0;
        size_t max_layer_flips = options ? options->max_layer_flips : 0;
        size_t max_blocks_per_final_node = options ? options->max_blocks_per_final_node : 0;
        size_t max_strict_distance_blocks = options ? options->max_strict_distance_blocks : 0;
        size_t max_blocker_blocks = options ? options->max_blocker_blocks : 0;
        bool limit_strict_distance_blocks = options ? options->limit_strict_distance_blocks : false;
        bool limit_blocker_blocks = options ? options->limit_blocker_blocks : false;

        itty_bit_string_list_t *layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                  input,
                                                                                  layer_index);
        itty_bit_string_list_t *layer_outputs = itty_feed_model_run_layer (model,
                                                                           layer_index,
                                                                           layer_input);
        itty_bit_string_list_t *next_outputs = itty_feed_model_run_layer (model,
                                                                          next_layer,
                                                                          layer_outputs);
        itty_feed_model_decoder_objective_t before_objective = { 0 };
        if (!itty_feed_model_evaluate_suffix_decoder_objective (model,
                                                                layer_outputs,
                                                                layer_index,
                                                                target,
                                                                &before_objective)) {
                if (layer_input != input)
                        itty_bit_string_list_free (layer_input);
                itty_bit_string_list_free (layer_outputs);
                itty_bit_string_list_free (next_outputs);
                return false;
        }

        if (stats) {
                stats->before_distance = before_objective.selected_distance;
                stats->before_blockers = before_objective.false_negative_blocker_bits;
        }

        itty_bit_string_t *before_folded = NULL;
        if (!itty_feed_model_fold_suffix_selected_output (model,
                                                          layer_outputs,
                                                          layer_index,
                                                          target,
                                                          &before_folded)) {
                if (layer_input != input)
                        itty_bit_string_list_free (layer_input);
                itty_bit_string_list_free (layer_outputs);
                itty_bit_string_list_free (next_outputs);
                return false;
        }

        itty_feed_model_final_repair_list_t repairs = { 0 };
        if (!itty_feed_model_collect_final_repairs (model,
                                                   next_outputs,
                                                   target,
                                                   &repairs)) {
                itty_bit_string_free (before_folded);
                if (layer_input != input)
                        itty_bit_string_list_free (layer_input);
                itty_bit_string_list_free (layer_outputs);
                itty_bit_string_list_free (next_outputs);
                return false;
        }

        itty_feed_model_projected_repair_candidate_t *candidates = NULL;
        size_t candidate_count = 0;
        size_t candidate_capacity = 0;
        size_t projected_blocks = 0;

        for (size_t repair_index = 0; repair_index < repairs.count; repair_index++) {
                if (max_projected_blocks != 0 &&
                    projected_blocks >= max_projected_blocks)
                        break;

                itty_feed_model_final_repair_t *repair = &repairs.items[repair_index];
                itty_feed_model_layer_assignment_list_t next_output_assignments = { 0 };
                if (!itty_feed_model_project_repair_to_previous_layer_outputs (model,
                                                                               final_layer,
                                                                               next_outputs,
                                                                               repair,
                                                                               &next_output_assignments)) {
                        itty_feed_model_layer_assignment_list_clear (&next_output_assignments);
                        continue;
                }

                itty_feed_model_layer_assignment_list_t next_condensed_assignments = { 0 };
                if (!itty_feed_model_make_condensed_assignments_from_outputs (model,
                                                                              next_layer,
                                                                              &next_output_assignments,
                                                                              &next_condensed_assignments)) {
                        itty_feed_model_layer_assignment_list_clear (&next_condensed_assignments);
                        itty_feed_model_layer_assignment_list_clear (&next_output_assignments);
                        continue;
                }

                projected_blocks++;
                if (stats)
                        stats->projected_blocks = projected_blocks;

                itty_feed_model_layer_assignment_list_t output_assignments = { 0 };
                for (size_t assignment_index = 0; assignment_index < next_condensed_assignments.count; assignment_index++) {
                        itty_feed_model_layer_assignment_t *assignment = &next_condensed_assignments.items[assignment_index];
                        if (!itty_feed_model_project_layer_repair_to_previous_outputs (model,
                                                                                       next_layer,
                                                                                       layer_outputs,
                                                                                       assignment->layer_node,
                                                                                       assignment->bit_index,
                                                                                       assignment->value,
                                                                                       &output_assignments))
                                break;
                }

                itty_feed_model_layer_assignment_list_t condensed_assignments = { 0 };
                if (!itty_feed_model_make_condensed_assignments_from_outputs (model,
                                                                              layer_index,
                                                                              &output_assignments,
                                                                              &condensed_assignments)) {
                        itty_feed_model_layer_assignment_list_clear (&condensed_assignments);
                        itty_feed_model_layer_assignment_list_clear (&output_assignments);
                        itty_feed_model_layer_assignment_list_clear (&next_condensed_assignments);
                        itty_feed_model_layer_assignment_list_clear (&next_output_assignments);
                        continue;
                }

                itty_bit_string_list_t *candidate_outputs = itty_feed_model_make_condensed_realistic_outputs (model,
                                                                                                              layer_index,
                                                                                                              layer_outputs,
                                                                                                              &condensed_assignments);
                itty_feed_model_decoder_objective_t candidate_objective = { 0 };
                itty_bit_string_t *candidate_folded = NULL;
                bool scored = itty_feed_model_evaluate_suffix_decoder_objective (model,
                                                                                 candidate_outputs,
                                                                                 layer_index,
                                                                                 target,
                                                                                 &candidate_objective) &&
                              itty_feed_model_fold_suffix_selected_output (model,
                                                                           candidate_outputs,
                                                                           layer_index,
                                                                           target,
                                                                           &candidate_folded);
                bool strict_helpful = scored &&
                                      candidate_objective.selected_distance < before_objective.selected_distance;
                bool blocker_helpful = scored &&
                                       candidate_objective.selected_distance == before_objective.selected_distance &&
                                       candidate_objective.false_negative_blocker_bits < before_objective.false_negative_blocker_bits &&
                                       itty_feed_model_preserves_correct_decoded_bits (before_folded,
                                                                                      candidate_folded,
                                                                                      target);

                if (stats && scored) {
                        stats->condensed_realistic_blocks++;
                        if (strict_helpful)
                                stats->condensed_realistic_strict_distance_helpful_blocks++;
                        else if (candidate_objective.selected_distance > before_objective.selected_distance)
                                stats->condensed_realistic_harmful_blocks++;
                        else if (blocker_helpful)
                                stats->condensed_realistic_blocker_helpful_blocks++;
                        else
                                stats->condensed_realistic_neutral_blocks++;
                }

                if (strict_helpful || blocker_helpful) {
                        if (candidate_count == candidate_capacity) {
                                candidate_capacity = candidate_capacity == 0 ? 16 : candidate_capacity * 2;
                                candidates = realloc (candidates,
                                                      candidate_capacity * sizeof (itty_feed_model_projected_repair_candidate_t));
                        }

                        itty_feed_model_projected_repair_candidate_t *candidate = &candidates[candidate_count];
                        *candidate = (itty_feed_model_projected_repair_candidate_t) {
                                .objective = candidate_objective,
                                .final_node = repair->final_node,
                                .original_index = repair_index,
                                .decoded_bit = repair->decoded_bit,
                                .quota_size = repair->quota_size,
                                .distance_delta = itty_feed_model_positive_delta (before_objective.selected_distance,
                                                                                  candidate_objective.selected_distance),
                                .false_negative_count_delta = itty_feed_model_positive_delta (before_objective.false_negative_count,
                                                                                              candidate_objective.false_negative_count),
                                .blocker_delta = itty_feed_model_positive_delta (before_objective.false_negative_blocker_bits,
                                                                                 candidate_objective.false_negative_blocker_bits),
                                .vote_deficit_delta = itty_feed_model_positive_delta (before_objective.false_negative_vote_deficit,
                                                                                      candidate_objective.false_negative_vote_deficit),
                                .target_one_margin_delta = itty_feed_model_positive_increase (before_objective.target_one_margin,
                                                                                              candidate_objective.target_one_margin),
                                .rank = strict_helpful ?
                                        ITTY_FEED_MODEL_PROJECTED_REPAIR_STRICT_DISTANCE :
                                        ITTY_FEED_MODEL_PROJECTED_REPAIR_BLOCKER
                        };
                        itty_feed_model_layer_assignment_list_copy (&candidate->output_assignments,
                                                                    &output_assignments);
                        itty_feed_model_layer_assignment_list_copy (&candidate->condensed_assignments,
                                                                    &condensed_assignments);
                        itty_feed_model_measure_condensed_assignment_cost (model,
                                                                          layer_index,
                                                                          layer_input,
                                                                          &candidate->condensed_assignments,
                                                                          &candidate->estimated_flips,
                                                                          &candidate->already_satisfied_bits,
                                                                          &candidate->bits_needing_flips,
                                                                          &candidate->available_flippable_votes);
                        candidate_count++;
                }

                if (candidate_folded)
                        itty_bit_string_free (candidate_folded);
                itty_bit_string_list_free (candidate_outputs);
                itty_feed_model_layer_assignment_list_clear (&condensed_assignments);
                itty_feed_model_layer_assignment_list_clear (&output_assignments);
                itty_feed_model_layer_assignment_list_clear (&next_condensed_assignments);
                itty_feed_model_layer_assignment_list_clear (&next_output_assignments);
        }

        qsort (candidates,
               candidate_count,
               sizeof (itty_feed_model_projected_repair_candidate_t),
               compare_projected_repair_candidates);

        size_t condensed_words = model->vocabulary_words << layer_index;
        itty_bit_string_t **condensed_targets = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));
        itty_bit_string_t **condensed_cares = calloc (model->nodes_per_layer, sizeof (itty_bit_string_t *));
        size_t *blocks_by_final_node = calloc (model->nodes_per_layer, sizeof (size_t));
        size_t remaining_estimated_flips = max_layer_flips;
        size_t selected_estimated_flips = 0;
        size_t accepted_strict_distance_blocks = 0;
        size_t accepted_blocker_blocks = 0;
        size_t strict_distance_estimated_flips = 0;
        size_t blocker_estimated_flips = 0;

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_t *current_condensed = itty_feed_model_run_node_condensed (layer_input,
                                                                                           model->masks_by_layer_node[layer_index][node_index]);
                condensed_targets[node_index] = itty_feed_model_bit_string_clone_to_words (current_condensed,
                                                                                           condensed_words);
                condensed_cares[node_index] = itty_feed_model_zero_mask_new (condensed_words);
                itty_bit_string_free (current_condensed);
        }

        for (size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
                itty_feed_model_projected_repair_candidate_t *candidate = &candidates[candidate_index];

                if (max_blocks_per_final_node != 0 &&
                    blocks_by_final_node[candidate->final_node] >= max_blocks_per_final_node)
                        continue;
                if (limit_strict_distance_blocks &&
                    candidate->rank == ITTY_FEED_MODEL_PROJECTED_REPAIR_STRICT_DISTANCE &&
                    accepted_strict_distance_blocks >= max_strict_distance_blocks)
                        continue;
                if (limit_blocker_blocks &&
                    candidate->rank == ITTY_FEED_MODEL_PROJECTED_REPAIR_BLOCKER &&
                    accepted_blocker_blocks >= max_blocker_blocks)
                        continue;
                if (max_layer_flips != 0 &&
                    candidate->estimated_flips > remaining_estimated_flips)
                        continue;

                bool conflicted = false;
                for (size_t assignment_index = 0; assignment_index < candidate->condensed_assignments.count; assignment_index++) {
                        itty_feed_model_layer_assignment_t *assignment = &candidate->condensed_assignments.items[assignment_index];

                        if (itty_bit_string_get_bit (condensed_cares[assignment->layer_node],
                                                     assignment->bit_index) &&
                            itty_bit_string_get_bit (condensed_targets[assignment->layer_node],
                                                     assignment->bit_index) != assignment->value) {
                                conflicted = true;
                                break;
                        }
                }

                if (conflicted) {
                        if (stats)
                                stats->conflicts++;
                        continue;
                }

                selected_estimated_flips += candidate->estimated_flips;
                if (candidate->rank == ITTY_FEED_MODEL_PROJECTED_REPAIR_STRICT_DISTANCE) {
                        accepted_strict_distance_blocks++;
                        strict_distance_estimated_flips += candidate->estimated_flips;
                } else if (candidate->rank == ITTY_FEED_MODEL_PROJECTED_REPAIR_BLOCKER) {
                        accepted_blocker_blocks++;
                        blocker_estimated_flips += candidate->estimated_flips;
                }
                if (max_layer_flips != 0)
                        remaining_estimated_flips -= candidate->estimated_flips;
                if (stats) {
                        stats->accepted_blocks++;
                        stats->quota_size_total += candidate->quota_size;
                        if (candidate->quota_size > stats->quota_size_max)
                                stats->quota_size_max = candidate->quota_size;
                        stats->direct_quota_vote_flips += candidate->bits_needing_flips;
                        if (candidate->estimated_flips > candidate->bits_needing_flips)
                                stats->majority_threshold_support_flips += candidate->estimated_flips -
                                                                           candidate->bits_needing_flips;
                        stats->accepted_strict_distance_blocks = accepted_strict_distance_blocks;
                        stats->accepted_blocker_blocks = accepted_blocker_blocks;
                        stats->strict_distance_layer_flips = strict_distance_estimated_flips;
                        stats->blocker_layer_flips = blocker_estimated_flips;
                        stats->estimated_layer_flips += candidate->estimated_flips;
                }

                for (size_t assignment_index = 0; assignment_index < candidate->condensed_assignments.count; assignment_index++) {
                        itty_feed_model_layer_assignment_t *assignment = &candidate->condensed_assignments.items[assignment_index];
                        itty_bit_string_set_bit (condensed_cares[assignment->layer_node],
                                                 assignment->bit_index,
                                                 true);
                        itty_bit_string_set_bit (condensed_targets[assignment->layer_node],
                                                 assignment->bit_index,
                                                 assignment->value);
                }
        }

        itty_feed_model_layer_state_snapshot_t *snapshot = itty_feed_model_snapshot_layer_state (model,
                                                                                                 layer_index);
        size_t remaining_flips = max_layer_flips == 0 ? 0 : selected_estimated_flips;
        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_feed_model_train_stats_t node_stats = { 0 };

                itty_feed_model_train_layer_one_node_with_care (model->masks_by_layer_node[layer_index][node_index],
                                                                layer_input,
                                                                condensed_targets[node_index],
                                                                condensed_cares[node_index],
                                                                remaining_flips,
                                                                &node_stats);
                if (stats)
                        stats->layer_flips += node_stats.flips;

                if (max_layer_flips != 0) {
                        if (node_stats.flips >= remaining_flips)
                                remaining_flips = 0;
                        else
                                remaining_flips -= node_stats.flips;
                        if (remaining_flips == 0)
                                break;
                }
        }

        itty_feed_model_decoder_objective_t after_objective = { 0 };
        bool keep = itty_feed_model_measure_decoder_objective (model,
                                                               input,
                                                               target,
                                                               &after_objective) &&
                    itty_feed_model_decoder_objective_accepts (&before_objective,
                                                               &after_objective);
        if (!keep) {
                itty_feed_model_restore_layer_state_snapshot (model,
                                                              layer_index,
                                                              snapshot);
                if (stats)
                        stats->layer_flips = 0;
        } else {
                itty_feed_model_free_layer_state_snapshot (model,
                                                           snapshot);
        }

        if (stats) {
                stats->after_distance = keep ? after_objective.selected_distance : before_objective.selected_distance;
                stats->after_blockers = keep ? after_objective.false_negative_blocker_bits : before_objective.false_negative_blocker_bits;
        }

        for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                itty_bit_string_free (condensed_targets[node_index]);
                itty_bit_string_free (condensed_cares[node_index]);
        }
        for (size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++)
                itty_feed_model_projected_repair_candidate_clear (&candidates[candidate_index]);
        if (stats) {
                stats->direct_quota_vote_flips = stats->requested_condensed_bits < stats->layer_flips ?
                                                 stats->requested_condensed_bits :
                                                 stats->layer_flips;
                stats->majority_threshold_support_flips = stats->layer_flips - stats->direct_quota_vote_flips;
                stats->conflict_resolution_flips = 0;
                stats->collateral_preservation_flips = 0;
                stats->target_zero_safety_preservation_flips = 0;
                stats->selection_preservation_flips = 0;
        }
        free (blocks_by_final_node);
        free (condensed_cares);
        free (condensed_targets);
        free (candidates);
        itty_feed_model_final_repair_list_clear (&repairs);
        itty_bit_string_free (before_folded);
        itty_bit_string_list_free (next_outputs);
        itty_bit_string_list_free (layer_outputs);
        if (layer_input != input)
                itty_bit_string_list_free (layer_input);

        return true;
}

bool
itty_feed_model_train_backwards_one (itty_feed_model_t      *model,
                                     itty_bit_string_list_t *input,
                                     itty_bit_string_t      *target)
{
        return itty_feed_model_train_backwards_one_with_stats (model, input, target, NULL, NULL);
}

bool
itty_feed_model_train_backwards_one_with_options (itty_feed_model_t                     *model,
                                                  itty_bit_string_list_t                *input,
                                                  itty_bit_string_t                     *target,
                                                  itty_feed_model_train_options_t const *options)
{
        return itty_feed_model_train_backwards_one_with_stats (model, input, target, options, NULL);
}

bool
itty_feed_model_train_backwards_one_with_stats (itty_feed_model_t                     *model,
                                                itty_bit_string_list_t                *input,
                                                itty_bit_string_t                     *target,
                                                itty_feed_model_train_options_t const *options,
                                                itty_feed_model_train_stats_t         *stats)
{
        return itty_feed_model_train_backwards_one_with_layer_stats (model,
                                                                     input,
                                                                     target,
                                                                     options,
                                                                     stats,
                                                                     NULL,
                                                                     0);
}

bool
itty_feed_model_train_backwards_one_with_layer_stats (itty_feed_model_t                     *model,
                                                      itty_bit_string_list_t                *input,
                                                      itty_bit_string_t                     *target,
                                                      itty_feed_model_train_options_t const *options,
                                                      itty_feed_model_train_stats_t         *stats,
                                                      itty_feed_model_train_stats_t         *layer_stats,
                                                      size_t                                 layer_stats_count)
{
        if (stats)
                *stats = (itty_feed_model_train_stats_t) { 0 };
        if (layer_stats) {
                for (size_t i = 0; i < layer_stats_count; i++)
                        layer_stats[i] = (itty_feed_model_train_stats_t) { 0 };
        }

        if (model->number_of_layers == 0 ||
            itty_bit_string_list_get_length (input) != model->inputs_per_node ||
            itty_bit_string_get_number_of_words (target) > model->vocabulary_words)
                return false;
        if (!itty_feed_model_can_train_with_options (options))
                return false;

        itty_feed_model_backward_fold_t backward_fold = itty_feed_model_get_backward_fold (options);
        itty_feed_model_backward_node_target_t backward_node_target = itty_feed_model_get_backward_node_target (options);
        if (model->nodes_per_layer != 1 &&
            backward_fold != ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE)
                return false;
        if ((backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DISAGREEMENT ||
             backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_PAIRWISE_AND_SEGMENT ||
             backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SEGMENT_PARTITION ||
             backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE ||
             backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_WEIGHTED ||
             backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_GATED) &&
            (backward_fold != ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE ||
             model->nodes_per_layer != 2))
                return false;

        if (layer_stats && layer_stats_count < model->number_of_layers)
                return false;

        itty_bit_string_t *desired_output = itty_feed_model_expand_target_for_layer (target,
                                                                                     model->number_of_layers);
        itty_bit_string_t *pairwise_desired_output = backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_PAIRWISE_AND_SEGMENT ?
                                                     itty_feed_model_expand_target_for_layer (target,
                                                                                              model->number_of_layers) :
                                                     NULL;

        for (size_t layer_count = model->number_of_layers; layer_count > 0; layer_count--) {
                size_t layer_index = layer_count - 1;
                itty_bit_string_list_t *layer_input = itty_feed_model_run_to_layer_input (model,
                                                                                          input,
                                                                                          layer_index);
                size_t *set_votes = NULL;
                size_t layer_words = 0;
                size_t layer_bit_capacity = 0;
                size_t votes_per_bit = 0;
                size_t *partition_set_votes = NULL;
                size_t *partition_vote_counts = NULL;
                size_t partition_layer_words = 0;
                size_t partition_layer_bit_capacity = 0;
                itty_bit_string_t *desired_condensed = NULL;
                itty_bit_string_t *pairwise_desired_condensed = NULL;
                itty_bit_string_t *next_desired_condensed = NULL;

                if (backward_fold == ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE &&
                    backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DISAGREEMENT) {
                        if (itty_feed_model_count_segment_votes_for_layer (model,
                                                                           desired_output,
                                                                           layer_index,
                                                                           &set_votes,
                                                                           &layer_words,
                                                                           &layer_bit_capacity,
                                                                           &votes_per_bit))
                                desired_condensed = itty_feed_model_make_majority_target_from_votes (set_votes,
                                                                                                     layer_words,
                                                                                                     layer_bit_capacity,
                                                                                                     votes_per_bit);
                } else {
                        desired_condensed = backward_fold == ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE ?
                                            itty_feed_model_segment_condense_desired_output_for_layer (model,
                                                                                                       desired_output,
                                                                                                       layer_index) :
                                            itty_feed_model_reduce_desired_output_for_layer (desired_output,
                                                                                                       model->rotations_by_layer[layer_index]);
                }

                if ((backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE ||
                     backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_WEIGHTED ||
                     backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_GATED) &&
                    layer_index + 1 < model->number_of_layers)
                        next_desired_condensed = itty_feed_model_segment_condense_desired_output_for_layer (model,
                                                                                                            desired_output,
                                                                                                            layer_index + 1);

                if (backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SEGMENT_PARTITION &&
                    !itty_feed_model_count_segment_partition_votes_for_layer (model,
                                                                              desired_output,
                                                                              layer_index,
                                                                              model->nodes_per_layer,
                                                                              &partition_set_votes,
                                                                              &partition_vote_counts,
                                                                              &partition_layer_words,
                                                                              &partition_layer_bit_capacity)) {
                        if (layer_input != input)
                                itty_bit_string_list_free (layer_input);
                        itty_bit_string_free (desired_condensed);
                        itty_bit_string_free (pairwise_desired_output);
                        itty_bit_string_free (desired_output);
                        return false;
                }

                if (backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_PAIRWISE_AND_SEGMENT)
                        pairwise_desired_condensed = itty_feed_model_reduce_desired_output_for_layer (pairwise_desired_output,
                                                                                                      model->rotations_by_layer[layer_index]);

                if (!desired_condensed) {
                        if (layer_input != input)
                                itty_bit_string_list_free (layer_input);
                        free (set_votes);
                        free (partition_set_votes);
                        free (partition_vote_counts);
                        itty_bit_string_free (pairwise_desired_condensed);
                        itty_bit_string_free (next_desired_condensed);
                        itty_bit_string_free (pairwise_desired_output);
                        itty_bit_string_free (desired_output);
                        return false;
                }

                itty_feed_model_train_stats_t current_layer_stats = { 0 };

                for (size_t node_index = 0; node_index < model->nodes_per_layer; node_index++) {
                        itty_feed_model_train_stats_t current_node_stats = { 0 };
                        itty_bit_string_list_t *masks = model->masks_by_layer_node[layer_index][node_index];
                        itty_bit_string_t *node_target = desired_condensed;
                        bool node_target_owned = false;

                        if (backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DISAGREEMENT) {
                                node_target = itty_feed_model_make_disagreement_target_for_node (model,
                                                                                                 layer_input,
                                                                                                 layer_index,
                                                                                                 node_index,
                                                                                                 set_votes,
                                                                                                 layer_words,
                                                                                                 layer_bit_capacity,
                                                                                                 votes_per_bit,
                                                                                                 desired_condensed);
                                node_target_owned = true;
                        } else if (backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_PAIRWISE_AND_SEGMENT &&
                                   node_index == 0) {
                                node_target = pairwise_desired_condensed;
                        } else if (backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SEGMENT_PARTITION) {
                                size_t partition_offset = node_index * partition_layer_bit_capacity;

                                node_target = itty_feed_model_make_partition_target_from_votes (partition_set_votes + partition_offset,
                                                                                               partition_vote_counts + partition_offset,
                                                                                                partition_layer_words,
                                                                                                partition_layer_bit_capacity);
                                node_target_owned = true;
                        } else if ((backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE ||
                                    backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_WEIGHTED ||
                                    backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_GATED) &&
                                   next_desired_condensed) {
                                itty_feed_model_downstream_request_mode_t request_mode = ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_EQUAL;

                                if (backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_WEIGHTED)
                                        request_mode = ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_WEIGHTED;
                                else if (backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_GATED)
                                        request_mode = ITTY_FEED_MODEL_DOWNSTREAM_REQUEST_GATED;

                                node_target = itty_feed_model_make_downstream_mask_aware_target_for_node (model,
                                                                                                           layer_input,
                                                                                                           layer_index,
                                                                                                           node_index,
                                                                                                           next_desired_condensed,
                                                                                                           request_mode);
                                node_target_owned = true;
                        }

                        if (!itty_feed_model_train_layer_one_node (masks,
                                                                   layer_input,
                                                                   node_target,
                                                                   options,
                                                                   &current_node_stats)) {
                                if (node_target_owned)
                                        itty_bit_string_free (node_target);
                                if (layer_input != input)
                                        itty_bit_string_list_free (layer_input);
                                free (set_votes);
                                free (partition_set_votes);
                                free (partition_vote_counts);
                                itty_bit_string_free (pairwise_desired_condensed);
                                itty_bit_string_free (next_desired_condensed);
                                itty_bit_string_free (pairwise_desired_output);
                                itty_bit_string_free (desired_condensed);
                                itty_bit_string_free (desired_output);
                                return false;
                        }

                        if (node_target_owned)
                                itty_bit_string_free (node_target);

                        itty_feed_model_accumulate_train_stats (&current_layer_stats,
                                                                &current_node_stats);
                }

                itty_feed_model_accumulate_train_stats (stats, &current_layer_stats);
                if (layer_stats)
                        layer_stats[layer_index] = current_layer_stats;

                if (layer_input != input)
                        itty_bit_string_list_free (layer_input);

                if (backward_node_target == ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_PAIRWISE_AND_SEGMENT) {
                        itty_bit_string_list_t *masks = model->masks_by_layer_node[layer_index][0];
                        itty_bit_string_list_t *desired_layer_input = itty_feed_model_make_desired_layer_input (pairwise_desired_condensed,
                                                                                                                masks);

                        itty_bit_string_free (pairwise_desired_output);
                        pairwise_desired_output = NULL;

                        if (layer_index > 0)
                                pairwise_desired_output = itty_feed_model_bit_string_clone (itty_bit_string_list_fetch (desired_layer_input, 0));

                        itty_bit_string_list_free (desired_layer_input);
                }

                if (backward_fold == ITTY_FEED_MODEL_BACKWARD_FOLD_CHAINED_REDUCE) {
                        itty_bit_string_list_t *masks = model->masks_by_layer_node[layer_index][0];
                        itty_bit_string_list_t *desired_layer_input = itty_feed_model_make_desired_layer_input (desired_condensed,
                                                                                                                masks);

                        itty_bit_string_free (desired_output);
                        desired_output = NULL;

                        if (layer_index > 0) {
                                desired_output = itty_feed_model_bit_string_clone (itty_bit_string_list_fetch (desired_layer_input, 0));
                                itty_bit_string_list_free (desired_layer_input);
                        } else {
                                itty_bit_string_list_free (desired_layer_input);
                        }
                }

                itty_bit_string_free (pairwise_desired_condensed);
                itty_bit_string_free (next_desired_condensed);
                itty_bit_string_free (desired_condensed);
                free (set_votes);
                free (partition_set_votes);
                free (partition_vote_counts);
        }

        itty_bit_string_free (pairwise_desired_output);
        itty_bit_string_free (desired_output);

        return true;
}
