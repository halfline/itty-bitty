#pragma once

#include "itty-bit-string.h"
#include "itty-bit-string-list.h"
#include "itty-model-metrics.h"
#include "itty-network.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct itty_feed_model_t itty_feed_model_t;
typedef struct itty_feed_model_layer_state_snapshot_t itty_feed_model_layer_state_snapshot_t;

#define ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS 9
#define ITTY_FEED_MODEL_DECODER_HISTOGRAM_OVERFLOW_BUCKET (ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS - 1)

typedef enum {
        ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
} itty_feed_model_train_budget_policy_t;

typedef enum {
        ITTY_FEED_MODEL_BACKWARD_FOLD_CHAINED_REDUCE,
        ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE,
} itty_feed_model_backward_fold_t;

typedef enum {
        ITTY_FEED_MODEL_DECODER_REPEATED_AND_FOLD,
        ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE,
} itty_feed_model_decoder_t;

typedef enum {
        ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SAME,
        ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DISAGREEMENT,
        ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_PAIRWISE_AND_SEGMENT,
        ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SEGMENT_PARTITION,
        ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE,
        ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_WEIGHTED,
        ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_GATED,
        ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DECODER_TRI_STATE,
} itty_feed_model_backward_node_target_t;

typedef struct {
        size_t                                      max_flips;
        size_t                                      finish_margin;
        size_t                                      selector_lane_bit_offset;
        size_t                                      selector_lane_bit_count;
        size_t                                      decoder_lane_bit_offset;
        size_t                                      decoder_lane_bit_count;
        itty_feed_model_train_budget_policy_t      budget_policy;
        itty_feed_model_backward_fold_t            backward_fold;
        itty_feed_model_backward_node_target_t     backward_node_target;
} itty_feed_model_train_options_t;

typedef struct {
        size_t flips;
        size_t candidate_bits;
        size_t largest_error;
} itty_feed_model_train_stats_t;

typedef enum {
        ITTY_FEED_MODEL_SELECTOR_PROTECTION_NONE,
        ITTY_FEED_MODEL_SELECTOR_PROTECTION_OWNER_LIFT,
        ITTY_FEED_MODEL_SELECTOR_PROTECTION_COMPETITOR_SUPPRESS,
} itty_feed_model_selector_protection_kind_t;

typedef struct {
        bool                                       accepted;
        itty_feed_model_selector_protection_kind_t kind;
        size_t                                     owner_route;
        size_t                                     competitor_route;
        size_t                                     owner_candidate_bits;
        size_t                                     competitor_candidate_bits;
        size_t                                     owner_safe_candidates;
        size_t                                     competitor_safe_candidates;
        size_t                                     owner_effective_candidates;
        size_t                                     competitor_effective_candidates;
        size_t                                     mixed_candidate_pairs;
        ptrdiff_t                                  margin_before;
        ptrdiff_t                                  margin_after;
        size_t                                     owner_distance_before;
        size_t                                     owner_distance_after;
        size_t                                     global_distance_before;
        size_t                                     global_distance_after;
        size_t                                     guard_route;
        size_t                                     guard_distance_before;
        size_t                                     guard_distance_after;
        size_t                                     guard_deficit_before;
        size_t                                     guard_deficit_after;
        size_t                                     flips;
} itty_feed_model_selector_protection_summary_t;

typedef struct {
        size_t layer_index;
        size_t desired_bits;
        size_t actual_bits;
        size_t mismatched_bits;
        size_t bit_count;
} itty_feed_model_backward_layer_diagnostic_t;

typedef struct {
        size_t node_index;
        size_t desired_bits;
        size_t actual_bits;
        size_t mismatched_bits;
        size_t bit_count;
} itty_feed_model_node_diagnostic_t;

typedef enum {
        ITTY_FEED_MODEL_SUFFIX_ORACLE_CANDIDATES_SYMBOLIC_TARGET,
        ITTY_FEED_MODEL_SUFFIX_ORACLE_CANDIDATES_RANDOM_ONE_BIT,
} itty_feed_model_suffix_oracle_candidate_source_t;

typedef struct {
        size_t                                           layer_index;
        size_t                                           node_index;
        size_t                                           max_candidate_bits;
        size_t                                           random_seed;
        itty_feed_model_backward_node_target_t           backward_node_target;
        itty_feed_model_suffix_oracle_candidate_source_t  candidate_source;
} itty_feed_model_suffix_oracle_options_t;

typedef struct {
        size_t candidate_bits;
        size_t helpful_bits;
        size_t harmful_bits;
        size_t neutral_bits;
        size_t blocker_helpful_bits;
        size_t true_neutral_bits;
        size_t neutral_same_folded_output_bits;
        size_t neutral_changed_folded_output_bits;
        size_t neutral_changed_selected_output_bits;
        size_t before_distance;
        size_t best_distance;
        size_t worst_distance;
} itty_feed_model_suffix_oracle_summary_t;

typedef struct {
        size_t distance;
        size_t false_positive_bits;
        size_t false_negative_bits;
        size_t false_negative_blocker_bits;
        size_t zero_veto_safety_bits;
        size_t selected_index;
} itty_feed_model_residual_decode_summary_t;

typedef struct {
        size_t selected_distance;
        size_t false_negative_count;
        size_t false_negative_blocker_bits;
        size_t zero_veto_safety_bits;
        size_t false_negative_vote_deficit;
        size_t false_negative_vote_deficit_min;
        size_t false_negative_vote_deficit_max;
        size_t false_positive_vote_excess;
        size_t target_one_margin;
        size_t target_zero_safety;
        size_t target_zero_safety_min;
        size_t false_negative_vote_deficit_histogram[ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS];
        size_t false_positive_vote_excess_histogram[ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS];
        size_t target_one_margin_histogram[ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS];
        size_t target_zero_safety_histogram[ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS];
        size_t nearest_wrong_margin;
        size_t selected_node;
        size_t selected_popcount;
        size_t best_decoded_node;
        size_t best_decoded_distance;
} itty_feed_model_decoder_objective_t;

typedef struct {
        size_t selected_by_popcount;
        size_t selected_popcount;
        size_t best_by_target_distance;
        size_t best_by_false_negative_deficit;
        size_t best_by_false_positive_excess;
        size_t popcount_gap;
        size_t selected_distance;
        size_t best_target_distance;
        size_t distance_gap_between_selected_and_best;
        size_t selected_false_negative_deficit;
        size_t best_false_negative_deficit;
        size_t selected_false_positive_excess;
        size_t best_false_positive_excess;
} itty_feed_model_segment_node_selection_summary_t;

typedef struct {
        size_t best_node;
        bool   best_complemented;
        size_t best_distance;
        size_t best_false_negative_deficit;
        size_t best_false_positive_excess;
        size_t best_target_zero_safety;
        size_t best_popcount;
        size_t best_normal_node;
        size_t best_normal_distance;
        size_t best_normal_false_negative_deficit;
        size_t best_complement_node;
        size_t best_complement_distance;
        size_t best_complement_false_negative_deficit;
} itty_feed_model_segment_node_polarity_summary_t;

typedef enum {
        ITTY_FEED_MODEL_OUTPUT_TRANSFORM_IDENTITY,
        ITTY_FEED_MODEL_OUTPUT_TRANSFORM_ODD_SWAP,
        ITTY_FEED_MODEL_OUTPUT_TRANSFORM_EVEN_SWAP,
        ITTY_FEED_MODEL_OUTPUT_TRANSFORM_HALF_SWAP,
} itty_feed_model_output_transform_t;

typedef struct {
        itty_feed_model_output_transform_t transform;
        size_t                             selected_distance;
        size_t                             false_negative_vote_deficit;
        size_t                             false_positive_vote_excess;
        size_t                             target_zero_safety;
        size_t                             selected_node;
        size_t                             selected_popcount;
        size_t                             best_decoded_node;
        size_t                             best_decoded_distance;
} itty_feed_model_segment_transform_summary_t;

typedef struct {
        size_t a_distance_before;
        size_t a_distance_after_b;
        size_t a_distance_after_repair;
        size_t a_false_positive_excess_before;
        size_t a_false_positive_excess_after_b;
        size_t a_false_positive_excess_after_repair;
        size_t b_distance_before;
        size_t b_distance_after_b;
        size_t b_distance_after_repair;
        size_t b_false_negative_deficit_before;
        size_t b_false_negative_deficit_after_b;
        size_t b_false_negative_deficit_after_repair;
        size_t b_steps;
        size_t a_repair_steps;
        size_t b_flips;
        size_t a_repair_flips;
        bool   a_restored;
        bool   b_preserved;
        bool   accepted;
} itty_feed_model_replay_transaction_summary_t;

typedef enum {
        ITTY_FEED_MODEL_RESTORE_REJECTION_NONE,
        ITTY_FEED_MODEL_RESTORE_REJECTION_NO_FALSE_POSITIVES,
        ITTY_FEED_MODEL_RESTORE_REJECTION_NO_CANDIDATE_REPAIRS,
        ITTY_FEED_MODEL_RESTORE_REJECTION_NO_CLEARABLE_SEGMENT_VOTES,
        ITTY_FEED_MODEL_RESTORE_REJECTION_NO_MASK_PROJECTION,
        ITTY_FEED_MODEL_RESTORE_REJECTION_NO_USEFUL_REPAIRS,
        ITTY_FEED_MODEL_RESTORE_REJECTION_NO_B_SAFE_REPAIRS,
        ITTY_FEED_MODEL_RESTORE_REJECTION_NO_FLIPS_ACCEPTED,
} itty_feed_model_restore_rejection_reason_t;

typedef enum {
        ITTY_FEED_MODEL_RESTORE_PROPAGATION_NONE,
        ITTY_FEED_MODEL_RESTORE_PROPAGATION_NO_MAJORITY_CROSSING,
        ITTY_FEED_MODEL_RESTORE_PROPAGATION_WRONG_POLARITY,
        ITTY_FEED_MODEL_RESTORE_PROPAGATION_DUPLICATE_CONDENSED_MAPPING,
        ITTY_FEED_MODEL_RESTORE_PROPAGATION_ROTATION_OR_EXPANSION_MISMATCH,
        ITTY_FEED_MODEL_RESTORE_PROPAGATION_SELECTED_NODE_MISMATCH,
        ITTY_FEED_MODEL_RESTORE_PROPAGATION_SEGMENT_CHANGED_DECODED_NOT_CROSSED,
        ITTY_FEED_MODEL_RESTORE_PROPAGATION_DECODED_CROSSED_OFFSET_ELSEWHERE,
} itty_feed_model_restore_propagation_failure_t;

#define ITTY_FEED_MODEL_RESTORE_TRACE_LIMIT 64
#define ITTY_FEED_MODEL_RESTORE_CLEAR_TRACE_LIMIT 128
#define ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_TRACE_LIMIT 16
#define ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_VOTE_LIMIT 8
#define ITTY_FEED_MODEL_FINISH_CLEAR_SET_TRACE_LIMIT 8

typedef struct {
        size_t decoded_bit;
        bool   target_bit;
        bool   current_decoded_bit;
        size_t current_ones;
        size_t threshold;
        size_t max_ones_for_zero;
        size_t excess;
        size_t segment_votes_currently_one;
        size_t clearable_segment_votes;
        size_t candidate_final_output_bits;
        size_t projected_condensed_bits;
        bool   direct_candidate_changed;
        size_t candidate_mask_flips;
        size_t candidate_segment_votes;
        size_t replay_safe_candidates;
        size_t min_final_layer_mask_flips_needed;
        size_t final_selected_node_before;
        size_t final_selected_node_after;
        size_t actual_final_condensed_ones_before;
        size_t actual_final_condensed_ones_after;
        size_t actual_final_segment_ones_before;
        size_t actual_final_segment_ones_after;
        size_t forced_node_distance_before;
        size_t forced_node_distance_after;
        size_t forced_node_false_positive_excess_before;
        size_t forced_node_false_positive_excess_after;
        bool   decoded_before;
        bool   decoded_after;
        itty_feed_model_restore_propagation_failure_t propagation_failure;
        bool   accepted;
        itty_feed_model_restore_rejection_reason_t rejection_reason;
} itty_feed_model_restore_failure_trace_t;

typedef struct {
        size_t decoded_bit;
        size_t segment_index;
        size_t final_output_bit;
        bool   raw_final_segment_before;
        bool   desired_final_segment_value;
        bool   raw_final_segment_after;
        size_t mapped_condensed_bit;
        bool   condensed_before;
        bool   desired_condensed_bit;
        bool   condensed_after;
        size_t mask_flip_count;
        size_t majority_ones_before;
        size_t majority_ones_after;
        size_t majority_threshold;
        bool   condensed_bit_changed;
        size_t decoded_ones_before;
        size_t decoded_ones_after;
        bool   decoded_bit_before;
        bool   decoded_bit_after;
        bool   cleared;
} itty_feed_model_restore_clear_vote_trace_t;

typedef struct {
        size_t a_distance_after_b;
        size_t a_false_positive_excess_after_b;
        size_t a_candidate_repair_count;
        size_t a_useful_repair_count;
        size_t a_target_zero_repair_count;
        size_t a_rejected_repair_count;
        itty_feed_model_restore_rejection_reason_t no_flip_reason;
        size_t trace_count;
        itty_feed_model_restore_failure_trace_t traces[ITTY_FEED_MODEL_RESTORE_TRACE_LIMIT];
        size_t clear_trace_count;
        itty_feed_model_restore_clear_vote_trace_t clear_traces[ITTY_FEED_MODEL_RESTORE_CLEAR_TRACE_LIMIT];
} itty_feed_model_restore_failure_summary_t;

typedef struct {
        size_t clear_vote_count;
        size_t segment_indices[ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_VOTE_LIMIT];
        size_t final_output_bits[ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_VOTE_LIMIT];
        size_t a_distance_after_restore;
        size_t a_false_positive_excess_after_restore;
        size_t b_distance_after_restore;
        size_t b_false_negative_deficit_after_restore;
        size_t b_false_positive_excess_after_restore;
        size_t total_flips;
        size_t selected_node_before_restore;
        size_t selected_node_after_restore;
        size_t selection_margin_after_restore;
        size_t mask_flip_count;
        size_t overlap_final_output_bits;
        size_t overlap_condensed_bits;
        size_t overlap_mask_flip_locations;
        size_t overlap_mask_flip_directions;
        bool   a_excess_reduced;
        bool   a_restored;
        bool   selected_node_preserved;
        bool   strict_preserved;
        bool   distance_preserved;
        bool   progress_preserved;
        bool   no_regression;
        bool   chosen_best;
} itty_feed_model_contender_clear_set_trace_t;

typedef struct {
        size_t decoded_bit;
        size_t a_distance_after_b;
        size_t a_false_positive_excess_after_b;
        size_t selected_node_before;
        size_t selected_node_after_first;
        size_t forced_node_distance_before;
        size_t forced_node_false_positive_excess_before;
        size_t forced_node_distance_after_first;
        size_t forced_node_false_positive_excess_after_first;
        size_t contender_node;
        size_t a_distance_after_contender;
        size_t a_false_positive_excess_after_contender;
        size_t b_distance_after_contender;
        size_t b_false_negative_deficit_after_contender;
        size_t total_flips;
        bool   forced_useful;
        bool   contender_useful;
        bool   contender_b_safe;
        bool   b_strict_preserved;
        bool   b_distance_preserved;
        bool   b_progress_preserved;
        bool   b_no_regression;
        bool   contender_accepted;
        size_t clear_set_trace_count;
        itty_feed_model_contender_clear_set_trace_t clear_set_traces[ITTY_FEED_MODEL_CONTENDER_CLEAR_SET_TRACE_LIMIT];
} itty_feed_model_contender_restore_summary_t;

#define ITTY_FEED_MODEL_FINISH_CANDIDATE_TRACE_LIMIT 64

typedef struct {
        size_t decoded_bit;
        size_t final_node;
        size_t final_output_bit;
        size_t finish_margin_required;
        size_t b_distance_before;
        size_t b_deficit_before;
        size_t b_target_one_margin_before;
        size_t b_distance_after_finish;
        size_t b_deficit_after_finish;
        size_t b_target_one_margin_after_finish;
        size_t a_distance_after_finish;
        size_t a_excess_after_finish;
        size_t a_deficit_after_finish;
        size_t a_selected_node_before;
        size_t a_selected_node_after_finish;
        size_t a_damaged_correct_zero_to_false_positive_bits;
        size_t a_damaged_correct_one_to_false_negative_bits;
        bool   family_one_finish;
        size_t family_one_damaged_bits;
        bool   restore_needed;
        size_t restore_contender_node;
        bool   restore_available;
        bool   restore_failed;
        size_t restore_a_candidate_repair_count;
        size_t restore_a_useful_repair_count;
        size_t restore_a_target_zero_repair_count;
        size_t restore_a_rejected_repair_count;
        itty_feed_model_restore_rejection_reason_t restore_a_no_flip_reason;
        size_t restore_audit_decoded_bit;
        size_t restore_audit_ones;
        size_t restore_audit_threshold;
        size_t restore_audit_max_ones_for_zero;
        size_t restore_audit_excess;
        size_t restore_audit_clearable_votes;
        size_t restore_audit_mask_flips;
        size_t damage_set_restore_selected_node_before;
        size_t damage_set_restore_selected_node_after;
        size_t finish_condensed_bit;
        size_t finish_mask_flip_count;
        size_t damage_set_restore_mask_flip_count;
        size_t overlap_final_output_bits;
        size_t overlap_condensed_bits;
        size_t overlap_mask_flip_locations;
        size_t overlap_mask_flip_directions;
        bool   damage_set_restore_available;
        bool   damage_set_restore_excess_reduced;
        bool   damage_set_restore_restored_a;
        bool   damage_set_restore_distance_preserved;
        bool   damage_set_restore_progress_preserved;
        size_t damage_set_restore_flips;
        size_t replacement_candidate_count;
        size_t replacement_a_safe_count;
        size_t replacement_distance_helpful_count;
        size_t replacement_progress_helpful_count;
        bool   replacement_available;
        bool   replacement_distance_preserved;
        bool   replacement_progress_preserved;
        size_t replacement_flips;
        size_t replacement_output_bit;
        size_t replacement_condensed_bit;
        size_t family1_clear_set_count;
        size_t family1_excess_reducing_clear_set_count;
        size_t family1_restoring_clear_set_count;
        size_t family1_restore_only_clear_set_count;
        size_t family1_restore_progress_clear_set_count;
        size_t family1_restore_distance_clear_set_count;
        size_t family1_distance_preserving_clear_set_count;
        size_t family1_progress_preserving_clear_set_count;
        size_t family1_best_clear_set_index;
        itty_feed_model_contender_clear_set_trace_t family1_clear_set_traces[ITTY_FEED_MODEL_FINISH_CLEAR_SET_TRACE_LIMIT];
        size_t a_distance_after_restore;
        size_t a_excess_after_restore;
        size_t b_distance_after_restore;
        size_t b_deficit_after_restore;
        size_t b_target_one_margin_after_restore;
        bool   finish_margin_met_before_restore;
        bool   finish_margin_met_after_restore;
        bool   strict_preserved;
        bool   distance_preserved;
        bool   progress_preserved;
        bool   no_regression;
        bool   frontier_improved;
        bool   rejected_no_pre_restore_gain;
        bool   rejected_a_not_restored;
        bool   rejected_b_gain_lost;
        bool   chosen_best;
} itty_feed_model_finish_candidate_trace_t;

typedef struct {
        size_t rounds_attempted;
        size_t rounds_accepted;
        size_t a_distance_before;
        size_t a_distance_after;
        size_t a_false_positive_excess_before;
        size_t a_false_positive_excess_after;
        size_t b_distance_before;
        size_t b_distance_after;
        size_t b_false_negative_deficit_before;
        size_t b_false_negative_deficit_after;
        size_t b_min_deficit_before;
        size_t b_min_deficit_after;
        size_t b_deficit_one_bits_before;
        size_t b_deficit_one_bits_after;
        size_t b_deficit_two_bits_before;
        size_t b_deficit_two_bits_after;
        size_t b_cheapest_completion_cost_before;
        size_t b_cheapest_completion_cost_after;
        size_t b_top_k_completion_cost_before;
        size_t b_top_k_completion_cost_after;
        size_t total_b_flips;
        size_t total_restore_flips;
        size_t strict_preserved_rounds;
        size_t distance_preserved_rounds;
        size_t progress_preserved_rounds;
        size_t no_regression_rounds;
        size_t frontier_improved_rounds;
        size_t finish_candidates;
        size_t finish_margin_required;
        size_t finish_pre_restore_distance_helpful;
        size_t finish_pre_restore_deficit_helpful;
        size_t finish_pre_restore_margin_met;
        size_t finish_clobbers_a;
        size_t finish_restores_a;
        size_t finish_post_restore_distance_preserved;
        size_t finish_post_restore_progress_preserved;
        size_t finish_post_restore_margin_preserved;
        size_t finish_rejected_no_pre_restore_gain;
        size_t finish_rejected_a_not_restored;
        size_t finish_rejected_b_gain_lost;
        size_t finish_complete_b_before_restore;
        size_t finish_create_a_correct_zero_to_false_positive;
        size_t finish_create_a_correct_one_to_false_negative;
        size_t finish_switch_a_selected_node;
        size_t finish_contender_restore_available;
        size_t finish_restore_erases_b;
        size_t finish_restore_fails;
        bool   b_distance_improved;
        bool   b_deficit_improved;
        bool   b_frontier_improved;
        bool   a_remains_solved;
        size_t finish_trace_count;
        itty_feed_model_finish_candidate_trace_t finish_traces[ITTY_FEED_MODEL_FINISH_CANDIDATE_TRACE_LIMIT];
} itty_feed_model_transaction_scaffold_summary_t;

#define ITTY_FEED_MODEL_TRANSACTION_SCAFFOLD_TRAJECTORY_LIMIT 16

typedef struct {
        size_t round_index;
        bool   accepted;
        bool   used_restore;
        bool   used_finish_nearest_bit;
        size_t finish_nearest_threshold;
        size_t a_distance_before;
        size_t a_false_positive_excess_before;
        size_t a_distance_after;
        size_t a_false_positive_excess_after;
        size_t b_distance_before;
        size_t b_false_negative_deficit_before;
        size_t b_distance_after;
        size_t b_false_negative_deficit_after;
        size_t b_min_deficit_before;
        size_t b_min_deficit_after;
        size_t b_deficit_one_bits_before;
        size_t b_deficit_one_bits_after;
        size_t b_deficit_two_bits_before;
        size_t b_deficit_two_bits_after;
        size_t b_cheapest_completion_cost_before;
        size_t b_cheapest_completion_cost_after;
        size_t b_top_k_completion_cost_before;
        size_t b_top_k_completion_cost_after;
        size_t b_flips;
        size_t restore_flips;
        bool   strict_preserved;
        bool   distance_preserved;
        bool   progress_preserved;
        bool   no_regression;
        bool   frontier_improved;
} itty_feed_model_transaction_scaffold_round_t;

typedef struct {
        itty_bit_string_list_t *input;
        itty_bit_string_t      *target;
} itty_feed_model_replay_example_t;

typedef struct {
        size_t correct_zero_to_false_positive_bits;
        size_t correct_one_to_false_negative_bits;
        size_t false_positive_to_correct_zero_bits;
        size_t false_negative_to_correct_one_bits;
        size_t false_positive_to_false_negative_bits;
        size_t false_negative_to_false_positive_bits;
        size_t unchanged_wrong_bits;
        size_t unchanged_correct_bits;
} itty_feed_model_replay_transition_matrix_t;

#define ITTY_FEED_MODEL_SAFE_QUOTA_EFFECT_TRACE_LIMIT 64
#define ITTY_FEED_MODEL_FINAL_SURFACE_FEASIBILITY_TRACE_LIMIT 64

typedef enum {
        ITTY_FEED_MODEL_FINAL_SURFACE_BLOCK_NONE,
        ITTY_FEED_MODEL_FINAL_SURFACE_BLOCK_REPLAY_UNSAFE,
        ITTY_FEED_MODEL_FINAL_SURFACE_BLOCK_DUPLICATE_FINAL_VOTE,
        ITTY_FEED_MODEL_FINAL_SURFACE_BLOCK_NO_CANDIDATE_PATH,
} itty_feed_model_final_surface_block_reason_t;

typedef struct {
        size_t                                           decoded_bit;
        size_t                                           needed_final_votes;
        size_t                                           safe_final_votes_available;
        size_t                                           unsafe_final_votes_available;
        size_t                                           safe_final_votes_shortfall;
        size_t                                           duplicate_safe_final_votes;
        size_t                                           duplicate_unsafe_final_votes;
        itty_feed_model_final_surface_block_reason_t     blocked_reason;
} itty_feed_model_final_surface_feasibility_t;

typedef struct {
        size_t decoded_bit;
        size_t penultimate_selected_node_before;
        size_t penultimate_selected_node_after;
        size_t target_bit;
        size_t penultimate_ones_before;
        size_t penultimate_ones_after;
        size_t penultimate_threshold;
        size_t needed_before;
        size_t safe_votes_available;
        size_t safe_votes_selected;
        size_t penultimate_votes_changed;
        size_t final_selected_node_before;
        size_t final_selected_node_after;
        size_t final_condensed_ones_before;
        size_t final_condensed_ones_after;
        size_t final_condensed_threshold;
        size_t final_segment_ones_before;
        size_t final_segment_ones_after;
        size_t final_segment_threshold;
        size_t final_distance_contribution_before;
        size_t final_distance_contribution_after;
        bool   quota_realized;
        bool   penultimate_threshold_crossed;
        bool   penultimate_decoded_before;
        bool   penultimate_decoded_after;
        bool   final_condensed_changed;
        bool   final_condensed_crossed;
        bool   final_segment_changed;
        bool   final_decoded_before;
        bool   final_decoded_after;
        bool   final_decode_crossed;
} itty_feed_model_safe_quota_effect_t;

typedef struct {
        size_t max_projected_blocks;
        size_t max_layer_flips;
        size_t max_blocks_per_final_node;
        size_t max_strict_distance_blocks;
        size_t max_blocker_blocks;
        bool   limit_strict_distance_blocks;
        bool   limit_blocker_blocks;
        bool   use_or_residual_repairs;
        bool   prefer_blocker_efficiency;
        bool   require_residual_blocker_efficiency;
        size_t top_k_segment_vote_alternatives;
        size_t replay_zero_protection_penalty;
        size_t replay_one_protection_penalty;
        size_t replay_taboo_flip_penalty;
        itty_feed_model_replay_example_t const *replay_examples;
        size_t                                  replay_example_count;
        bool                                    strict_replay_guard;
        bool                                    strict_replay_taboo_rejection;
        bool                                    replay_safe_quota_complete_only;
        bool                                    reserve_replay_protected_zero_votes;
} itty_feed_model_projected_repair_options_t;

typedef struct {
        size_t projected_blocks;
        size_t accepted_blocks;
        size_t fully_realized_blocks;
        size_t partially_realized_blocks;
        size_t unrealized_blocks;
        size_t condensed_realistic_blocks;
        size_t condensed_realistic_strict_distance_helpful_blocks;
        size_t condensed_realistic_blocker_helpful_blocks;
        size_t condensed_realistic_objective_helpful_blocks;
        size_t condensed_realistic_harmful_blocks;
        size_t condensed_realistic_neutral_blocks;
        size_t accepted_strict_distance_blocks;
        size_t accepted_blocker_blocks;
        size_t strict_distance_layer_flips;
        size_t blocker_layer_flips;
        size_t residual_enable_flips;
        size_t residual_mask_flips;
        size_t residual_active_bits;
        size_t residual_candidate_blocks;
        size_t residual_accepted_blocks;
        size_t previous_layer_projected_blocks;
        size_t previous_layer_strict_distance_helpful_blocks;
        size_t previous_layer_blocker_helpful_blocks;
        size_t previous_layer_objective_helpful_blocks;
        size_t previous_layer_harmful_blocks;
        size_t previous_layer_neutral_blocks;
        size_t previous_layer_harmful_distance_blocks;
        size_t previous_layer_harmful_blocker_blocks;
        size_t previous_layer_harmful_margin_blocks;
        size_t previous_layer_harmful_safety_blocks;
        size_t previous_layer_harm_correct_zero_to_false_positive_bits;
        size_t previous_layer_harm_correct_one_to_false_negative_bits;
        size_t previous_layer_harm_false_positive_to_correct_zero_bits;
        size_t previous_layer_harm_false_negative_to_correct_one_bits;
        size_t previous_layer_harm_false_positive_to_false_negative_bits;
        size_t previous_layer_harm_false_negative_to_false_positive_bits;
        size_t previous_layer_harm_unchanged_wrong_bits;
        size_t previous_layer_harm_unchanged_correct_bits;
        size_t previous_layer_pinned_projected_blocks;
        size_t previous_layer_pinned_strict_distance_helpful_blocks;
        size_t previous_layer_pinned_blocker_helpful_blocks;
        size_t previous_layer_pinned_objective_helpful_blocks;
        size_t previous_layer_pinned_harmful_blocks;
        size_t previous_layer_pinned_neutral_blocks;
        size_t conflicts;
        size_t estimated_layer_flips;
        size_t direct_quota_vote_flips;
        size_t majority_threshold_support_flips;
        size_t conflict_resolution_flips;
        size_t collateral_preservation_flips;
        size_t target_zero_safety_preservation_flips;
        size_t selection_preservation_flips;
        size_t quota_vote_support_cost_histogram[ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS];
        size_t decoded_bits_fixed;
        size_t fixed_decoded_bit_flips_total;
        size_t fixed_decoded_bit_flips_min;
        size_t fixed_decoded_bit_flips_max;
        size_t quota_size_total;
        size_t quota_size_max;
        size_t requested_condensed_bits;
        size_t realized_condensed_bits;
        size_t already_satisfied_condensed_bits;
        size_t condensed_bits_needing_flips;
        size_t available_flippable_votes;
        size_t requested_output_bits;
        size_t realized_output_bits;
        size_t lost_output_bits;
        size_t extra_output_bits;
        size_t structural_extra_output_bits;
        size_t collateral_extra_output_bits;
        size_t no_effect_candidates;
        size_t no_effect_candidate_already_satisfied;
        size_t no_effect_candidate_no_majority_crossing;
        size_t no_effect_candidate_unselected_node;
        size_t no_effect_candidate_irrelevant_segment;
        size_t no_effect_candidate_vote_tied;
        size_t replay_safe_candidates;
        size_t replay_safe_strict_distance_candidates;
        size_t replay_safe_deficit_candidates;
        size_t replay_safe_frontier_candidates;
        size_t replay_safe_vote_movement_candidates;
        size_t replay_safe_noop_candidates;
        size_t replay_safe_irrelevant_candidates;
        size_t current_candidate_net_positive;
        size_t current_candidate_net_zero;
        size_t current_candidate_net_negative;
        size_t replay_safe_quota_feasible_decoded_bits;
        size_t replay_safe_quota_selected_decoded_bits;
        size_t replay_safe_quota_accepted_decoded_bits;
        size_t replay_safe_quota_completed_decoded_bits;
        size_t replay_safe_quota_local_realized_decoded_bits;
        size_t replay_safe_quota_local_crossed_decoded_bits;
        size_t replay_safe_quota_net_positive_decoded_bits;
        size_t replay_safe_quota_cancelled_decoded_bits;
        size_t replay_safe_quota_final_vote_reached;
        size_t replay_safe_quota_final_vote_changed;
        size_t replay_safe_quota_final_vote_crossed;
        size_t replay_safe_quota_decoded_vote_changed;
        size_t replay_safe_quota_decoded_threshold_crossed;
        size_t replay_safe_quota_lost_at_final_majority;
        size_t replay_safe_quota_lost_at_final_rotation_or_expansion;
        size_t replay_safe_quota_lost_at_unselected_final_node;
        size_t replay_safe_quota_lost_at_segment_vote_margin;
        size_t replay_safe_quota_lost_due_to_duplicate_final_segment;
        size_t replay_safe_quota_local_votes_selected;
        size_t replay_safe_quota_unique_local_votes_changed;
        size_t replay_safe_quota_unique_final_condensed_bits_affected;
        size_t replay_safe_quota_unique_final_segment_votes_affected;
        size_t replay_safe_quota_unique_decoded_bits_affected;
        size_t replay_safe_quota_distance_flip_decoded_bits;
        size_t replay_safe_quota_rejected_local_only_decoded_bits;
        size_t replay_safe_quota_blocked_decoded_bits;
        size_t replay_final_surface_feasibility_trace_count;
        itty_feed_model_final_surface_feasibility_t replay_final_surface_feasibility_traces[ITTY_FEED_MODEL_FINAL_SURFACE_FEASIBILITY_TRACE_LIMIT];
        size_t replay_safe_quota_effect_trace_count;
        itty_feed_model_safe_quota_effect_t replay_safe_quota_effect_traces[ITTY_FEED_MODEL_SAFE_QUOTA_EFFECT_TRACE_LIMIT];
        size_t replay_safe_quota_complete_candidates;
        size_t replay_safe_quota_incomplete_candidates;
        size_t replay_direct_protected_zero_hit_candidates;
        size_t replay_reserved_zero_votes;
        size_t replay_realization_collateral_false_positive_candidates;
        size_t replay_sensitive_mask_flips;
        size_t replay_safe_mask_flips;
        size_t replay_false_positive_mask_flips;
        size_t replay_false_negative_mask_flips;
        size_t replay_margin_or_safety_weakening_mask_flips;
        size_t replay_collateral_cost;
        size_t replay_decomposed_candidates;
        size_t replay_decomposed_mask_flips;
        size_t replay_decomposed_unsafe_mask_flips;
        size_t replay_one_bad_flip_candidates;
        size_t replay_mostly_unsafe_candidates;
        size_t replay_alternate_mask_flips;
        size_t replay_alternate_unsafe_mask_flips;
        size_t replay_alternate_collateral_cost;
        size_t replay_alternate_better_candidates;
        size_t replay_bad_flip_unique;
        size_t replay_bad_flip_top_frequency;
        size_t replay_bad_flip_top_harmless_uses;
        size_t replay_bad_flip_top_damaged_bits;
        size_t replay_bad_flip_top_helped_decoded_bit;
        size_t replay_bad_flip_top_layer;
        size_t replay_bad_flip_top_node;
        size_t replay_bad_flip_top_input;
        size_t replay_bad_flip_top_bit;
        bool   replay_bad_flip_top_value;
        size_t replay_taboo_vote_candidates;
        size_t replay_taboo_mask_flips;
        size_t replay_taboo_penalty_total;
        size_t replay_taboo_rejected_vote_candidates;
        size_t replay_minus_one_bad_candidates;
        size_t replay_minus_one_bad_safe_candidates;
        size_t replay_minus_one_bad_deficit_candidates;
        size_t replay_minus_one_bad_strict_candidates;
        size_t replay_unsafe_candidates;
        size_t replay_best_candidate_unsafe;
        size_t replay_unsafe_distance_regressions;
        size_t replay_unsafe_false_positive_excess_regressions;
        size_t replay_unsafe_target_one_margin_regressions;
        size_t replay_unsafe_target_zero_safety_regressions;
        size_t replay_unsafe_selected_node_switches;
        size_t replay_unsafe_best_decoded_node_switches;
        itty_feed_model_replay_transition_matrix_t replay_unsafe_transitions;
        size_t layer_flips;
        size_t before_distance;
        size_t after_distance;
        size_t before_blockers;
        size_t after_blockers;
} itty_feed_model_projected_repair_stats_t;

typedef enum {
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_CONVERGED,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_BATCH,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_DISTANCE_REGRESSION,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_EXCESS_REGRESSION,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_ZERO_SAFETY_REGRESSION,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_TARGET_ONE_MARGIN_LOSS,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_NO_OBJECTIVE_DELTA,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_TRUE_NOOP,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_LOCAL_ACTIVATION_ONLY,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_VOTE_MOVEMENT_TIED,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_FUTURE_COST_IMPROVEMENT,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_REALIZATION_MISMATCH,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_BATCH_INTERACTION,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_REPLAY_GUARD,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_BLOCKED_ALL_CANDIDATES,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_SAFE_NO_CURRENT_GAIN,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_PREFIX_ORDERING_FAILURE,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_SINGLE_CANDIDATE_CONFLICT,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_CAPACITY_CONFLICT,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_MAX_ROUNDS,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_REPAIRS,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_EFFECTIVE_CANDIDATES,
        ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_PROGRESS,
} itty_feed_model_segment_train_stop_reason_t;

typedef struct {
        size_t round_index;
        bool   accepted;
        bool   reverted;
        itty_feed_model_segment_train_stop_reason_t rejection_reason;
        size_t selected_node;
        size_t selected_popcount;
        size_t popcount_gap;
        size_t selected_node_after;
        size_t best_decoded_node;
        size_t best_decoded_distance;
        size_t best_decoded_node_after;
        size_t best_decoded_distance_after;
        bool   selected_is_best_decoded;
        size_t before_distance;
        size_t after_distance;
        size_t distance_delta;
        ptrdiff_t estimated_distance_delta;
        ptrdiff_t actual_distance_delta;
        size_t before_blockers;
        size_t after_blockers;
        size_t blocker_delta;
        size_t before_false_negative_deficit;
        size_t after_false_negative_deficit;
        ptrdiff_t estimated_false_negative_deficit_delta;
        ptrdiff_t actual_false_negative_deficit_delta;
        size_t before_false_positive_excess;
        size_t after_false_positive_excess;
        ptrdiff_t estimated_false_positive_excess_delta;
        ptrdiff_t actual_false_positive_excess_delta;
        size_t before_target_one_margin;
        size_t after_target_one_margin;
        ptrdiff_t estimated_target_one_margin_delta;
        ptrdiff_t actual_target_one_margin_delta;
        size_t before_target_zero_safety;
        size_t after_target_zero_safety;
        ptrdiff_t estimated_target_zero_safety_delta;
        ptrdiff_t actual_target_zero_safety_delta;
        bool selected_activation_changed;
        bool segment_votes_changed;
        bool future_cost_improved;
        size_t candidate_blocks_proposed;
        size_t candidate_blocks_selected;
        size_t requested_constraints;
        size_t realized_constraints;
        size_t collateral_changed_bits;
        size_t replay_examples_scored;
        size_t replay_solved_before;
        size_t replay_unsolved_after;
        ptrdiff_t replay_distance_delta;
        ptrdiff_t replay_false_negative_deficit_delta;
        ptrdiff_t replay_false_positive_excess_delta;
        ptrdiff_t replay_target_one_margin_delta;
        ptrdiff_t replay_target_zero_safety_delta;
        itty_feed_model_replay_transition_matrix_t replay_transitions;
        itty_feed_model_replay_transition_matrix_t current_transitions;
        size_t replay_selected_node_switches;
        size_t replay_best_decoded_node_switches;
        size_t accepted_blocks;
        size_t layer_flips;
} itty_feed_model_refreshed_projected_repair_round_t;

typedef struct {
        size_t                                                 batch_size;
        size_t                                                 max_rounds;
        size_t                                                 max_layer_flips_per_batch;
        size_t                                                 max_blocks_per_final_node;
        bool                                                   use_or_residual_repairs;
        bool                                                   prefer_blocker_efficiency;
        bool                                                   require_residual_blocker_efficiency;
        size_t                                                 top_k_segment_vote_alternatives;
        size_t                                                 replay_zero_protection_penalty;
        size_t                                                 replay_one_protection_penalty;
        size_t                                                 replay_taboo_flip_penalty;
        itty_feed_model_replay_example_t const                *replay_examples;
        size_t                                                 replay_example_count;
        bool                                                   strict_replay_guard;
        bool                                                   strict_replay_taboo_rejection;
        bool                                                   replay_safe_quota_complete_only;
        bool                                                   reserve_replay_protected_zero_votes;
        itty_feed_model_refreshed_projected_repair_round_t    *trajectory;
        size_t                                                 trajectory_count;
} itty_feed_model_refreshed_projected_repair_options_t;

typedef struct {
        size_t                                   rounds;
        size_t                                   accepted_rounds;
        size_t                                   rejected_rounds;
        size_t                                   before_distance;
        size_t                                   after_distance;
        size_t                                   before_blockers;
        size_t                                   after_blockers;
        itty_feed_model_segment_train_stop_reason_t rejection_reason;
        itty_feed_model_refreshed_projected_repair_round_t rejected_round;
        size_t replay_rejected_batches;
        size_t replay_bisections;
        size_t replay_examples_scored;
        itty_feed_model_replay_transition_matrix_t replay_transitions;
        itty_feed_model_replay_transition_matrix_t current_transitions;
        size_t current_helpful_decoded_bits;
        size_t current_harmed_decoded_bits;
        size_t current_neutral_decoded_bits;
        size_t current_candidate_net_positive;
        size_t current_candidate_net_zero;
        size_t current_candidate_net_negative;
        size_t current_batch_cancel_target_one_loss;
        size_t current_batch_cancel_target_zero_loss;
        size_t current_batch_cancel_selected_node_change;
        size_t current_batch_cancel_duplicate_or_overlap_side_effect;
        itty_feed_model_projected_repair_stats_t projected;
} itty_feed_model_refreshed_projected_repair_stats_t;

typedef struct {
        size_t initial_distance;
        size_t final_distance;
        itty_feed_model_segment_train_stop_reason_t stop_reason;
        size_t layer6_flips;
        size_t rounds;
        size_t accepted_blocks;
        size_t quota_size_total;
        size_t quota_size_max;
        double average_quota_size;
        size_t false_negative_vote_deficit_before;
        size_t false_negative_vote_deficit_after;
        size_t false_positive_vote_excess_before;
        size_t false_positive_vote_excess_after;
        size_t target_zero_safety_minimum;
        double mask_entropy_before;
        double mask_entropy_after;
        double quota_completion_efficiency;
        double vote_efficiency;
        size_t direct_quota_vote_flips;
        size_t majority_threshold_support_flips;
        size_t conflict_resolution_flips;
        size_t collateral_preservation_flips;
        size_t target_zero_safety_preservation_flips;
        size_t selection_preservation_flips;
        size_t quota_vote_support_cost_histogram[ITTY_FEED_MODEL_DECODER_HISTOGRAM_BUCKETS];
        size_t decoded_bits_fixed;
        double average_flips_per_fixed_decoded_bit;
        size_t min_flips_per_fixed_decoded_bit;
        size_t max_flips_per_fixed_decoded_bit;
        double average_final_target_one_margin;
        itty_feed_model_refreshed_projected_repair_stats_t training;
} itty_feed_model_segment_training_summary_t;

itty_feed_model_t *itty_feed_model_new (size_t number_of_layers,
                                        size_t nodes_per_layer,
                                        size_t inputs_per_node,
                                        size_t vocabulary_words);
void itty_feed_model_free (itty_feed_model_t *model);
void itty_feed_model_set_layer_rotation (itty_feed_model_t *model,
                                         size_t             layer_index,
                                         size_t             rotation);
void itty_feed_model_set_decoder (itty_feed_model_t        *model,
                                  itty_feed_model_decoder_t decoder);
bool itty_feed_model_randomize_masks (itty_feed_model_t *model,
                                      size_t             seed,
                                      size_t             numerator,
                                      size_t             denominator);
bool itty_feed_model_measure_masks (itty_feed_model_t                 *model,
                                    itty_model_metrics_bit_summary_t  *summary);
bool itty_feed_model_measure_layer_masks (itty_feed_model_t                 *model,
                                          size_t                            layer_index,
                                          itty_model_metrics_bit_summary_t  *summary);
bool itty_feed_model_measure_backward_layer_diagnostics (itty_feed_model_t                                  *model,
                                                         itty_bit_string_list_t                             *input,
                                                         itty_bit_string_t                                  *target,
                                                         itty_feed_model_train_options_t const              *options,
                                                         itty_feed_model_backward_layer_diagnostic_t        *diagnostics,
                                                         size_t                                              diagnostic_count);
bool itty_feed_model_measure_final_layer_node_diagnostics (itty_feed_model_t                     *model,
                                                           itty_bit_string_list_t                *input,
                                                           itty_bit_string_t                     *target,
                                                           itty_feed_model_node_diagnostic_t     *diagnostics,
                                                           size_t                                 diagnostic_count);
bool itty_feed_model_measure_suffix_oracle (itty_feed_model_t                              *model,
                                            itty_bit_string_list_t                         *input,
                                            itty_bit_string_t                              *target,
                                            itty_feed_model_suffix_oracle_options_t const   *options,
                                            itty_feed_model_suffix_oracle_summary_t         *summary);
bool itty_feed_model_measure_residual_decode (itty_feed_model_t                         *model,
                                              itty_bit_string_list_t                    *input,
                                              itty_bit_string_t                         *target,
                                              itty_feed_model_residual_decode_summary_t *summary);
bool itty_feed_model_measure_decoder_objective (itty_feed_model_t                    *model,
                                                itty_bit_string_list_t               *input,
                                                itty_bit_string_t                    *target,
                                                itty_feed_model_decoder_objective_t  *objective);
bool itty_feed_model_measure_decoder_objective_with_lane_split (itty_feed_model_t                    *model,
                                                                itty_bit_string_list_t               *input,
                                                                itty_bit_string_t                    *target,
                                                                size_t                                selector_lane_bit_offset,
                                                                size_t                                selector_lane_bit_count,
                                                                size_t                                decoder_lane_bit_offset,
                                                                size_t                                decoder_lane_bit_count,
                                                                itty_feed_model_decoder_objective_t  *objective);
bool itty_feed_model_measure_decoder_objective_for_node (itty_feed_model_t                    *model,
                                                         itty_bit_string_list_t               *input,
                                                         itty_bit_string_t                    *target,
                                                         size_t                                selected_node,
                                                         itty_feed_model_decoder_objective_t  *objective);
bool itty_feed_model_measure_decoder_objective_for_node_with_lane_split (itty_feed_model_t                    *model,
                                                                         itty_bit_string_list_t               *input,
                                                                         itty_bit_string_t                    *target,
                                                                         size_t                                selected_node,
                                                                         size_t                                selector_lane_bit_offset,
                                                                         size_t                                selector_lane_bit_count,
                                                                         size_t                                decoder_lane_bit_offset,
                                                                         size_t                                decoder_lane_bit_count,
                                                                         itty_feed_model_decoder_objective_t  *objective);
bool itty_feed_model_measure_segment_node_selection (itty_feed_model_t                                *model,
                                                     itty_bit_string_list_t                           *input,
                                                     itty_bit_string_t                                *target,
                                                     itty_feed_model_segment_node_selection_summary_t *summary);
bool itty_feed_model_measure_segment_node_selection_with_lane_split (itty_feed_model_t                                *model,
                                                                     itty_bit_string_list_t                           *input,
                                                                     itty_bit_string_t                                *target,
                                                                     size_t                                            selector_lane_bit_offset,
                                                                     size_t                                            selector_lane_bit_count,
                                                                     size_t                                            decoder_lane_bit_offset,
                                                                     size_t                                            decoder_lane_bit_count,
                                                                     itty_feed_model_segment_node_selection_summary_t *summary);
bool itty_feed_model_measure_segment_node_polarity_selection (itty_feed_model_t                                         *model,
                                                              itty_bit_string_list_t                                    *input,
                                                              itty_bit_string_t                                         *target,
                                                              itty_feed_model_segment_node_polarity_summary_t          *summary);
bool itty_feed_model_measure_segment_transform (itty_feed_model_t                               *model,
                                                itty_bit_string_list_t                          *input,
                                                itty_bit_string_t                               *target,
                                                itty_feed_model_output_transform_t              transform,
                                                itty_feed_model_segment_transform_summary_t    *summary);
bool itty_feed_model_measure_final_layer_replay_transaction (itty_feed_model_t                            *model,
                                                             itty_bit_string_list_t                       *first_input,
                                                             itty_bit_string_t                            *first_target,
                                                             itty_bit_string_list_t                       *second_input,
                                                             itty_bit_string_t                            *second_target,
                                                             itty_feed_model_train_options_t const       *options,
                                                             size_t                                        max_second_steps,
                                                             size_t                                        max_first_repair_steps,
                                                             itty_feed_model_replay_transaction_summary_t *summary);
bool itty_feed_model_measure_final_layer_restore_failure (itty_feed_model_t                               *model,
                                                          itty_bit_string_list_t                          *first_input,
                                                          itty_bit_string_t                               *first_target,
                                                          itty_bit_string_list_t                          *second_input,
                                                          itty_bit_string_t                               *second_target,
                                                          itty_feed_model_train_options_t const          *options,
                                                          size_t                                           max_second_steps,
                                                          bool                                             preserve_second,
                                                          itty_feed_model_restore_failure_summary_t      *summary);
bool itty_feed_model_measure_final_layer_contender_restore (itty_feed_model_t                                 *model,
                                                            itty_bit_string_list_t                            *first_input,
                                                            itty_bit_string_t                                 *first_target,
                                                            itty_bit_string_list_t                            *second_input,
                                                            itty_bit_string_t                                 *second_target,
                                                            itty_feed_model_train_options_t const            *options,
                                                            size_t                                             max_second_steps,
                                                            bool                                               preserve_second,
                                                            itty_feed_model_contender_restore_summary_t      *summary);
bool itty_feed_model_train_final_layer_transaction_scaffold (itty_feed_model_t                                *model,
                                                             itty_bit_string_list_t                           *first_input,
                                                             itty_bit_string_t                                *first_target,
                                                             itty_bit_string_list_t                           *second_input,
                                                             itty_bit_string_t                                *second_target,
                                                             itty_feed_model_train_options_t const           *options,
                                                             size_t                                            max_rounds,
                                                             itty_feed_model_transaction_scaffold_round_t    *trajectory,
                                                             size_t                                            trajectory_count,
                                                             itty_feed_model_transaction_scaffold_summary_t  *summary);
bool itty_feed_model_train_final_layer_with_suffix_oracle_for_node (itty_feed_model_t                     *model,
                                                                    itty_bit_string_list_t                *input,
                                                                    itty_bit_string_t                     *target,
                                                                    size_t                                 node_index,
                                                                    itty_feed_model_train_options_t const *options,
                                                                    itty_feed_model_train_stats_t         *stats);
bool itty_feed_model_train_final_layer_selector_protection_for_node (itty_feed_model_t                                  *model,
                                                                     itty_bit_string_list_t                             *input,
                                                                     itty_bit_string_t                                  *target,
                                                                     size_t                                              owner_route,
                                                                     itty_feed_model_train_options_t const              *options,
                                                                     itty_feed_model_selector_protection_summary_t      *summary);
bool itty_feed_model_train_final_layer_selector_protection_for_node_with_guard (itty_feed_model_t                                  *model,
                                                                                itty_bit_string_list_t                             *input,
                                                                                itty_bit_string_t                                  *target,
                                                                                size_t                                              owner_route,
                                                                                itty_bit_string_list_t                             *guard_input,
                                                                                itty_bit_string_t                                  *guard_target,
                                                                                size_t                                              guard_route,
                                                                                itty_feed_model_train_options_t const              *options,
                                                                                itty_feed_model_selector_protection_summary_t      *summary);
bool itty_feed_model_measure_final_layer_selector_protection_for_node_with_guard (itty_feed_model_t                                  *model,
                                                                                  itty_bit_string_list_t                             *input,
                                                                                  itty_bit_string_t                                  *target,
                                                                                  size_t                                              owner_route,
                                                                                  itty_bit_string_list_t                             *guard_input,
                                                                                  itty_bit_string_t                                  *guard_target,
                                                                                  size_t                                              guard_route,
                                                                                  itty_feed_model_train_options_t const              *options,
                                                                                  itty_feed_model_selector_protection_summary_t      *summary);
bool itty_feed_model_train_final_layer_selector_margin_for_node (itty_feed_model_t                                  *model,
                                                                 itty_bit_string_list_t                             *input,
                                                                 itty_bit_string_t                                  *target,
                                                                 size_t                                              owner_route,
                                                                 itty_feed_model_train_options_t const              *options,
                                                                 itty_feed_model_selector_protection_summary_t      *summary);
bool itty_feed_model_measure_final_layer_selector_margin_for_node (itty_feed_model_t                                  *model,
                                                                   itty_bit_string_list_t                             *input,
                                                                   itty_bit_string_t                                  *target,
                                                                   size_t                                              owner_route,
                                                                   itty_feed_model_train_options_t const              *options,
                                                                   itty_feed_model_selector_protection_summary_t      *summary);
itty_feed_model_layer_state_snapshot_t *itty_feed_model_snapshot_final_layer_state (itty_feed_model_t *model);
void itty_feed_model_restore_final_layer_state_snapshot (itty_feed_model_t                      *model,
                                                         itty_feed_model_layer_state_snapshot_t *snapshot);
void itty_feed_model_free_final_layer_state_snapshot (itty_feed_model_t                      *model,
                                                      itty_feed_model_layer_state_snapshot_t *snapshot);

itty_network_t *itty_feed_model_build_network (itty_feed_model_t *model);

bool itty_feed_model_train_one (itty_feed_model_t      *model,
                                itty_bit_string_list_t *input,
                                itty_bit_string_t      *target);
bool itty_feed_model_train_one_with_options (itty_feed_model_t                     *model,
                                             itty_bit_string_list_t                *input,
                                             itty_bit_string_t                     *target,
                                             itty_feed_model_train_options_t const *options);
bool itty_feed_model_train_one_with_stats (itty_feed_model_t                     *model,
                                           itty_bit_string_list_t                *input,
                                           itty_bit_string_t                     *target,
                                           itty_feed_model_train_options_t const *options,
                                           itty_feed_model_train_stats_t         *stats);
bool itty_feed_model_train_backwards_one (itty_feed_model_t      *model,
                                          itty_bit_string_list_t *input,
                                          itty_bit_string_t      *target);
bool itty_feed_model_train_backwards_one_with_options (itty_feed_model_t                     *model,
                                                       itty_bit_string_list_t                *input,
                                                       itty_bit_string_t                     *target,
                                                       itty_feed_model_train_options_t const *options);
bool itty_feed_model_train_backwards_one_with_stats (itty_feed_model_t                     *model,
                                                     itty_bit_string_list_t                *input,
                                                     itty_bit_string_t                     *target,
                                                     itty_feed_model_train_options_t const *options,
                                                     itty_feed_model_train_stats_t         *stats);
bool itty_feed_model_train_backwards_one_with_layer_stats (itty_feed_model_t                     *model,
                                                           itty_bit_string_list_t                *input,
                                                           itty_bit_string_t                     *target,
                                                           itty_feed_model_train_options_t const *options,
                                                           itty_feed_model_train_stats_t         *stats,
                                                           itty_feed_model_train_stats_t         *layer_stats,
                                                           size_t                                 layer_stats_count);
bool itty_feed_model_train_final_layer_with_suffix_oracle (itty_feed_model_t                     *model,
                                                           itty_bit_string_list_t                *input,
                                                           itty_bit_string_t                     *target,
                                                           itty_feed_model_train_options_t const *options,
                                                           itty_feed_model_train_stats_t         *stats);
bool itty_feed_model_train_penultimate_layer_with_final_repairs (itty_feed_model_t                                  *model,
                                                                 itty_bit_string_list_t                             *input,
                                                                 itty_bit_string_t                                  *target,
                                                                 itty_feed_model_projected_repair_options_t const   *options,
                                                                 itty_feed_model_projected_repair_stats_t           *stats);
bool itty_feed_model_train_penultimate_layer_with_refreshed_final_repairs (itty_feed_model_t                                           *model,
                                                                          itty_bit_string_list_t                                      *input,
                                                                          itty_bit_string_t                                           *target,
                                                                          itty_feed_model_refreshed_projected_repair_options_t const  *options,
                                                                          itty_feed_model_refreshed_projected_repair_stats_t          *stats);
bool itty_feed_model_train_segment_condense_quota_repair_projection (itty_feed_model_t                                          *model,
                                                                     itty_bit_string_list_t                                     *input,
                                                                     itty_bit_string_t                                          *target,
                                                                     itty_feed_model_refreshed_projected_repair_options_t const *options,
                                                                     itty_feed_model_segment_training_summary_t                 *summary);
bool itty_feed_model_train_segment_condense_with_summary (itty_feed_model_t                                          *model,
                                                          itty_bit_string_list_t                                     *input,
                                                          itty_bit_string_t                                          *target,
                                                          itty_feed_model_refreshed_projected_repair_options_t const *options,
                                                          itty_feed_model_segment_training_summary_t                 *summary);
bool itty_feed_model_train_antepenultimate_layer_with_projected_repairs (itty_feed_model_t                                *model,
                                                                         itty_bit_string_list_t                           *input,
                                                                         itty_bit_string_t                                *target,
                                                                         itty_feed_model_projected_repair_options_t const *options,
                                                                         itty_feed_model_projected_repair_stats_t         *stats);
