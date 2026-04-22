#include "itty-bit-string.h"
#include "itty-bit-string-list.h"
#include "itty-feed-model.h"
#include "itty-inference.h"
#include "itty-manager.h"
#include "itty-network.h"
#include "itty-vocabulary.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char *
create_temp_file (const char *content,
                  size_t      size)
{
        char *filename = strdup ("/tmp/test-XXXXXX");
        int fd = mkstemp (filename);
        assert (fd != -1);
        assert (write (fd, content, size) == (ssize_t) size);
        close (fd);
        return filename;
}

static itty_bit_string_t *
create_bit_string (size_t word)
{
        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        itty_bit_string_append_word (bit_string, word);
        return bit_string;
}

static itty_vocabulary_t *
create_vocabulary (char **text_file,
                   char **bit_string_file)
{
        const char *text_content = " a\n b\n";
        size_t bit_string_words[] = { 0b0011, 0b1100 };

        *text_file = create_temp_file (text_content, strlen (text_content));
        *bit_string_file = create_temp_file ((char *) bit_string_words, sizeof (bit_string_words));

        return itty_vocabulary_new (*text_file, *bit_string_file);
}

static itty_bit_string_list_t *
create_input (void)
{
        itty_bit_string_list_t *input = itty_bit_string_list_new ();
        itty_bit_string_list_append (input, create_bit_string (0b0011));
        return input;
}

static itty_bit_string_list_t *
create_input_with_count (size_t count,
                         size_t word)
{
        itty_bit_string_list_t *input = itty_bit_string_list_new ();

        for (size_t i = 0; i < count; i++)
                itty_bit_string_list_append (input, create_bit_string (word));

        return input;
}

typedef struct {
        size_t a_selected_route;
        size_t a_selected_gap;
        size_t b_selected_route;
        size_t b_selected_gap;
} route_key_selection_summary_t;

typedef struct {
        size_t a_selected_route;
        size_t a_selected_gap;
        size_t b_selected_route;
        size_t b_selected_gap;
        size_t c_selected_route;
        size_t c_selected_gap;
} route_key_selection_summary_abc_t;

static route_key_selection_summary_t
measure_route_key_selection (itty_bit_string_t **route_keys,
                             size_t              nodes,
                             itty_bit_string_t  *a_probe,
                             itty_bit_string_t  *b_probe)
{
        route_key_selection_summary_t summary = { 0 };
        size_t a_selected_score = 0;
        size_t b_selected_score = 0;
        size_t a_runner_up_score = 0;
        size_t b_runner_up_score = 0;

        for (size_t route = 0; route < nodes; route++) {
                size_t a_score = itty_bit_string_evaluate_similarity (a_probe, route_keys[route]);
                size_t b_score = itty_bit_string_evaluate_similarity (b_probe, route_keys[route]);

                if (route == 0 || a_score > a_selected_score) {
                        a_runner_up_score = a_selected_score;
                        a_selected_score = a_score;
                        summary.a_selected_route = route;
                } else if (a_score > a_runner_up_score) {
                        a_runner_up_score = a_score;
                }

                if (route == 0 || b_score > b_selected_score) {
                        b_runner_up_score = b_selected_score;
                        b_selected_score = b_score;
                        summary.b_selected_route = route;
                } else if (b_score > b_runner_up_score) {
                        b_runner_up_score = b_score;
                }
        }

        summary.a_selected_gap = a_selected_score - a_runner_up_score;
        summary.b_selected_gap = b_selected_score - b_runner_up_score;
        return summary;
}

static route_key_selection_summary_abc_t
measure_route_key_selection_abc (itty_bit_string_t **route_keys,
                                 size_t              nodes,
                                 itty_bit_string_t  *a_probe,
                                 itty_bit_string_t  *b_probe,
                                 itty_bit_string_t  *c_probe)
{
        route_key_selection_summary_abc_t summary = { 0 };
        size_t a_selected_score = 0;
        size_t b_selected_score = 0;
        size_t c_selected_score = 0;
        size_t a_runner_up_score = 0;
        size_t b_runner_up_score = 0;
        size_t c_runner_up_score = 0;

        for (size_t route = 0; route < nodes; route++) {
                size_t a_score = itty_bit_string_evaluate_similarity (a_probe, route_keys[route]);
                size_t b_score = itty_bit_string_evaluate_similarity (b_probe, route_keys[route]);
                size_t c_score = itty_bit_string_evaluate_similarity (c_probe, route_keys[route]);

                if (route == 0 || a_score > a_selected_score) {
                        a_runner_up_score = a_selected_score;
                        a_selected_score = a_score;
                        summary.a_selected_route = route;
                } else if (a_score > a_runner_up_score) {
                        a_runner_up_score = a_score;
                }

                if (route == 0 || b_score > b_selected_score) {
                        b_runner_up_score = b_selected_score;
                        b_selected_score = b_score;
                        summary.b_selected_route = route;
                } else if (b_score > b_runner_up_score) {
                        b_runner_up_score = b_score;
                }

                if (route == 0 || c_score > c_selected_score) {
                        c_runner_up_score = c_selected_score;
                        c_selected_score = c_score;
                        summary.c_selected_route = route;
                } else if (c_score > c_runner_up_score) {
                        c_runner_up_score = c_score;
                }
        }

        summary.a_selected_gap = a_selected_score - a_runner_up_score;
        summary.b_selected_gap = b_selected_score - b_runner_up_score;
        summary.c_selected_gap = c_selected_score - c_runner_up_score;
        return summary;
}

static size_t
create_half_populated_word (void)
{
        return ((size_t) 1 << (ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2)) - 1;
}

static size_t
create_mixed_word (void)
{
        size_t word = 0;

        for (size_t bit_index = 0; bit_index < ITTY_BIT_STRING_WORD_SIZE_IN_BITS; bit_index += 3)
                word |= (size_t) 1 << bit_index;

        return word;
}

static itty_inference_result_t *
run_model (itty_feed_model_t     *model,
           itty_bit_string_list_t *input,
           itty_vocabulary_t      *vocabulary,
           itty_manager_t         *manager)
{
        itty_network_t *network = itty_feed_model_build_network (model);
        itty_inference_result_t *result = itty_inference_run (network,
                                                              input,
                                                              vocabulary,
                                                              manager);
        itty_network_free (network);
        return result;
}

static size_t
get_result_distance_to_target (itty_inference_result_t *result,
                               itty_bit_string_t       *target)
{
        itty_bit_string_t *activation = itty_inference_result_get_selected_activation (result);
        itty_bit_string_t *folded_activation = activation;
        while (itty_bit_string_get_number_of_words (folded_activation) > itty_bit_string_get_number_of_words (target)) {
                itty_bit_string_t *next = itty_bit_string_reduce_by_half (folded_activation);
                if (folded_activation != activation)
                        itty_bit_string_free (folded_activation);
                folded_activation = next;
        }

        itty_bit_string_t *difference = itty_bit_string_exclusive_or (folded_activation,
                                                                      target);
        size_t distance = itty_bit_string_get_pop_count (difference);
        if (folded_activation != activation)
                itty_bit_string_free (folded_activation);
        itty_bit_string_free (difference);
        return distance;
}

static void
test_itty_feed_model_train_one_learns_single_mask (void)
{
        char *text_file = NULL;
        char *bit_string_file = NULL;
        itty_vocabulary_t *vocabulary = create_vocabulary (&text_file, &bit_string_file);
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = itty_vocabulary_translate_to_bit_string (vocabulary, " b");
        itty_manager_t *manager = itty_manager_new ();

        itty_inference_result_t *before = run_model (model, input, vocabulary, manager);
        assert (before != NULL);
        assert (strcmp (itty_inference_result_get_text (before), " a") == 0);
        itty_inference_result_free (before);

        assert (itty_feed_model_train_one (model, input, target));

        itty_inference_result_t *after = run_model (model, input, vocabulary, manager);
        assert (after != NULL);
        assert (strcmp (itty_inference_result_get_text (after), " b") == 0);
        assert (itty_inference_result_get_distance (after) == 0);
        itty_inference_result_free (after);

        itty_manager_free (manager);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

static void
test_itty_feed_model_train_one_with_budget_moves_toward_target (void)
{
        char *text_file = NULL;
        char *bit_string_file = NULL;
        itty_vocabulary_t *vocabulary = create_vocabulary (&text_file, &bit_string_file);
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = itty_vocabulary_translate_to_bit_string (vocabulary, " b");
        itty_manager_t *manager = itty_manager_new ();
        itty_feed_model_train_options_t options = {
                .max_flips = 1,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST
        };

        assert (itty_feed_model_train_one_with_options (model, input, target, &options));

        itty_inference_result_t *after_one_step = run_model (model, input, vocabulary, manager);
        assert (after_one_step != NULL);
        assert (get_result_distance_to_target (after_one_step, target) == 3);
        itty_inference_result_free (after_one_step);

        for (size_t i = 0; i < 3; i++)
                assert (itty_feed_model_train_one_with_options (model, input, target, &options));

        itty_inference_result_t *after_four_steps = run_model (model, input, vocabulary, manager);
        assert (after_four_steps != NULL);
        assert (strcmp (itty_inference_result_get_text (after_four_steps), " b") == 0);
        assert (itty_inference_result_get_distance (after_four_steps) == 0);
        itty_inference_result_free (after_four_steps);

        itty_manager_free (manager);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

static void
test_itty_feed_model_train_one_with_budget_prioritizes_largest_error (void)
{
        char *text_file = NULL;
        char *bit_string_file = NULL;
        itty_vocabulary_t *vocabulary = create_vocabulary (&text_file, &bit_string_file);
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 3, 1);
        itty_bit_string_list_t *input = create_input_with_count (3, 0);
        itty_bit_string_t *target = itty_vocabulary_translate_to_bit_string (vocabulary, " b");
        itty_manager_t *manager = itty_manager_new ();
        itty_feed_model_train_options_t options = {
                .max_flips = 3,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST
        };
        itty_feed_model_train_stats_t stats = {
                .flips = 99,
                .candidate_bits = 99,
                .largest_error = 99
        };

        assert (itty_feed_model_train_one_with_stats (model, input, target, &options, &stats));
        assert (stats.flips == 3);
        assert (stats.candidate_bits == 2);
        assert (stats.largest_error == 2);

        itty_inference_result_t *after_one_step = run_model (model, input, vocabulary, manager);
        assert (after_one_step != NULL);
        assert (itty_inference_result_get_distance (after_one_step) == 1);
        itty_inference_result_free (after_one_step);

        assert (itty_feed_model_train_one_with_options (model, input, target, &options));

        itty_inference_result_t *after_two_steps = run_model (model, input, vocabulary, manager);
        assert (after_two_steps != NULL);
        assert (strcmp (itty_inference_result_get_text (after_two_steps), " b") == 0);
        assert (itty_inference_result_get_distance (after_two_steps) == 0);
        itty_inference_result_free (after_two_steps);

        itty_manager_free (manager);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

static void
test_itty_feed_model_train_one_trains_final_layer (void)
{
        char *text_file = NULL;
        char *bit_string_file = NULL;
        itty_vocabulary_t *vocabulary = create_vocabulary (&text_file, &bit_string_file);
        itty_feed_model_t *model = itty_feed_model_new (2, 1, 1, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = itty_vocabulary_translate_to_bit_string (vocabulary, " b");
        itty_manager_t *manager = itty_manager_new ();

        itty_inference_result_t *before = run_model (model, input, vocabulary, manager);
        assert (before != NULL);
        assert (strcmp (itty_inference_result_get_text (before), " a") == 0);
        itty_inference_result_free (before);

        assert (itty_feed_model_train_one (model, input, target));

        itty_inference_result_t *after = run_model (model, input, vocabulary, manager);
        assert (after != NULL);
        assert (strcmp (itty_inference_result_get_text (after), " b") == 0);
        assert (itty_inference_result_get_distance (after) == 0);
        itty_inference_result_free (after);

        itty_manager_free (manager);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

static void
test_itty_feed_model_train_one_supports_multi_node_final_layer (void)
{
        char *text_file = NULL;
        char *bit_string_file = NULL;
        itty_vocabulary_t *vocabulary = create_vocabulary (&text_file, &bit_string_file);
        itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = itty_vocabulary_translate_to_bit_string (vocabulary, " b");
        itty_manager_t *manager = itty_manager_new ();
        itty_model_metrics_bit_summary_t mask_summary;

        assert (itty_feed_model_measure_masks (model, &mask_summary));
        assert (mask_summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS * 10);
        assert (itty_feed_model_train_one (model, input, target));

        itty_inference_result_t *after = run_model (model, input, vocabulary, manager);
        assert (after != NULL);
        assert (strcmp (itty_inference_result_get_text (after), " b") == 0);
        assert (itty_inference_result_get_distance (after) == 0);
        itty_inference_result_free (after);

        itty_manager_free (manager);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

static void
test_itty_feed_model_train_backwards_one_trains_all_layers (void)
{
        char *text_file = NULL;
        char *bit_string_file = NULL;
        itty_vocabulary_t *vocabulary = create_vocabulary (&text_file, &bit_string_file);
        itty_feed_model_t *model = itty_feed_model_new (2, 1, 1, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = itty_vocabulary_translate_to_bit_string (vocabulary, " b");
        itty_manager_t *manager = itty_manager_new ();

        assert (itty_feed_model_train_backwards_one (model, input, target));

        itty_inference_result_t *after = run_model (model, input, vocabulary, manager);
        assert (after != NULL);
        assert (strcmp (itty_inference_result_get_text (after), " b") == 0);
        assert (itty_inference_result_get_distance (after) == 0);
        itty_inference_result_free (after);

        itty_manager_free (manager);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

static void
test_itty_feed_model_rotation_affects_forward_output (void)
{
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_feed_model_set_layer_rotation (model, 0, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_network_t *network = itty_feed_model_build_network (model);
        itty_bit_string_list_t *output = itty_network_feed (network, input);
        itty_bit_string_t *activation = itty_bit_string_list_fetch (output, 0);
        size_t *words = itty_bit_string_get_words (activation);

        assert (itty_bit_string_get_number_of_words (activation) == 2);
        assert (words[0] == 0b0011);
        assert (words[1] == 0b0110);

        itty_bit_string_list_free (output);
        itty_bit_string_list_free (input);
        itty_network_free (network);
        itty_feed_model_free (model);
}

static void
test_itty_feed_model_randomizes_masks_with_fixed_seed (void)
{
        itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
        itty_model_metrics_bit_summary_t summary;

        assert (!itty_feed_model_randomize_masks (model, 7, 1, 0));
        assert (!itty_feed_model_randomize_masks (model, 7, 2, 1));

        assert (itty_feed_model_randomize_masks (model, 7, 1, 8));
        assert (itty_feed_model_measure_masks (model, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS * 10);
        assert (summary.set_bits == 94);

        assert (itty_feed_model_randomize_masks (model, 8, 1, 8));
        assert (itty_feed_model_measure_masks (model, &summary));
        assert (summary.set_bits == 90);

        assert (itty_feed_model_randomize_masks (model, 7, 1, 8));
        assert (itty_feed_model_measure_masks (model, &summary));
        assert (summary.set_bits == 94);

        itty_feed_model_free (model);
}

static void
test_itty_feed_model_measures_final_layer_node_diagnostics (void)
{
        itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = create_bit_string (0b1100);
        itty_feed_model_node_diagnostic_t diagnostics[2];
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST
        };
        itty_feed_model_train_stats_t stats;

        assert (itty_feed_model_measure_final_layer_node_diagnostics (model,
                                                                      input,
                                                                      target,
                                                                      diagnostics,
                                                                      2));
        assert (diagnostics[0].node_index == 0);
        assert (diagnostics[0].desired_bits == 4);
        assert (diagnostics[0].actual_bits == 4);
        assert (diagnostics[0].mismatched_bits == 8);
        assert (diagnostics[0].bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS * 2);
        assert (diagnostics[1].node_index == 1);
        assert (diagnostics[1].desired_bits == diagnostics[0].desired_bits);
        assert (diagnostics[1].actual_bits == diagnostics[0].actual_bits);
        assert (diagnostics[1].mismatched_bits == diagnostics[0].mismatched_bits);

        assert (itty_feed_model_train_one_with_stats (model,
                                                      input,
                                                      target,
                                                      &options,
                                                      &stats));
        assert (stats.flips == 16);
        assert (stats.candidate_bits == 16);
        assert (itty_feed_model_measure_final_layer_node_diagnostics (model,
                                                                      input,
                                                                      target,
                                                                      diagnostics,
                                                                      2));
        assert (diagnostics[0].mismatched_bits == 4);
        assert (diagnostics[1].mismatched_bits == 4);

        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_feed_model_randomized_masks_diverge_final_layer_node_diagnostics (void)
{
        itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = create_bit_string (0b1100);
        itty_feed_model_node_diagnostic_t diagnostics[2];
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST
        };
        itty_feed_model_train_stats_t stats;

        assert (itty_feed_model_randomize_masks (model, 7, 1, 8));
        assert (itty_feed_model_measure_final_layer_node_diagnostics (model,
                                                                      input,
                                                                      target,
                                                                      diagnostics,
                                                                      2));
        assert (diagnostics[0].actual_bits == 9);
        assert (diagnostics[0].mismatched_bits == 13);
        assert (diagnostics[1].actual_bits == 10);
        assert (diagnostics[1].mismatched_bits == 14);

        assert (itty_feed_model_train_one_with_stats (model,
                                                      input,
                                                      target,
                                                      &options,
                                                      &stats));
        assert (stats.flips == 16);
        assert (stats.candidate_bits == 27);
        assert (itty_feed_model_measure_final_layer_node_diagnostics (model,
                                                                      input,
                                                                      target,
                                                                      diagnostics,
                                                                      2));
        assert (diagnostics[0].actual_bits == 13);
        assert (diagnostics[0].mismatched_bits == 9);
        assert (diagnostics[1].actual_bits == 11);
        assert (diagnostics[1].mismatched_bits == 9);

        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_feed_model_train_backwards_one_supports_hidden_rotation (void)
{
        char *text_file = NULL;
        char *bit_string_file = NULL;
        itty_vocabulary_t *vocabulary = create_vocabulary (&text_file, &bit_string_file);
        itty_feed_model_t *model = itty_feed_model_new (2, 1, 1, 1);
        itty_feed_model_set_layer_rotation (model, 0, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = itty_vocabulary_translate_to_bit_string (vocabulary, " b");
        itty_manager_t *manager = itty_manager_new ();

        assert (itty_feed_model_train_backwards_one (model, input, target));

        itty_inference_result_t *after = run_model (model, input, vocabulary, manager);
        assert (after != NULL);
        assert (strcmp (itty_inference_result_get_text (after), " b") == 0);
        assert (itty_inference_result_get_distance (after) == 0);
        itty_inference_result_free (after);

        itty_manager_free (manager);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

static void
test_itty_feed_model_measures_segment_node_selection (void)
{
        itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = create_bit_string (0b1100);
        itty_feed_model_segment_node_selection_summary_t summary;

        assert (itty_feed_model_randomize_masks (model, 7, 1, 8));
        assert (itty_feed_model_measure_segment_node_selection (model,
                                                                input,
                                                                target,
                                                                &summary));
        assert (summary.selected_by_popcount < 2);
        assert (summary.best_by_target_distance < 2);
        assert (summary.best_by_false_negative_deficit < 2);
        assert (summary.best_by_false_positive_excess < 2);
        assert (summary.best_target_distance <= summary.selected_distance);
        assert (summary.distance_gap_between_selected_and_best ==
                summary.selected_distance - summary.best_target_distance);
        assert (summary.best_false_negative_deficit <= summary.selected_false_negative_deficit);
        assert (summary.best_false_positive_excess <= summary.selected_false_positive_excess);

        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_feed_model_train_backwards_one_rejects_multi_node_chained_reduce (void)
{
        itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = create_bit_string (0b1100);

        assert (!itty_feed_model_train_backwards_one (model, input, target));

        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_feed_model_measures_backward_layer_diagnostics (void)
{
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *input = create_input_with_count (1, 0);
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_backward_layer_diagnostic_t diagnostics[1];

        assert (itty_feed_model_measure_backward_layer_diagnostics (model,
                                                                    input,
                                                                    target,
                                                                    &options,
                                                                    diagnostics,
                                                                    1));
        assert (diagnostics[0].layer_index == 0);
        assert (diagnostics[0].desired_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2);
        assert (diagnostics[0].actual_bits == 0);
        assert (diagnostics[0].mismatched_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2);
        assert (diagnostics[0].bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);

        assert (itty_feed_model_train_backwards_one_with_options (model,
                                                                  input,
                                                                  target,
                                                                  &options));
        assert (itty_feed_model_measure_backward_layer_diagnostics (model,
                                                                    input,
                                                                    target,
                                                                    &options,
                                                                    diagnostics,
                                                                    1));
        assert (diagnostics[0].mismatched_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2 - 8);

        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_feed_model_measures_suffix_oracle (void)
{
        itty_feed_model_t *model = itty_feed_model_new (3, 2, 1, 1);
        itty_bit_string_list_t *input = create_input_with_count (1, 0);
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t train_options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_suffix_oracle_options_t oracle_options = {
                .layer_index = 2,
                .node_index = 0,
                .max_candidate_bits = 32,
                .backward_node_target = ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SAME
        };
        itty_feed_model_suffix_oracle_summary_t summary;

        itty_feed_model_set_layer_rotation (model, 0, 1);
        itty_feed_model_set_layer_rotation (model, 1, 2);

        assert (itty_feed_model_train_backwards_one_with_options (model,
                                                                  input,
                                                                  target,
                                                                  &train_options));
        assert (itty_feed_model_train_backwards_one_with_options (model,
                                                                  input,
                                                                  target,
                                                                  &train_options));

        assert (itty_feed_model_measure_suffix_oracle (model,
                                                       input,
                                                       target,
                                                       &oracle_options,
                                                       &summary));
        assert (summary.before_distance == 24);
        assert (summary.candidate_bits == 32);
        assert (summary.helpful_bits == 5);
        assert (summary.harmful_bits == 0);
        assert (summary.neutral_bits == 27);
        assert (summary.neutral_same_folded_output_bits == 27);
        assert (summary.neutral_changed_folded_output_bits == 0);
        assert (summary.neutral_changed_selected_output_bits == 0);
        assert (summary.best_distance == 23);
        assert (summary.worst_distance == 24);

        oracle_options.candidate_source = ITTY_FEED_MODEL_SUFFIX_ORACLE_CANDIDATES_RANDOM_ONE_BIT;
        oracle_options.random_seed = 11;
        assert (itty_feed_model_measure_suffix_oracle (model,
                                                       input,
                                                       target,
                                                       &oracle_options,
                                                       &summary));
        assert (summary.before_distance == 24);
        assert (summary.candidate_bits == 32);
        assert (summary.helpful_bits == 2);
        assert (summary.harmful_bits == 0);
        assert (summary.neutral_bits == 30);
        assert (summary.neutral_same_folded_output_bits == 22);
        assert (summary.neutral_changed_folded_output_bits == 0);
        assert (summary.neutral_changed_selected_output_bits == 8);
        assert (summary.best_distance == 23);
        assert (summary.worst_distance == 24);

        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static size_t
measure_feed_model_distance_with_suffix_oracle (itty_feed_model_t      *model,
                                                itty_bit_string_list_t *input,
                                                itty_bit_string_t      *target)
{
        itty_feed_model_suffix_oracle_options_t oracle_options = {
                .layer_index = 0,
                .node_index = 0,
                .max_candidate_bits = 1,
                .backward_node_target = ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SAME
        };
        itty_feed_model_suffix_oracle_summary_t summary;

        assert (itty_feed_model_measure_suffix_oracle (model,
                                                       input,
                                                       target,
                                                       &oracle_options,
                                                       &summary));
        return summary.before_distance;
}

static void
train_hard_feed_model_to_stall (itty_feed_model_t              *model,
                                itty_bit_string_list_t         *input,
                                itty_bit_string_t              *target,
                                itty_feed_model_train_options_t *options)
{
        for (size_t layer_index = 0; layer_index < 7; layer_index++)
                itty_feed_model_set_layer_rotation (model, layer_index, layer_index + 1);

        for (size_t step = 0; step < 64; step++)
                assert (itty_feed_model_train_backwards_one_with_options (model,
                                                                          input,
                                                                          target,
                                                                          options));
}

static void
train_hard_feed_model_to_stall_with_shape (itty_feed_model_t              *model,
                                           itty_bit_string_list_t         *input,
                                           itty_bit_string_t              *target,
                                           itty_feed_model_train_options_t *options,
                                           size_t                          rotation_layers)
{
        for (size_t layer_index = 0; layer_index < rotation_layers; layer_index++)
                itty_feed_model_set_layer_rotation (model, layer_index, layer_index + 1);

        for (size_t step = 0; step < 64; step++)
                assert (itty_feed_model_train_backwards_one_with_options (model,
                                                                          input,
                                                                          target,
                                                                          options));
}

static void
cleanup_hard_feed_model_with_final_oracle (itty_feed_model_t                     *model,
                                           itty_bit_string_list_t                *input,
                                           itty_bit_string_t                     *target,
                                           itty_feed_model_train_options_t const *options,
                                           size_t                                *steps,
                                           size_t                                *flips)
{
        itty_feed_model_decoder_objective_t objective;

        *steps = 0;
        *flips = 0;

        for (size_t step = 0; step < 256; step++) {
                assert (itty_feed_model_measure_decoder_objective (model,
                                                                   input,
                                                                   target,
                                                                   &objective));
                if (objective.selected_distance == 0)
                        break;

                itty_feed_model_train_stats_t stats;
                assert (itty_feed_model_train_final_layer_with_suffix_oracle (model,
                                                                              input,
                                                                              target,
                                                                              options,
                                                                              &stats));
                (*steps)++;
                *flips += stats.flips;
        }
}

static void
test_itty_feed_model_final_layer_suffix_oracle_trainer_improves_hard_case (void)
{
        itty_feed_model_t *model = itty_feed_model_new (8, 2, 1, 1);
        itty_bit_string_list_t *input = create_input_with_count (1, 0);
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_train_stats_t stats;
        itty_feed_model_residual_decode_summary_t residual;
        itty_feed_model_decoder_objective_t objective;

        train_hard_feed_model_to_stall (model, input, target, &options);

        assert (measure_feed_model_distance_with_suffix_oracle (model, input, target) == 14);

        itty_feed_model_suffix_oracle_options_t oracle_options = {
                .layer_index = 7,
                .node_index = 0,
                .max_candidate_bits = 256,
                .backward_node_target = ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SAME
        };
        itty_feed_model_suffix_oracle_summary_t oracle_summary;

        assert (itty_feed_model_measure_suffix_oracle (model,
                                                       input,
                                                       target,
                                                       &oracle_options,
                                                       &oracle_summary));
        assert (oracle_summary.candidate_bits == 256);
        assert (oracle_summary.helpful_bits == 6);
        assert (oracle_summary.blocker_helpful_bits == 248);
        assert (oracle_summary.harmful_bits == 0);
        assert (oracle_summary.true_neutral_bits == 2);

        oracle_options.backward_node_target = ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DECODER_TRI_STATE;
        assert (itty_feed_model_measure_suffix_oracle (model,
                                                       input,
                                                       target,
                                                       &oracle_options,
                                                       &oracle_summary));
        assert (oracle_summary.candidate_bits == 256);
        assert (oracle_summary.helpful_bits == 6);
        assert (oracle_summary.blocker_helpful_bits == 250);
        assert (oracle_summary.harmful_bits == 0);
        assert (oracle_summary.true_neutral_bits == 0);

        assert (itty_feed_model_train_final_layer_with_suffix_oracle (model,
                                                                      input,
                                                                      target,
                                                                      &options,
                                                                      &stats));
        assert (stats.flips == 16);
        assert (measure_feed_model_distance_with_suffix_oracle (model, input, target) == 11);

        assert (itty_feed_model_train_final_layer_with_suffix_oracle (model,
                                                                      input,
                                                                      target,
                                                                      &options,
                                                                      &stats));
        assert (stats.flips == 16);
        assert (measure_feed_model_distance_with_suffix_oracle (model, input, target) == 8);
        assert (itty_feed_model_measure_residual_decode (model,
                                                         input,
                                                         target,
                                                         &residual));
        assert (residual.distance == 8);
        assert (residual.false_positive_bits == 0);
        assert (residual.false_negative_bits == 8);
        assert (residual.false_negative_blocker_bits == 956);
        assert (residual.zero_veto_safety_bits == 7816);
        assert (itty_feed_model_measure_decoder_objective (model,
                                                           input,
                                                           target,
                                                           &objective));
        assert (objective.selected_distance == 8);
        assert (objective.false_negative_count == 8);
        assert (objective.false_negative_blocker_bits == 956);
        assert (objective.zero_veto_safety_bits == 7816);
        assert (objective.nearest_wrong_margin == 0);
        assert (objective.selected_node == 0);
        assert (objective.selected_popcount == 7612);
        assert (objective.best_decoded_node == 0);
        assert (objective.best_decoded_distance == 8);

        for (size_t step = 2; step < 122; step++)
                assert (itty_feed_model_train_final_layer_with_suffix_oracle (model,
                                                                              input,
                                                                              target,
                                                                              &options,
                                                                              &stats));

        assert (itty_feed_model_measure_residual_decode (model,
                                                         input,
                                                         target,
                                                         &residual));
        assert (residual.distance == 0);
        assert (residual.false_positive_bits == 0);
        assert (residual.false_negative_bits == 0);
        assert (residual.false_negative_blocker_bits == 0);
        assert (itty_feed_model_measure_decoder_objective (model,
                                                           input,
                                                           target,
                                                           &objective));
        assert (objective.selected_distance == 0);
        assert (objective.false_negative_count == 0);
        assert (objective.false_negative_blocker_bits == 0);
        assert (objective.selected_node == 0);
        assert (objective.best_decoded_node == 0);
        assert (objective.best_decoded_distance == 0);

        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_feed_model_penultimate_projected_repairs_reduce_final_cleanup (void)
{
        itty_feed_model_t *baseline_model = itty_feed_model_new (8, 2, 1, 1);
        itty_feed_model_t *projected_model = itty_feed_model_new (8, 2, 1, 1);
        itty_feed_model_t *refreshed_model = itty_feed_model_new (8, 2, 1, 1);
        itty_feed_model_t *residual_model = itty_feed_model_new (8, 2, 1, 1);
        itty_feed_model_t *layer5_model = itty_feed_model_new (8, 2, 1, 1);
        itty_bit_string_list_t *input = create_input_with_count (1, 0);
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t train_options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_projected_repair_options_t projected_options = {
                .max_projected_blocks = 128,
                .max_layer_flips = 256
        };
        itty_feed_model_refreshed_projected_repair_round_t trajectory[64] = { 0 };
        itty_feed_model_refreshed_projected_repair_options_t refreshed_options = {
                .batch_size = 16,
                .max_rounds = 128,
                .max_layer_flips_per_batch = 32,
                .trajectory = trajectory,
                .trajectory_count = 64
        };
        itty_feed_model_refreshed_projected_repair_options_t residual_options = {
                .batch_size = 16,
                .max_rounds = 128,
                .max_layer_flips_per_batch = 32,
                .use_or_residual_repairs = true
        };
        itty_feed_model_projected_repair_options_t layer5_options = {
                .max_projected_blocks = 8,
                .max_layer_flips = 4,
                .max_strict_distance_blocks = 1,
                .limit_strict_distance_blocks = true,
                .max_blocker_blocks = 0,
                .limit_blocker_blocks = true
        };
        itty_feed_model_projected_repair_stats_t projected_stats;
        itty_feed_model_projected_repair_stats_t layer5_stats;
        itty_feed_model_refreshed_projected_repair_stats_t refreshed_stats;
        itty_feed_model_refreshed_projected_repair_stats_t residual_stats;
        itty_feed_model_refreshed_projected_repair_stats_t layer5_cleanup_stats;
        size_t baseline_cleanup_steps;
        size_t baseline_cleanup_flips;
        size_t projected_cleanup_steps;
        size_t projected_cleanup_flips;
        size_t refreshed_cleanup_steps;
        size_t refreshed_cleanup_flips;

        train_hard_feed_model_to_stall (baseline_model, input, target, &train_options);
        cleanup_hard_feed_model_with_final_oracle (baseline_model,
                                                   input,
                                                   target,
                                                   &train_options,
                                                   &baseline_cleanup_steps,
                                                   &baseline_cleanup_flips);
        assert (baseline_cleanup_steps == 122);
        assert (baseline_cleanup_flips == 1952);

        train_hard_feed_model_to_stall (projected_model, input, target, &train_options);
        assert (itty_feed_model_train_penultimate_layer_with_final_repairs (projected_model,
                                                                           input,
                                                                           target,
                                                                           &projected_options,
                                                                           &projected_stats));
        assert (projected_stats.accepted_blocks == 98);
        assert (projected_stats.fully_realized_blocks == 98);
        assert (projected_stats.partially_realized_blocks == 0);
        assert (projected_stats.unrealized_blocks == 0);
        assert (projected_stats.condensed_realistic_blocks == 128);
        assert (projected_stats.condensed_realistic_strict_distance_helpful_blocks == 6);
        assert (projected_stats.condensed_realistic_blocker_helpful_blocks == 120);
        assert (projected_stats.condensed_realistic_objective_helpful_blocks == 0);
        assert (projected_stats.condensed_realistic_harmful_blocks == 0);
        assert (projected_stats.condensed_realistic_neutral_blocks == 2);
        assert (projected_stats.estimated_layer_flips == 254);
        assert (projected_stats.requested_condensed_bits == 196);
        assert (projected_stats.realized_condensed_bits == 196);
        assert (projected_stats.already_satisfied_condensed_bits == 0);
        assert (projected_stats.condensed_bits_needing_flips == 196);
        assert (projected_stats.available_flippable_votes == 254);
        assert (projected_stats.requested_output_bits == 196);
        assert (projected_stats.realized_output_bits == 196);
        assert (projected_stats.lost_output_bits == 0);
        assert (projected_stats.extra_output_bits == 196);
        assert (projected_stats.structural_extra_output_bits == 196);
        assert (projected_stats.collateral_extra_output_bits == 0);
        assert (projected_stats.layer_flips == 254);
        assert (projected_stats.before_distance == 14);
        assert (projected_stats.after_distance == 8);
        assert (projected_stats.before_blockers == 972);
        assert (projected_stats.after_blockers == 776);

        cleanup_hard_feed_model_with_final_oracle (projected_model,
                                                   input,
                                                   target,
                                                   &train_options,
                                                   &projected_cleanup_steps,
                                                   &projected_cleanup_flips);
        assert (projected_cleanup_steps == 97);
        assert (projected_cleanup_flips == 1552);
        assert (projected_cleanup_steps < baseline_cleanup_steps);
        assert (projected_cleanup_flips < baseline_cleanup_flips);

        train_hard_feed_model_to_stall (refreshed_model, input, target, &train_options);
        assert (itty_feed_model_train_penultimate_layer_with_refreshed_final_repairs (refreshed_model,
                                                                                     input,
                                                                                     target,
                                                                                     &refreshed_options,
                                                                                     &refreshed_stats));
        assert (refreshed_stats.rounds == 50);
        assert (refreshed_stats.accepted_rounds == 50);
        assert (refreshed_stats.rejected_rounds == 0);
        assert (refreshed_stats.before_distance == 14);
        assert (refreshed_stats.after_distance == 0);
        assert (refreshed_stats.before_blockers == 972);
        assert (refreshed_stats.after_blockers == 0);
        assert (refreshed_stats.projected.accepted_blocks == 490);
        assert (refreshed_stats.projected.fully_realized_blocks == 490);
        assert (refreshed_stats.projected.partially_realized_blocks == 0);
        assert (refreshed_stats.projected.unrealized_blocks == 0);
        assert (refreshed_stats.projected.layer_flips == 1560);
        assert (refreshed_stats.projected.collateral_extra_output_bits == 0);
        assert (refreshed_stats.projected.previous_layer_projected_blocks == 490);
        assert (refreshed_stats.projected.previous_layer_strict_distance_helpful_blocks == 14);
        assert (refreshed_stats.projected.previous_layer_blocker_helpful_blocks == 221);
        assert (refreshed_stats.projected.previous_layer_objective_helpful_blocks == 0);
        assert (refreshed_stats.projected.previous_layer_harmful_blocks == 221);
        assert (refreshed_stats.projected.previous_layer_neutral_blocks == 34);
        assert (refreshed_stats.projected.previous_layer_harmful_distance_blocks == 221);
        assert (refreshed_stats.projected.previous_layer_harmful_blocker_blocks == 0);
        assert (refreshed_stats.projected.previous_layer_harmful_margin_blocks == 0);
        assert (refreshed_stats.projected.previous_layer_harmful_safety_blocks == 0);
        assert (refreshed_stats.projected.previous_layer_pinned_projected_blocks == 490);
        assert (refreshed_stats.projected.previous_layer_pinned_strict_distance_helpful_blocks == 14);
        assert (refreshed_stats.projected.previous_layer_pinned_blocker_helpful_blocks == 221);
        assert (refreshed_stats.projected.previous_layer_pinned_objective_helpful_blocks == 0);
        assert (refreshed_stats.projected.previous_layer_pinned_harmful_blocks == 221);
        assert (refreshed_stats.projected.previous_layer_pinned_neutral_blocks == 34);
        assert (trajectory[0].accepted);
        assert (!trajectory[0].reverted);
        assert (trajectory[0].selected_node == 0);
        assert (trajectory[0].popcount_gap == 0);
        assert (trajectory[0].best_decoded_node == 0);
        assert (trajectory[0].best_decoded_distance == 14);
        assert (trajectory[0].selected_is_best_decoded);
        assert (trajectory[0].before_distance == 14);
        assert (trajectory[0].after_distance == 8);
        assert (trajectory[0].distance_delta == 6);
        assert (trajectory[0].before_blockers == 972);
        assert (trajectory[0].after_blockers == 956);
        assert (trajectory[0].blocker_delta == 16);
        assert (trajectory[0].accepted_blocks == 8);
        assert (trajectory[0].layer_flips == 32);
        assert (trajectory[49].accepted);
        assert (trajectory[49].after_distance == 0);
        assert (trajectory[49].after_blockers == 0);
        cleanup_hard_feed_model_with_final_oracle (refreshed_model,
                                                   input,
                                                   target,
                                                   &train_options,
                                                   &refreshed_cleanup_steps,
                                                   &refreshed_cleanup_flips);
        assert (refreshed_cleanup_steps == 0);
        assert (refreshed_cleanup_flips == 0);
        assert (refreshed_stats.projected.layer_flips < projected_stats.layer_flips + projected_cleanup_flips);

        train_hard_feed_model_to_stall (residual_model, input, target, &train_options);
        assert (itty_feed_model_train_penultimate_layer_with_refreshed_final_repairs (residual_model,
                                                                                     input,
                                                                                     target,
                                                                                     &residual_options,
                                                                                     &residual_stats));
        assert (residual_stats.after_distance == 0);
        assert (residual_stats.projected.layer_flips == 1560);
        assert (residual_stats.projected.residual_candidate_blocks == 613);
        assert (residual_stats.projected.residual_accepted_blocks == 0);
        assert (residual_stats.projected.residual_enable_flips == 0);
        assert (residual_stats.projected.residual_mask_flips == 0);

        train_hard_feed_model_to_stall (layer5_model, input, target, &train_options);
        assert (itty_feed_model_train_antepenultimate_layer_with_projected_repairs (layer5_model,
                                                                                   input,
                                                                                   target,
                                                                                   &layer5_options,
                                                                                   &layer5_stats));
        assert (layer5_stats.condensed_realistic_blocks == 8);
        assert (layer5_stats.accepted_blocks == 1);
        assert (layer5_stats.accepted_strict_distance_blocks == 1);
        assert (layer5_stats.accepted_blocker_blocks == 0);
        assert (layer5_stats.condensed_realistic_strict_distance_helpful_blocks == 6);
        assert (layer5_stats.condensed_realistic_blocker_helpful_blocks == 2);
        assert (layer5_stats.strict_distance_layer_flips == 2);
        assert (layer5_stats.blocker_layer_flips == 0);
        assert (layer5_stats.layer_flips == 2);
        assert (layer5_stats.after_distance == 13);
        assert (layer5_stats.after_blockers == 968);
        assert (itty_feed_model_train_penultimate_layer_with_refreshed_final_repairs (layer5_model,
                                                                                     input,
                                                                                     target,
                                                                                     &refreshed_options,
                                                                                     &layer5_cleanup_stats));
        assert (layer5_cleanup_stats.after_distance == 0);
        assert (layer5_cleanup_stats.projected.layer_flips == 1552);
        assert (layer5_stats.layer_flips + layer5_cleanup_stats.projected.layer_flips == 1554);

        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (baseline_model);
        itty_feed_model_free (projected_model);
        itty_feed_model_free (refreshed_model);
        itty_feed_model_free (residual_model);
        itty_feed_model_free (layer5_model);
}

static void
test_itty_feed_model_segment_condense_decoder_reduces_target_one_fragility (void)
{
        itty_feed_model_t *hard_model = itty_feed_model_new (8, 2, 1, 1);
        itty_feed_model_t *wide_model = itty_feed_model_new (8, 4, 1, 1);
        itty_bit_string_list_t *input = create_input_with_count (1, 0);
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t train_options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_residual_decode_summary_t repeated_summary;
        itty_feed_model_residual_decode_summary_t segment_summary;
        itty_feed_model_decoder_objective_t segment_objective;
        itty_feed_model_refreshed_projected_repair_options_t refreshed_options = {
                .batch_size = 64,
                .max_rounds = 128,
                .max_layer_flips_per_batch = 256
        };
        itty_feed_model_segment_training_summary_t segment_training_summary;

        train_hard_feed_model_to_stall (hard_model,
                                        input,
                                        target,
                                        &train_options);
        assert (itty_feed_model_measure_residual_decode (hard_model,
                                                         input,
                                                         target,
                                                         &repeated_summary));
        itty_feed_model_set_decoder (hard_model,
                                     ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        assert (itty_feed_model_measure_residual_decode (hard_model,
                                                         input,
                                                         target,
                                                         &segment_summary));
        assert (repeated_summary.distance == 14);
        assert (segment_summary.distance == 5);
        assert (segment_summary.false_positive_bits == 0);
        assert (segment_summary.false_negative_blocker_bits < repeated_summary.false_negative_blocker_bits);
        assert (itty_feed_model_measure_decoder_objective (hard_model,
                                                           input,
                                                           target,
                                                           &segment_objective));
        assert (segment_objective.false_negative_vote_deficit == 5);
        assert (segment_objective.false_negative_vote_deficit_histogram[1] == 5);
        assert (segment_objective.false_positive_vote_excess == 0);
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (hard_model,
                                                                                input,
                                                                                target,
                                                                                &refreshed_options,
                                                                                &segment_training_summary));
        assert (segment_training_summary.initial_distance == 5);
        assert (segment_training_summary.final_distance == 0);
        assert (segment_training_summary.layer6_flips >= 8);
        assert (segment_training_summary.layer6_flips <= 12);
        assert (segment_training_summary.rounds == 1);
        assert (segment_training_summary.accepted_blocks == 10);
        assert (segment_training_summary.quota_size_total >= segment_training_summary.accepted_blocks);
        assert (segment_training_summary.quota_size_max >= 1);
        assert (segment_training_summary.quota_size_max <= 2);
        assert (segment_training_summary.average_quota_size >= 1.0);
        assert (segment_training_summary.average_quota_size <= 2.0);
        assert (segment_training_summary.false_negative_vote_deficit_before == 5);
        assert (segment_training_summary.false_negative_vote_deficit_after == 0);
        assert (segment_training_summary.false_positive_vote_excess_before == 0);
        assert (segment_training_summary.false_positive_vote_excess_after == 0);
        assert (segment_training_summary.target_zero_safety_minimum > 0);
        assert (segment_training_summary.quota_completion_efficiency > 0.0);
        assert (segment_training_summary.vote_efficiency > 0.0);
        assert (segment_training_summary.direct_quota_vote_flips +
                segment_training_summary.majority_threshold_support_flips ==
                segment_training_summary.layer6_flips);
        assert (segment_training_summary.quota_vote_support_cost_histogram[1] ==
                segment_training_summary.direct_quota_vote_flips);
        assert (segment_training_summary.decoded_bits_fixed == 5);
        assert (segment_training_summary.average_flips_per_fixed_decoded_bit == 2.0);
        assert (segment_training_summary.min_flips_per_fixed_decoded_bit == 2);
        assert (segment_training_summary.max_flips_per_fixed_decoded_bit == 2);
        assert (segment_training_summary.average_final_target_one_margin > 0.0);
        assert (segment_training_summary.conflict_resolution_flips == 0);
        assert (segment_training_summary.collateral_preservation_flips == 0);
        assert (segment_training_summary.target_zero_safety_preservation_flips == 0);
        assert (segment_training_summary.selection_preservation_flips == 0);
        assert (segment_training_summary.training.projected.fully_realized_blocks ==
                segment_training_summary.training.projected.accepted_blocks);
        assert (itty_feed_model_measure_decoder_objective (hard_model,
                                                           input,
                                                           target,
                                                           &segment_objective));
        assert (segment_objective.selected_distance == 0);
        assert (segment_objective.false_negative_vote_deficit == 0);

        train_hard_feed_model_to_stall_with_shape (wide_model,
                                                   input,
                                                   target,
                                                   &train_options,
                                                   7);
        assert (itty_feed_model_measure_residual_decode (wide_model,
                                                         input,
                                                         target,
                                                         &repeated_summary));
        itty_feed_model_set_decoder (wide_model,
                                     ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        assert (itty_feed_model_measure_residual_decode (wide_model,
                                                         input,
                                                         target,
                                                         &segment_summary));
        assert (repeated_summary.distance == 32);
        assert (segment_summary.distance == 0);
        assert (segment_summary.false_positive_bits == 0);
        assert (segment_summary.false_negative_bits == 0);

        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (hard_model);
        itty_feed_model_free (wide_model);
}

static void
test_itty_feed_model_segment_shape_matrix_smoke (void)
{
        size_t layers[] = { 1, 2, 4, 8, 12 };
        size_t nodes[] = { 1, 2, 4, 8 };
        size_t target_words[] = {
                create_half_populated_word (),
                1,
                create_mixed_word ()
        };
        size_t checked_shapes = 0;

        for (size_t layer_index = 0; layer_index < sizeof (layers) / sizeof (layers[0]); layer_index++) {
                for (size_t node_index = 0; node_index < sizeof (nodes) / sizeof (nodes[0]); node_index++) {
                        for (size_t rotation_mode = 0; rotation_mode < 3; rotation_mode++) {
                                for (size_t init_mode = 0; init_mode < 2; init_mode++) {
                                        for (size_t target_index = 0; target_index < sizeof (target_words) / sizeof (target_words[0]); target_index++) {
                                                itty_feed_model_t *model = itty_feed_model_new (layers[layer_index],
                                                                                                nodes[node_index],
                                                                                                1,
                                                                                                1);
                                                itty_bit_string_list_t *input = create_input_with_count (1,
                                                                                                         target_index);
                                                itty_bit_string_t *target = create_bit_string (target_words[target_index]);
                                                itty_feed_model_decoder_objective_t objective;
                                                itty_feed_model_segment_node_selection_summary_t selection;

                                                itty_feed_model_set_decoder (model,
                                                                             ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                                                for (size_t model_layer = 0; model_layer < layers[layer_index]; model_layer++) {
                                                        if (rotation_mode == 1)
                                                                itty_feed_model_set_layer_rotation (model,
                                                                                                    model_layer,
                                                                                                    1);
                                                        else if (rotation_mode == 2)
                                                                itty_feed_model_set_layer_rotation (model,
                                                                                                    model_layer,
                                                                                                    model_layer + 1);
                                                }
                                                if (init_mode == 1)
                                                        assert (itty_feed_model_randomize_masks (model,
                                                                                                 0x5eed + checked_shapes,
                                                                                                 1,
                                                                                                 8));

                                                assert (itty_feed_model_measure_decoder_objective (model,
                                                                                                   input,
                                                                                                   target,
                                                                                                   &objective));
                                                assert (itty_feed_model_measure_segment_node_selection (model,
                                                                                                        input,
                                                                                                        target,
                                                                                                        &selection));
                                                assert (objective.selected_distance <= ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
                                                assert (selection.selected_by_popcount < nodes[node_index]);
                                                assert (selection.best_by_target_distance < nodes[node_index]);

                                                checked_shapes++;
                                                itty_bit_string_free (target);
                                                itty_bit_string_list_free (input);
                                                itty_feed_model_free (model);
                                        }
                                }
                        }
                }
        }

        assert (checked_shapes == 5 * 4 * 3 * 2 * 3);
}

static void
test_itty_feed_model_segment_training_limits_multi_example_clobber (void)
{
        itty_feed_model_t *model = itty_feed_model_new (8, 2, 1, 1);
        itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
        itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t warm_options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_refreshed_projected_repair_options_t repair_options = {
                .batch_size = 64,
                .max_rounds = 128,
                .max_layer_flips_per_batch = 256
        };
        itty_feed_model_decoder_objective_t first_before;
        itty_feed_model_decoder_objective_t first_after;
        itty_feed_model_decoder_objective_t second_after;
        itty_feed_model_decoder_objective_t first_after_second;
        itty_feed_model_segment_training_summary_t summary;

        train_hard_feed_model_to_stall (model,
                                        first_input,
                                        target,
                                        &warm_options);
        itty_feed_model_set_decoder (model,
                                     ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        assert (itty_feed_model_measure_decoder_objective (model,
                                                           first_input,
                                                           target,
                                                           &first_before));

        assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                first_input,
                                                                                target,
                                                                                &repair_options,
                                                                                &summary));
        assert (itty_feed_model_measure_decoder_objective (model,
                                                           first_input,
                                                           target,
                                                           &first_after));
        assert (first_after.selected_distance == 0);

        assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                second_input,
                                                                                target,
                                                                                &repair_options,
                                                                                &summary));
        assert (itty_feed_model_measure_decoder_objective (model,
                                                           second_input,
                                                           target,
                                                           &second_after));
        assert (itty_feed_model_measure_decoder_objective (model,
                                                           first_input,
                                                           target,
                                                           &first_after_second));

        assert (second_after.selected_distance == 0);
        assert (first_after_second.selected_distance <= first_before.selected_distance);

        itty_bit_string_free (target);
        itty_bit_string_list_free (first_input);
        itty_bit_string_list_free (second_input);
        itty_feed_model_free (model);
}

typedef struct {
        size_t initial_distance;
        size_t final_distance;
        size_t flips;
        size_t rounds;
        size_t deficit_before;
        size_t deficit_after;
        size_t excess_before;
        size_t excess_after;
        size_t zero_safety;
        double entropy_before;
        double entropy_after;
        itty_feed_model_segment_train_stop_reason_t stop_reason;
} itty_feed_model_test_segment_training_row_t;

static void
capture_segment_training_row (itty_feed_model_segment_training_summary_t const *summary,
                              itty_feed_model_test_segment_training_row_t      *row)
{
        row->initial_distance = summary->initial_distance;
        row->final_distance = summary->final_distance;
        row->flips = summary->layer6_flips;
        row->rounds = summary->rounds;
        row->deficit_before = summary->false_negative_vote_deficit_before;
        row->deficit_after = summary->false_negative_vote_deficit_after;
        row->excess_before = summary->false_positive_vote_excess_before;
        row->excess_after = summary->false_positive_vote_excess_after;
        row->zero_safety = summary->target_zero_safety_minimum;
        row->entropy_before = summary->mask_entropy_before;
        row->entropy_after = summary->mask_entropy_after;
        row->stop_reason = summary->stop_reason;
}

static const char *
segment_stop_reason_to_string (itty_feed_model_segment_train_stop_reason_t stop_reason)
{
        switch (stop_reason) {
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_CONVERGED:
                return "converged";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_BATCH:
                return "rejected";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_DISTANCE_REGRESSION:
                return "rejected-distance";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_EXCESS_REGRESSION:
                return "rejected-excess";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_ZERO_SAFETY_REGRESSION:
                return "rejected-zero-safety";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_TARGET_ONE_MARGIN_LOSS:
                return "rejected-one-margin";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_NO_OBJECTIVE_DELTA:
                return "rejected-no-delta";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_TRUE_NOOP:
                return "rejected-true-noop";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_LOCAL_ACTIVATION_ONLY:
                return "rejected-local-only";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_VOTE_MOVEMENT_TIED:
                return "rejected-vote-tied";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_FUTURE_COST_IMPROVEMENT:
                return "rejected-future-cost";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_REALIZATION_MISMATCH:
                return "rejected-realization";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_BATCH_INTERACTION:
                return "rejected-interaction";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_REPLAY_GUARD:
                return "rejected-replay";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_BLOCKED_ALL_CANDIDATES:
                return "replay-blocked-all";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_SAFE_NO_CURRENT_GAIN:
                return "replay-safe-no-gain";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_PREFIX_ORDERING_FAILURE:
                return "replay-prefix-failed";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_SINGLE_CANDIDATE_CONFLICT:
                return "replay-single-candidate-conflict";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_CAPACITY_CONFLICT:
                return "replay-capacity-conflict";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_MAX_ROUNDS:
                return "max-rounds";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_REPAIRS:
                return "no-repairs";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_EFFECTIVE_CANDIDATES:
                return "no-effective-candidates";
        case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_PROGRESS:
                return "no-progress";
        }

        return "unknown";
}

static const char *
segment_target_name (size_t target_index)
{
        switch (target_index) {
        case 0:
                return "dense";
        case 1:
                return "sparse";
        case 2:
                return "mixed";
        }

        return "unknown";
}

static const char *
classify_segment_clobber (itty_feed_model_decoder_objective_t const *before_second,
                          itty_feed_model_decoder_objective_t const *after_second)
{
        if (after_second->selected_distance > before_second->selected_distance) {
                if (before_second->selected_distance == 0 &&
                    after_second->selected_distance >= ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 8)
                        return "catastrophic";
                return "distance-clobber";
        }
        if (before_second->selected_distance == 0 &&
            after_second->selected_distance >= ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 8)
                return "catastrophic";
        if (after_second->selected_distance < before_second->selected_distance)
                return "beneficial";
        if (after_second->target_one_margin < before_second->target_one_margin ||
            after_second->target_zero_safety_min < before_second->target_zero_safety_min)
                return "soft";
        if (after_second->target_one_margin > before_second->target_one_margin ||
            after_second->target_zero_safety_min > before_second->target_zero_safety_min)
                return "beneficial";
        return "neutral";
}

static void
run_segment_distinct_target_clobber_pair (const char *name,
                                          size_t      first_input_word,
                                          size_t      first_target_word,
                                          size_t      second_input_word,
                                          size_t      second_target_word,
                                          size_t      start_seed)
{
        for (size_t seed_offset = 0; seed_offset < 256; seed_offset++) {
                itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
                itty_bit_string_list_t *first_input = create_input_with_count (1,
                                                                               first_input_word);
                itty_bit_string_list_t *second_input = create_input_with_count (1,
                                                                                second_input_word);
                itty_bit_string_t *first_target = create_bit_string (first_target_word);
                itty_bit_string_t *second_target = create_bit_string (second_target_word);
                itty_feed_model_refreshed_projected_repair_options_t repair_options = {
                        .batch_size = 64,
                        .max_rounds = 32,
                        .max_layer_flips_per_batch = 128
                };
                itty_feed_model_decoder_objective_t first_before;
                itty_feed_model_decoder_objective_t first_after;
                itty_feed_model_decoder_objective_t second_before;
                itty_feed_model_decoder_objective_t second_after;
                itty_feed_model_decoder_objective_t first_after_second;
                itty_feed_model_segment_training_summary_t first_summary;
                itty_feed_model_segment_training_summary_t second_summary;
                itty_feed_model_test_segment_training_row_t first_row;
                itty_feed_model_test_segment_training_row_t second_row;
                size_t clobbered_bits;
                const char *classification;
                bool accepted_pair = false;
                size_t seed = start_seed + seed_offset;

                itty_feed_model_set_decoder (model,
                                             ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                assert (itty_feed_model_randomize_masks (model,
                                                         seed,
                                                         1,
                                                         8));

                assert (itty_feed_model_measure_decoder_objective (model,
                                                                   first_input,
                                                                   first_target,
                                                                   &first_before));
                assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                        first_input,
                                                                                        first_target,
                                                                                        &repair_options,
                                                                                        &first_summary));
                capture_segment_training_row (&first_summary,
                                              &first_row);
                assert (itty_feed_model_measure_decoder_objective (model,
                                                                   first_input,
                                                                   first_target,
                                                                   &first_after));
                if (first_after.selected_distance < first_before.selected_distance &&
                    first_row.flips > 0) {
                        assert (itty_feed_model_measure_decoder_objective (model,
                                                                           second_input,
                                                                           second_target,
                                                                           &second_before));
                        assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                                second_input,
                                                                                                second_target,
                                                                                                &repair_options,
                                                                                                &second_summary));
                        capture_segment_training_row (&second_summary,
                                                      &second_row);
                        assert (itty_feed_model_measure_decoder_objective (model,
                                                                           second_input,
                                                                           second_target,
                                                                           &second_after));
                        assert (itty_feed_model_measure_decoder_objective (model,
                                                                           first_input,
                                                                           first_target,
                                                                           &first_after_second));
                        accepted_pair = second_after.selected_distance < second_before.selected_distance &&
                                        second_row.flips > 0;
                }

                if (accepted_pair) {
                        clobbered_bits = first_after_second.selected_distance > first_after.selected_distance ?
                                         first_after_second.selected_distance - first_after.selected_distance :
                                         0;
                        classification = classify_segment_clobber (&first_after,
                                                                   &first_after_second);
                        printf ("segment distinct-target clobber %-24s "
                                "seed %zu, A %zu->%zu then %zu, B %zu->%zu, clobber %zu, "
                                "A margin %zu->%zu safety %zu->%zu, class %s, "
                                "A flips %zu stop %s, B flips %zu stop %s\n",
                                name,
                                seed,
                                first_row.initial_distance,
                                first_row.final_distance,
                                first_after_second.selected_distance,
                                second_row.initial_distance,
                                second_row.final_distance,
                                clobbered_bits,
                                first_after.target_one_margin,
                                first_after_second.target_one_margin,
                                first_after.target_zero_safety_min,
                                first_after_second.target_zero_safety_min,
                                classification,
                                first_row.flips,
                                segment_stop_reason_to_string (first_row.stop_reason),
                                second_row.flips,
                                segment_stop_reason_to_string (second_row.stop_reason));

                        itty_bit_string_free (first_target);
                        itty_bit_string_free (second_target);
                        itty_bit_string_list_free (first_input);
                        itty_bit_string_list_free (second_input);
                        itty_feed_model_free (model);
                        return;
                }

                itty_bit_string_free (first_target);
                itty_bit_string_free (second_target);
                itty_bit_string_list_free (first_input);
                itty_bit_string_list_free (second_input);
                itty_feed_model_free (model);
        }

        assert (false);
}

static void
test_itty_feed_model_segment_training_tracks_distinct_target_clobber (void)
{
        size_t dense = create_half_populated_word ();
        size_t sparse_s = (size_t) 1 << 3;
        size_t sparse_t = (size_t) 1 << 47;
        size_t mixed = create_mixed_word ();
        size_t complement_mixed = ~mixed;

        run_segment_distinct_target_clobber_pair ("dense-then-sparse",
                                                  0,
                                                  dense,
                                                  1,
                                                  sparse_s,
                                                  0xad01);
        run_segment_distinct_target_clobber_pair ("sparse-then-dense",
                                                  0,
                                                  sparse_s,
                                                  1,
                                                  dense,
                                                  0xad02);
        run_segment_distinct_target_clobber_pair ("disjoint-sparse",
                                                  0,
                                                  sparse_s,
                                                  1,
                                                  sparse_t,
                                                  0xad03);
        run_segment_distinct_target_clobber_pair ("mixed-then-complement",
                                                  0,
                                                  mixed,
                                                  1,
                                                  complement_mixed,
                                                  0xad04);
}

static char const *
final_surface_block_reason_to_string (itty_feed_model_final_surface_block_reason_t reason)
{
        switch (reason) {
        case ITTY_FEED_MODEL_FINAL_SURFACE_BLOCK_NONE:
                return "none";
        case ITTY_FEED_MODEL_FINAL_SURFACE_BLOCK_REPLAY_UNSAFE:
                return "replay-unsafe";
        case ITTY_FEED_MODEL_FINAL_SURFACE_BLOCK_DUPLICATE_FINAL_VOTE:
                return "duplicate-final-vote";
        case ITTY_FEED_MODEL_FINAL_SURFACE_BLOCK_NO_CANDIDATE_PATH:
                return "no-candidate-path";
        default:
                return "unknown";
        }
}

static char const *
output_transform_to_string (itty_feed_model_output_transform_t transform)
{
        switch (transform) {
        case ITTY_FEED_MODEL_OUTPUT_TRANSFORM_IDENTITY:
                return "identity";
        case ITTY_FEED_MODEL_OUTPUT_TRANSFORM_ODD_SWAP:
                return "odd-swap";
        case ITTY_FEED_MODEL_OUTPUT_TRANSFORM_EVEN_SWAP:
                return "even-swap";
        case ITTY_FEED_MODEL_OUTPUT_TRANSFORM_HALF_SWAP:
                return "half-swap";
        default:
                return "unknown";
        }
}

static char const *
restore_rejection_reason_to_string (itty_feed_model_restore_rejection_reason_t reason)
{
        switch (reason) {
        case ITTY_FEED_MODEL_RESTORE_REJECTION_NONE:
                return "none";
        case ITTY_FEED_MODEL_RESTORE_REJECTION_NO_FALSE_POSITIVES:
                return "no-false-positives";
        case ITTY_FEED_MODEL_RESTORE_REJECTION_NO_CANDIDATE_REPAIRS:
                return "no-candidate-repairs";
        case ITTY_FEED_MODEL_RESTORE_REJECTION_NO_CLEARABLE_SEGMENT_VOTES:
                return "no-clearable-segment-votes";
        case ITTY_FEED_MODEL_RESTORE_REJECTION_NO_MASK_PROJECTION:
                return "no-mask-projection";
        case ITTY_FEED_MODEL_RESTORE_REJECTION_NO_USEFUL_REPAIRS:
                return "no-useful-repairs";
        case ITTY_FEED_MODEL_RESTORE_REJECTION_NO_B_SAFE_REPAIRS:
                return "no-b-safe-repairs";
        case ITTY_FEED_MODEL_RESTORE_REJECTION_NO_FLIPS_ACCEPTED:
                return "no-flips-accepted";
        default:
                return "unknown";
        }
}

static char const *
restore_propagation_failure_to_string (itty_feed_model_restore_propagation_failure_t failure)
{
        switch (failure) {
        case ITTY_FEED_MODEL_RESTORE_PROPAGATION_NONE:
                return "none";
        case ITTY_FEED_MODEL_RESTORE_PROPAGATION_NO_MAJORITY_CROSSING:
                return "no-majority-crossing";
        case ITTY_FEED_MODEL_RESTORE_PROPAGATION_WRONG_POLARITY:
                return "wrong-polarity";
        case ITTY_FEED_MODEL_RESTORE_PROPAGATION_DUPLICATE_CONDENSED_MAPPING:
                return "duplicate-condensed-mapping";
        case ITTY_FEED_MODEL_RESTORE_PROPAGATION_ROTATION_OR_EXPANSION_MISMATCH:
                return "rotation-expansion-mismatch";
        case ITTY_FEED_MODEL_RESTORE_PROPAGATION_SELECTED_NODE_MISMATCH:
                return "selected-node-mismatch";
        case ITTY_FEED_MODEL_RESTORE_PROPAGATION_SEGMENT_CHANGED_DECODED_NOT_CROSSED:
                return "segment-changed-decoded-not-crossed";
        case ITTY_FEED_MODEL_RESTORE_PROPAGATION_DECODED_CROSSED_OFFSET_ELSEWHERE:
                return "decoded-crossed-offset-elsewhere";
        default:
                return "unknown";
        }
}

static void
print_replay_final_surface_feasibility (itty_feed_model_projected_repair_stats_t const *stats)
{
        fprintf (stderr,
                 "segment replay final-surface feasibility:\n");
        for (size_t trace_index = 0;
             trace_index < stats->replay_final_surface_feasibility_trace_count;
             trace_index++) {
                itty_feed_model_final_surface_feasibility_t const *trace =
                        &stats->replay_final_surface_feasibility_traces[trace_index];

                fprintf (stderr,
                         "  bit %zu need %zu safe-final %zu unsafe-final %zu shortfall %zu "
                         "dup-safe %zu dup-unsafe %zu reason %s\n",
                         trace->decoded_bit,
                         trace->needed_final_votes,
                         trace->safe_final_votes_available,
                         trace->unsafe_final_votes_available,
                         trace->safe_final_votes_shortfall,
                         trace->duplicate_safe_final_votes,
                         trace->duplicate_unsafe_final_votes,
                         final_surface_block_reason_to_string (trace->blocked_reason));
        }
}

static void
run_segment_replay_final_layer_oracle_diagnostic (void)
{
        itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
        itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
        itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
        itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
        itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
        itty_feed_model_refreshed_projected_repair_options_t first_options = {
                .batch_size = 64,
                .max_rounds = 32,
                .max_layer_flips_per_batch = 128
        };
        itty_feed_model_train_options_t oracle_options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_segment_training_summary_t first_summary;
        itty_feed_model_train_stats_t oracle_stats;
        itty_feed_model_decoder_objective_t first_before;
        itty_feed_model_decoder_objective_t first_after;
        itty_feed_model_decoder_objective_t second_before;
        itty_feed_model_decoder_objective_t second_after;
        size_t safe_steps = 0;
        size_t safe_distance = 0;
        size_t safe_deficit = 0;

        itty_feed_model_set_decoder (model,
                                     ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        assert (itty_feed_model_randomize_masks (model,
                                                 44291,
                                                 1,
                                                 8));
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &first_summary));
        assert (itty_feed_model_measure_decoder_objective (model,
                                                           first_input,
                                                           first_target,
                                                           &first_before));
        assert (itty_feed_model_measure_decoder_objective (model,
                                                           second_input,
                                                           second_target,
                                                           &second_before));

        safe_distance = second_before.selected_distance;
        safe_deficit = second_before.false_negative_vote_deficit;
        for (size_t step = 0; step < 16; step++) {
                assert (itty_feed_model_train_final_layer_with_suffix_oracle (model,
                                                                              second_input,
                                                                              second_target,
                                                                              &oracle_options,
                                                                              &oracle_stats));
                assert (itty_feed_model_measure_decoder_objective (model,
                                                                   first_input,
                                                                   first_target,
                                                                   &first_after));
                assert (itty_feed_model_measure_decoder_objective (model,
                                                                   second_input,
                                                                   second_target,
                                                                   &second_after));
                if (first_after.selected_distance != 0)
                        break;
                safe_steps++;
                safe_distance = second_after.selected_distance;
                safe_deficit = second_after.false_negative_vote_deficit;
        }

        fprintf (stderr,
                 "segment replay final-layer oracle: "
                 "A %zu->%zu B %zu->%zu B-deficit %zu->%zu safe-steps %zu oracle-flips %zu replay-safe %s\n",
                 first_before.selected_distance,
                 first_after.selected_distance,
                 second_before.selected_distance,
                 safe_distance,
                 second_before.false_negative_vote_deficit,
                 safe_deficit,
                 safe_steps,
                 oracle_stats.flips,
                 first_after.selected_distance == 0 ? "yes" : "no");

        itty_bit_string_free (first_target);
        itty_bit_string_free (second_target);
        itty_bit_string_list_free (first_input);
        itty_bit_string_list_free (second_input);
        itty_feed_model_free (model);
}

static void
run_segment_replay_final_layer_transaction_diagnostic (void)
{
        itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
        itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
        itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
        itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
        itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
        itty_feed_model_refreshed_projected_repair_options_t first_options = {
                .batch_size = 64,
                .max_rounds = 32,
                .max_layer_flips_per_batch = 128
        };
        itty_feed_model_train_options_t oracle_options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_segment_training_summary_t first_summary;
        itty_feed_model_replay_transaction_summary_t transaction = { 0 };
        itty_feed_model_replay_transaction_summary_t single_step = { 0 };

        itty_feed_model_set_decoder (model,
                                     ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        assert (itty_feed_model_randomize_masks (model,
                                                 44291,
                                                 1,
                                                 8));
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &first_summary));
        assert (itty_feed_model_measure_final_layer_replay_transaction (model,
                                                                        first_input,
                                                                        first_target,
                                                                        second_input,
                                                                        second_target,
                                                                        &oracle_options,
                                                                        16,
                                                                        16,
                                                                        &transaction));
        assert (itty_feed_model_measure_final_layer_replay_transaction (model,
                                                                        first_input,
                                                                        first_target,
                                                                        second_input,
                                                                        second_target,
                                                                        &oracle_options,
                                                                        1,
                                                                        16,
                                                                        &single_step));

        fprintf (stderr,
                 "segment replay final-layer transaction: "
                 "A %zu->%zu->%zu A-fp %zu->%zu->%zu "
                 "B %zu->%zu->%zu B-deficit %zu->%zu->%zu "
                 "B-steps %zu A-repair-steps %zu "
                 "B-flips %zu A-repair-flips %zu accepted %s\n",
                 transaction.a_distance_before,
                 transaction.a_distance_after_b,
                 transaction.a_distance_after_repair,
                 transaction.a_false_positive_excess_before,
                 transaction.a_false_positive_excess_after_b,
                 transaction.a_false_positive_excess_after_repair,
                 transaction.b_distance_before,
                 transaction.b_distance_after_b,
                 transaction.b_distance_after_repair,
                 transaction.b_false_negative_deficit_before,
                 transaction.b_false_negative_deficit_after_b,
                 transaction.b_false_negative_deficit_after_repair,
                 transaction.b_steps,
                 transaction.a_repair_steps,
                 transaction.b_flips,
                 transaction.a_repair_flips,
                 transaction.accepted ? "yes" : "no");
        fprintf (stderr,
                 "segment replay final-layer transaction-single: "
                 "A %zu->%zu->%zu B %zu->%zu->%zu "
                 "B-deficit %zu->%zu->%zu "
                 "A-restored %s B-preserved %s accepted %s\n",
                 single_step.a_distance_before,
                 single_step.a_distance_after_b,
                 single_step.a_distance_after_repair,
                 single_step.b_distance_before,
                 single_step.b_distance_after_b,
                 single_step.b_distance_after_repair,
                 single_step.b_false_negative_deficit_before,
                 single_step.b_false_negative_deficit_after_b,
                 single_step.b_false_negative_deficit_after_repair,
                 single_step.a_restored ? "yes" : "no",
                 single_step.b_preserved ? "yes" : "no",
                 single_step.accepted ? "yes" : "no");

        itty_bit_string_free (first_target);
        itty_bit_string_free (second_target);
        itty_bit_string_list_free (first_input);
        itty_bit_string_list_free (second_input);
        itty_feed_model_free (model);
}

static void
run_segment_replay_restore_failure_diagnostic (void)
{
        itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
        itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
        itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
        itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
        itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
        itty_feed_model_refreshed_projected_repair_options_t first_options = {
                .batch_size = 64,
                .max_rounds = 32,
                .max_layer_flips_per_batch = 128
        };
        itty_feed_model_train_options_t oracle_options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_segment_training_summary_t first_summary;
        itty_feed_model_restore_failure_summary_t restore_free = { 0 };
        itty_feed_model_restore_failure_summary_t restore_preserve = { 0 };

        itty_feed_model_set_decoder (model,
                                     ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        assert (itty_feed_model_randomize_masks (model,
                                                 44291,
                                                 1,
                                                 8));
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &first_summary));
        assert (itty_feed_model_measure_final_layer_restore_failure (model,
                                                                     first_input,
                                                                     first_target,
                                                                     second_input,
                                                                     second_target,
                                                                     &oracle_options,
                                                                     16,
                                                                     false,
                                                                     &restore_free));
        assert (itty_feed_model_measure_final_layer_restore_failure (model,
                                                                     first_input,
                                                                     first_target,
                                                                     second_input,
                                                                     second_target,
                                                                     &oracle_options,
                                                                     16,
                                                                     true,
                                                                     &restore_preserve));

        fprintf (stderr,
                 "segment replay restore failure free: "
                 "A %zu fp-excess %zu candidates %zu useful %zu veto %zu rejected %zu reason %s\n",
                 restore_free.a_distance_after_b,
                 restore_free.a_false_positive_excess_after_b,
                 restore_free.a_candidate_repair_count,
                 restore_free.a_useful_repair_count,
                 restore_free.a_target_zero_repair_count,
                 restore_free.a_rejected_repair_count,
                 restore_rejection_reason_to_string (restore_free.no_flip_reason));
        fprintf (stderr,
                 "segment replay restore failure preserve-B: "
                 "A %zu fp-excess %zu candidates %zu useful %zu veto %zu rejected %zu reason %s\n",
                 restore_preserve.a_distance_after_b,
                 restore_preserve.a_false_positive_excess_after_b,
                 restore_preserve.a_candidate_repair_count,
                 restore_preserve.a_useful_repair_count,
                 restore_preserve.a_target_zero_repair_count,
                 restore_preserve.a_rejected_repair_count,
                 restore_rejection_reason_to_string (restore_preserve.no_flip_reason));

        for (size_t trace_index = 0; trace_index < restore_preserve.trace_count; trace_index++) {
                itty_feed_model_restore_failure_trace_t const *trace = &restore_preserve.traces[trace_index];
                fprintf (stderr,
                         "  A-fp bit %zu target %zu decoded %zu ones %zu threshold %zu max-zero %zu excess %zu "
                         "segment-ones %zu clearable %zu final-output-bits %zu projected-condensed %zu direct-changed %zu mask-flips %zu "
                         "candidate-votes %zu b-safe %zu flips %zu final-node %zu->%zu condensed %zu->%zu segment %zu->%zu "
                         "forced %zu/%zu->%zu/%zu decoded %zu->%zu accepted %s reason %s propagation %s\n",
                         trace->decoded_bit,
                         (size_t) (trace->target_bit ? 1 : 0),
                         (size_t) (trace->current_decoded_bit ? 1 : 0),
                         trace->current_ones,
                         trace->threshold,
                         trace->max_ones_for_zero,
                         trace->excess,
                         trace->segment_votes_currently_one,
                         trace->clearable_segment_votes,
                         trace->candidate_final_output_bits,
                         trace->projected_condensed_bits,
                         (size_t) (trace->direct_candidate_changed ? 1 : 0),
                         trace->candidate_mask_flips,
                         trace->candidate_segment_votes,
                         trace->replay_safe_candidates,
                         trace->min_final_layer_mask_flips_needed,
                         trace->final_selected_node_before,
                         trace->final_selected_node_after,
                         trace->actual_final_condensed_ones_before,
                         trace->actual_final_condensed_ones_after,
                         trace->actual_final_segment_ones_before,
                         trace->actual_final_segment_ones_after,
                         trace->forced_node_distance_before,
                         trace->forced_node_false_positive_excess_before,
                         trace->forced_node_distance_after,
                         trace->forced_node_false_positive_excess_after,
                         (size_t) (trace->decoded_before ? 1 : 0),
                         (size_t) (trace->decoded_after ? 1 : 0),
                         trace->accepted ? "yes" : "no",
                         restore_rejection_reason_to_string (trace->rejection_reason),
                         restore_propagation_failure_to_string (trace->propagation_failure));
        }
        for (size_t trace_index = 0; trace_index < restore_preserve.clear_trace_count; trace_index++) {
                itty_feed_model_restore_clear_vote_trace_t const *trace = &restore_preserve.clear_traces[trace_index];
                fprintf (stderr,
                         "    clear-vote bit %zu seg %zu out-bit %zu final %zu->%zu->%zu "
                         "condensed-bit %zu condensed %zu->%zu->%zu changed %zu "
                         "mask-flips %zu majority %zu->%zu threshold %zu decoded-ones %zu->%zu decoded %zu->%zu "
                         "cleared %zu\n",
                         trace->decoded_bit,
                         trace->segment_index,
                         trace->final_output_bit,
                         (size_t) (trace->raw_final_segment_before ? 1 : 0),
                         (size_t) (trace->desired_final_segment_value ? 1 : 0),
                         (size_t) (trace->raw_final_segment_after ? 1 : 0),
                         trace->mapped_condensed_bit,
                         (size_t) (trace->condensed_before ? 1 : 0),
                         (size_t) (trace->desired_condensed_bit ? 1 : 0),
                         (size_t) (trace->condensed_after ? 1 : 0),
                         (size_t) (trace->condensed_bit_changed ? 1 : 0),
                         trace->mask_flip_count,
                         trace->majority_ones_before,
                         trace->majority_ones_after,
                         trace->majority_threshold,
                         trace->decoded_ones_before,
                         trace->decoded_ones_after,
                         (size_t) (trace->decoded_bit_before ? 1 : 0),
                         (size_t) (trace->decoded_bit_after ? 1 : 0),
                         (size_t) (trace->cleared ? 1 : 0));
        }

        itty_bit_string_free (first_target);
        itty_bit_string_free (second_target);
        itty_bit_string_list_free (first_input);
        itty_bit_string_list_free (second_input);
        itty_feed_model_free (model);
}

static void
run_segment_replay_contender_restore_diagnostic (void)
{
        itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
        itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
        itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
        itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
        itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
        itty_feed_model_refreshed_projected_repair_options_t first_options = {
                .batch_size = 64,
                .max_rounds = 32,
                .max_layer_flips_per_batch = 128
        };
        itty_feed_model_train_options_t oracle_options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_segment_training_summary_t first_summary;
        itty_feed_model_contender_restore_summary_t free_summary = { 0 };
        itty_feed_model_contender_restore_summary_t preserve_summary = { 0 };

        itty_feed_model_set_decoder (model,
                                     ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        assert (itty_feed_model_randomize_masks (model,
                                                 44291,
                                                 1,
                                                 8));
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &first_summary));
        assert (itty_feed_model_measure_final_layer_contender_restore (model,
                                                                       first_input,
                                                                       first_target,
                                                                       second_input,
                                                                       second_target,
                                                                       &oracle_options,
                                                                       16,
                                                                       false,
                                                                       &free_summary));
        assert (itty_feed_model_measure_final_layer_contender_restore (model,
                                                                       first_input,
                                                                       first_target,
                                                                       second_input,
                                                                       second_target,
                                                                       &oracle_options,
                                                                       16,
                                                                       true,
                                                                       &preserve_summary));

        fprintf (stderr,
                 "segment replay contender restore free: "
                 "bit %zu A-after-B %zu/%zu node %zu->%zu forced %zu/%zu->%zu/%zu "
                 "contender %zu A %zu/%zu B %zu/%zu flips %zu useful %s b-safe %s "
                 "strict %s dist %s progress %s no-reg %s accepted %s\n",
                 free_summary.decoded_bit,
                 free_summary.a_distance_after_b,
                 free_summary.a_false_positive_excess_after_b,
                 free_summary.selected_node_before,
                 free_summary.selected_node_after_first,
                 free_summary.forced_node_distance_before,
                 free_summary.forced_node_false_positive_excess_before,
                 free_summary.forced_node_distance_after_first,
                 free_summary.forced_node_false_positive_excess_after_first,
                 free_summary.contender_node,
                 free_summary.a_distance_after_contender,
                 free_summary.a_false_positive_excess_after_contender,
                 free_summary.b_distance_after_contender,
                 free_summary.b_false_negative_deficit_after_contender,
                 free_summary.total_flips,
                 free_summary.contender_useful ? "yes" : "no",
                 free_summary.contender_b_safe ? "yes" : "no",
                 free_summary.b_strict_preserved ? "yes" : "no",
                 free_summary.b_distance_preserved ? "yes" : "no",
                 free_summary.b_progress_preserved ? "yes" : "no",
                 free_summary.b_no_regression ? "yes" : "no",
                 free_summary.contender_accepted ? "yes" : "no");
        for (size_t trace_index = 0; trace_index < free_summary.clear_set_trace_count; trace_index++) {
                itty_feed_model_contender_clear_set_trace_t const *trace =
                        &free_summary.clear_set_traces[trace_index];
                fprintf (stderr,
                         "contender clear set %zu: votes {",
                         trace_index);
                for (size_t vote_index = 0; vote_index < trace->clear_vote_count; vote_index++)
                        fprintf (stderr,
                                 "%s%zu",
                                 vote_index == 0 ? "" : ", ",
                                 trace->segment_indices[vote_index]);
                fprintf (stderr,
                         "} A %zu/%zu B %zu/%zu B-fp %zu flips %zu node %zu margin %zu "
                         "strict %s dist %s progress %s no-reg %s best %s\n",
                         trace->a_distance_after_restore,
                         trace->a_false_positive_excess_after_restore,
                         trace->b_distance_after_restore,
                         trace->b_false_negative_deficit_after_restore,
                         trace->b_false_positive_excess_after_restore,
                         trace->total_flips,
                         trace->selected_node_after_restore,
                         trace->selection_margin_after_restore,
                         trace->strict_preserved ? "yes" : "no",
                         trace->distance_preserved ? "yes" : "no",
                         trace->progress_preserved ? "yes" : "no",
                         trace->no_regression ? "yes" : "no",
                         trace->chosen_best ? "yes" : "no");
        }
        fprintf (stderr,
                 "segment replay contender restore preserve-B: "
                 "bit %zu A-after-B %zu/%zu node %zu->%zu forced %zu/%zu->%zu/%zu "
                 "contender %zu A %zu/%zu B %zu/%zu flips %zu useful %s b-safe %s "
                 "strict %s dist %s progress %s no-reg %s accepted %s\n",
                 preserve_summary.decoded_bit,
                 preserve_summary.a_distance_after_b,
                 preserve_summary.a_false_positive_excess_after_b,
                 preserve_summary.selected_node_before,
                 preserve_summary.selected_node_after_first,
                 preserve_summary.forced_node_distance_before,
                 preserve_summary.forced_node_false_positive_excess_before,
                 preserve_summary.forced_node_distance_after_first,
                 preserve_summary.forced_node_false_positive_excess_after_first,
                 preserve_summary.contender_node,
                 preserve_summary.a_distance_after_contender,
                 preserve_summary.a_false_positive_excess_after_contender,
                 preserve_summary.b_distance_after_contender,
                 preserve_summary.b_false_negative_deficit_after_contender,
                 preserve_summary.total_flips,
                 preserve_summary.contender_useful ? "yes" : "no",
                 preserve_summary.contender_b_safe ? "yes" : "no",
                 preserve_summary.b_strict_preserved ? "yes" : "no",
                 preserve_summary.b_distance_preserved ? "yes" : "no",
                 preserve_summary.b_progress_preserved ? "yes" : "no",
                 preserve_summary.b_no_regression ? "yes" : "no",
                 preserve_summary.contender_accepted ? "yes" : "no");

        itty_bit_string_free (first_target);
        itty_bit_string_free (second_target);
        itty_bit_string_list_free (first_input);
        itty_bit_string_list_free (second_input);
        itty_feed_model_free (model);
}

static void
run_segment_replay_transaction_scaffold_diagnostic (void)
{
        for (size_t finish_margin = 0; finish_margin <= 3; finish_margin++) {
                itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
                itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
                itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
                itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
                itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
                itty_feed_model_refreshed_projected_repair_options_t first_options = {
                        .batch_size = 64,
                        .max_rounds = 32,
                        .max_layer_flips_per_batch = 128
                };
                itty_feed_model_train_options_t oracle_options = {
                        .max_flips = 8,
                        .finish_margin = finish_margin,
                        .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                        .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
                };
                itty_feed_model_segment_training_summary_t first_summary;
                itty_feed_model_transaction_scaffold_round_t trajectory[ITTY_FEED_MODEL_TRANSACTION_SCAFFOLD_TRAJECTORY_LIMIT] = { 0 };
                itty_feed_model_transaction_scaffold_summary_t scaffold = { 0 };

                itty_feed_model_set_decoder (model,
                                             ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                assert (itty_feed_model_randomize_masks (model,
                                                         44291,
                                                         1,
                                                         8));
                assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                        first_input,
                                                                                        first_target,
                                                                                        &first_options,
                                                                                        &first_summary));
                assert (itty_feed_model_train_final_layer_transaction_scaffold (model,
                                                                                first_input,
                                                                                first_target,
                                                                                second_input,
                                                                                second_target,
                                                                                &oracle_options,
                                                                                8,
                                                                                trajectory,
                                                                                ITTY_FEED_MODEL_TRANSACTION_SCAFFOLD_TRAJECTORY_LIMIT,
                                                                                &scaffold));

                fprintf (stderr,
                 "segment replay transaction scaffold margin %zu: "
                 "rounds %zu/%zu A %zu/%zu->%zu/%zu B %zu/%zu->%zu/%zu "
                 "frontier min %zu->%zu d1 %zu->%zu d2 %zu->%zu cheap %zu->%zu top4 %zu->%zu "
                 "b-flips %zu restore-flips %zu strict %zu dist %zu progress %zu no-reg %zu frontier %zu "
                 "finish cand %zu pre-dist %zu pre-def %zu pre-margin %zu clobber-A %zu restore-A %zu "
                 "post-dist %zu post-progress %zu post-margin %zu rej-no-pre %zu rej-A %zu rej-B %zu "
                 "finish-complete %zu A-c0fp %zu A-c1fn %zu A-switch %zu restore-avail %zu erase-B %zu restore-fail %zu "
                 "distance-improved %s deficit-improved %s frontier-improved %s A-solved %s\n",
                 finish_margin,
                 scaffold.rounds_accepted,
                 scaffold.rounds_attempted,
                 scaffold.a_distance_before,
                 scaffold.a_false_positive_excess_before,
                 scaffold.a_distance_after,
                 scaffold.a_false_positive_excess_after,
                 scaffold.b_distance_before,
                 scaffold.b_false_negative_deficit_before,
                 scaffold.b_distance_after,
                 scaffold.b_false_negative_deficit_after,
                 scaffold.b_min_deficit_before,
                 scaffold.b_min_deficit_after,
                 scaffold.b_deficit_one_bits_before,
                 scaffold.b_deficit_one_bits_after,
                 scaffold.b_deficit_two_bits_before,
                 scaffold.b_deficit_two_bits_after,
                 scaffold.b_cheapest_completion_cost_before,
                 scaffold.b_cheapest_completion_cost_after,
                 scaffold.b_top_k_completion_cost_before,
                 scaffold.b_top_k_completion_cost_after,
                 scaffold.total_b_flips,
                 scaffold.total_restore_flips,
                 scaffold.strict_preserved_rounds,
                 scaffold.distance_preserved_rounds,
                 scaffold.progress_preserved_rounds,
                 scaffold.no_regression_rounds,
                 scaffold.frontier_improved_rounds,
                 scaffold.finish_candidates,
                 scaffold.finish_pre_restore_distance_helpful,
                 scaffold.finish_pre_restore_deficit_helpful,
                 scaffold.finish_pre_restore_margin_met,
                 scaffold.finish_clobbers_a,
                 scaffold.finish_restores_a,
                 scaffold.finish_post_restore_distance_preserved,
                 scaffold.finish_post_restore_progress_preserved,
                 scaffold.finish_post_restore_margin_preserved,
                 scaffold.finish_rejected_no_pre_restore_gain,
                 scaffold.finish_rejected_a_not_restored,
                 scaffold.finish_rejected_b_gain_lost,
                 scaffold.finish_complete_b_before_restore,
                 scaffold.finish_create_a_correct_zero_to_false_positive,
                 scaffold.finish_create_a_correct_one_to_false_negative,
                 scaffold.finish_switch_a_selected_node,
                 scaffold.finish_contender_restore_available,
                 scaffold.finish_restore_erases_b,
                 scaffold.finish_restore_fails,
                 scaffold.b_distance_improved ? "yes" : "no",
                 scaffold.b_deficit_improved ? "yes" : "no",
                 scaffold.b_frontier_improved ? "yes" : "no",
                 scaffold.a_remains_solved ? "yes" : "no");
                fprintf (stderr,
                 "segment replay finish candidates:\n");
                for (size_t trace_index = 0;
             trace_index < scaffold.finish_trace_count &&
             trace_index < ITTY_FEED_MODEL_FINISH_CANDIDATE_TRACE_LIMIT;
             trace_index++) {
                        itty_feed_model_finish_candidate_trace_t const *trace = &scaffold.finish_traces[trace_index];
                        char const *reject_reason = trace->rejected_no_pre_restore_gain ? "no-pre-restore-gain" :
                                                    trace->rejected_a_not_restored ? "a-not-restored" :
                                                    trace->rejected_b_gain_lost ? "b-gain-lost" :
                                                    trace->restore_failed ? "restore-failed" :
                                                    trace->chosen_best ? "chosen-best" : "kept-out";
                        fprintf (stderr,
                         "  bit %zu out %zu "
                         "B %zu/%zu/%zu -> %zu/%zu/%zu "
                         "A %zu/%zu/%zu node %zu->%zu damage %zu/%zu "
                         "family1 %s finish-margin %zu met %s restore contender %zu avail %s failed %s "
                         "A %zu/%zu B %zu/%zu/%zu margin-after %s reject %s\n",
                         trace->decoded_bit,
                         trace->final_output_bit,
                         trace->b_distance_before,
                         trace->b_deficit_before,
                         trace->b_target_one_margin_before,
                         trace->b_distance_after_finish,
                         trace->b_deficit_after_finish,
                         trace->b_target_one_margin_after_finish,
                         trace->a_distance_after_finish,
                         trace->a_excess_after_finish,
                         trace->a_deficit_after_finish,
                         trace->a_selected_node_before,
                         trace->a_selected_node_after_finish,
                         trace->a_damaged_correct_zero_to_false_positive_bits,
                         trace->a_damaged_correct_one_to_false_negative_bits,
                         trace->family_one_finish ? "yes" : "no",
                         trace->finish_margin_required,
                         trace->finish_margin_met_before_restore ? "yes" : "no",
                         trace->restore_contender_node,
                         trace->restore_available ? "yes" : "no",
                         trace->restore_failed ? "yes" : "no",
                         trace->a_distance_after_restore,
                         trace->a_excess_after_restore,
                         trace->b_distance_after_restore,
                         trace->b_deficit_after_restore,
                         trace->b_target_one_margin_after_restore,
                         trace->finish_margin_met_after_restore ? "yes" : "no",
                         reject_reason);
                        if (trace->family_one_finish) {
                                fprintf (stderr,
                                 "    family1-restore cand %zu useful %zu zero %zu rejected %zu reason %d "
                                 "bit %zu ones %zu threshold %zu max-zero %zu excess %zu clearable %zu mask-flips %zu "
                                 "damage-set bits %zu node %zu->%zu avail %s excess-reduced %s restored %s dist %s progress %s flips %zu B %zu/%zu "
                                 "repl cand %zu safe %zu dist %zu progress %zu avail %s out %zu cond %zu flips %zu "
                                 "finish-out %zu finish-cond %zu finish-mask %zu overlap out %zu cond %zu mask %zu dir %zu "
                                 "sets %zu excess-red %zu restoring %zu restore-only %zu restore-progress %zu restore-distance %zu "
                                 "dist-pres %zu progress-pres %zu best %zu\n",
                                 trace->restore_a_candidate_repair_count,
                                 trace->restore_a_useful_repair_count,
                                 trace->restore_a_target_zero_repair_count,
                                 trace->restore_a_rejected_repair_count,
                                 (int) trace->restore_a_no_flip_reason,
                                 trace->restore_audit_decoded_bit,
                                 trace->restore_audit_ones,
                                 trace->restore_audit_threshold,
                                 trace->restore_audit_max_ones_for_zero,
                                 trace->restore_audit_excess,
                                 trace->restore_audit_clearable_votes,
                                 trace->restore_audit_mask_flips,
                                 trace->family_one_damaged_bits,
                                 trace->damage_set_restore_selected_node_before,
                                 trace->damage_set_restore_selected_node_after,
                                 trace->damage_set_restore_available ? "yes" : "no",
                                 trace->damage_set_restore_excess_reduced ? "yes" : "no",
                                 trace->damage_set_restore_restored_a ? "yes" : "no",
                                 trace->damage_set_restore_distance_preserved ? "yes" : "no",
                                 trace->damage_set_restore_progress_preserved ? "yes" : "no",
                                 trace->damage_set_restore_flips,
                                 trace->b_distance_after_restore,
                                 trace->b_deficit_after_restore,
                                 trace->replacement_candidate_count,
                                 trace->replacement_a_safe_count,
                                 trace->replacement_distance_helpful_count,
                                 trace->replacement_progress_helpful_count,
                                 trace->replacement_available ? "yes" : "no",
                                 trace->replacement_output_bit,
                                 trace->replacement_condensed_bit,
                                 trace->replacement_flips,
                                 trace->final_output_bit,
                                 trace->finish_condensed_bit,
                                 trace->finish_mask_flip_count,
                                 trace->overlap_final_output_bits,
                                 trace->overlap_condensed_bits,
                                 trace->overlap_mask_flip_locations,
                                 trace->overlap_mask_flip_directions,
                                 trace->family1_clear_set_count,
                                 trace->family1_excess_reducing_clear_set_count,
                                 trace->family1_restoring_clear_set_count,
                                 trace->family1_restore_only_clear_set_count,
                                 trace->family1_restore_progress_clear_set_count,
                                 trace->family1_restore_distance_clear_set_count,
                                 trace->family1_distance_preserving_clear_set_count,
                                 trace->family1_progress_preserving_clear_set_count,
                                 trace->family1_best_clear_set_index);
                                for (size_t clear_index = 0;
                             clear_index < trace->family1_clear_set_count &&
                             clear_index < ITTY_FEED_MODEL_FINISH_CLEAR_SET_TRACE_LIMIT;
                             clear_index++) {
                                        itty_feed_model_contender_clear_set_trace_t const *clear =
                                                &trace->family1_clear_set_traces[clear_index];
                                        fprintf (stderr,
                                         "      clear-set %zu votes {",
                                         clear_index);
                                        for (size_t vote_index = 0; vote_index < clear->clear_vote_count; vote_index++) {
                                                fprintf (stderr,
                                                 "%s%zu",
                                                 vote_index == 0 ? "" : ", ",
                                                 clear->segment_indices[vote_index]);
                                        }
                                        fprintf (stderr,
                                         "} A %zu/%zu B %zu/%zu excess-red %s restored %s node %zu->%zu preserved %s strict %s dist %s progress %s no-reg %s flips %zu overlap out %zu cond %zu mask %zu dir %zu margin %zu%s\n",
                                         clear->a_distance_after_restore,
                                         clear->a_false_positive_excess_after_restore,
                                         clear->b_distance_after_restore,
                                         clear->b_false_negative_deficit_after_restore,
                                         clear->a_excess_reduced ? "yes" : "no",
                                         clear->a_restored ? "yes" : "no",
                                         clear->selected_node_before_restore,
                                         clear->selected_node_after_restore,
                                         clear->selected_node_preserved ? "yes" : "no",
                                         clear->strict_preserved ? "yes" : "no",
                                         clear->distance_preserved ? "yes" : "no",
                                         clear->progress_preserved ? "yes" : "no",
                                         clear->no_regression ? "yes" : "no",
                                         clear->total_flips,
                                         clear->overlap_final_output_bits,
                                         clear->overlap_condensed_bits,
                                         clear->overlap_mask_flip_locations,
                                         clear->overlap_mask_flip_directions,
                                         clear->selection_margin_after_restore,
                                         clear_index == trace->family1_best_clear_set_index ? " best" : "");
                                }
                        }
                }
                fprintf (stderr,
                 "segment replay scaffold trajectory:\n");
                for (size_t round = 0; round < scaffold.rounds_attempted &&
                               round < ITTY_FEED_MODEL_TRANSACTION_SCAFFOLD_TRAJECTORY_LIMIT;
             round++) {
                        itty_feed_model_transaction_scaffold_round_t const *row = &trajectory[round];
                        fprintf (stderr,
                         "  round %zu %s restore %s finish1 %s(%zu) A %zu/%zu->%zu/%zu "
                         "B %zu/%zu->%zu/%zu frontier %zu/%zu/%zu->%zu/%zu/%zu "
                         "cheap %zu->%zu top4 %zu->%zu flips %zu+%zu "
                         "strict %s dist %s progress %s no-reg %s frontier %s\n",
                         row->round_index,
                         row->accepted ? "accepted" : "rejected",
                         row->used_restore ? "yes" : "no",
                         row->used_finish_nearest_bit ? "yes" : "no",
                         row->finish_nearest_threshold,
                         row->a_distance_before,
                         row->a_false_positive_excess_before,
                         row->a_distance_after,
                         row->a_false_positive_excess_after,
                         row->b_distance_before,
                         row->b_false_negative_deficit_before,
                         row->b_distance_after,
                         row->b_false_negative_deficit_after,
                         row->b_min_deficit_before,
                         row->b_deficit_one_bits_before,
                         row->b_deficit_two_bits_before,
                         row->b_min_deficit_after,
                         row->b_deficit_one_bits_after,
                         row->b_deficit_two_bits_after,
                         row->b_cheapest_completion_cost_before,
                         row->b_cheapest_completion_cost_after,
                         row->b_top_k_completion_cost_before,
                         row->b_top_k_completion_cost_after,
                         row->b_flips,
                         row->restore_flips,
                         row->strict_preserved ? "yes" : "no",
                         row->distance_preserved ? "yes" : "no",
                         row->progress_preserved ? "yes" : "no",
                         row->no_regression ? "yes" : "no",
                         row->frontier_improved ? "yes" : "no");
                }

                itty_bit_string_free (first_target);
                itty_bit_string_free (second_target);
                itty_bit_string_list_free (first_input);
                itty_bit_string_list_free (second_input);
                itty_feed_model_free (model);
        }
}

static void
test_itty_feed_model_segment_replay_guard_blocks_solved_example_clobber (void)
{
        itty_feed_model_t *model = itty_feed_model_new (2, 2, 1, 1);
        itty_bit_string_list_t *first_input = create_input_with_count (1,
                                                                       0);
        itty_bit_string_list_t *second_input = create_input_with_count (1,
                                                                        1);
        itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
        itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
        itty_feed_model_replay_example_t replay_examples[] = {
                {
                        .input = first_input,
                        .target = first_target
                }
        };
        itty_feed_model_refreshed_projected_repair_options_t first_options = {
                .batch_size = 64,
                .max_rounds = 32,
                .max_layer_flips_per_batch = 128
        };
        itty_feed_model_refreshed_projected_repair_options_t second_options = {
                .batch_size = 64,
                .max_rounds = 1,
                .max_layer_flips_per_batch = 128,
                .top_k_segment_vote_alternatives = 16,
                .replay_zero_protection_penalty = 1024,
                .replay_one_protection_penalty = 1024,
                .replay_taboo_flip_penalty = 1024,
                .replay_examples = replay_examples,
                .replay_example_count = 1,
                .strict_replay_guard = true,
                .strict_replay_taboo_rejection = true,
                .replay_safe_quota_complete_only = true,
                .reserve_replay_protected_zero_votes = true
        };
        itty_feed_model_segment_training_summary_t first_summary;
        itty_feed_model_segment_training_summary_t second_summary;
        itty_feed_model_decoder_objective_t first_after_first;
        itty_feed_model_decoder_objective_t first_after_second;
        itty_feed_model_decoder_objective_t second_before;
        itty_feed_model_decoder_objective_t second_after;
        itty_feed_model_segment_node_selection_summary_t first_selection_after_second;
        itty_feed_model_segment_node_selection_summary_t second_selection_after_second;
        itty_feed_model_segment_node_polarity_summary_t first_polarity_after_second;
        itty_feed_model_segment_node_polarity_summary_t second_polarity_after_second;
        size_t scaffold_rounds = 0;
        size_t scaffold_threshold_crossings = 0;
        size_t total_replay_rejected_batches = 0;
        size_t total_replay_bisections = 0;
        size_t total_replay_examples_scored = 0;
        size_t total_replay_safe_candidates = 0;
        size_t total_replay_unsafe_candidates = 0;
        size_t total_replay_best_candidate_unsafe = 0;
        size_t total_replay_safe_strict_candidates = 0;
        size_t total_replay_safe_deficit_candidates = 0;
        size_t total_replay_safe_frontier_candidates = 0;
        size_t total_replay_safe_vote_candidates = 0;
        size_t total_replay_safe_noop_candidates = 0;
        size_t total_replay_safe_irrelevant_candidates = 0;
        size_t total_replay_safe_quota_feasible_bits = 0;
        size_t total_replay_safe_quota_selected_bits = 0;
        size_t total_replay_safe_quota_accepted_bits = 0;
        size_t total_replay_safe_quota_completed_bits = 0;
        size_t total_replay_safe_quota_local_realized_bits = 0;
        size_t total_replay_safe_quota_local_crossed_bits = 0;
        size_t total_replay_safe_quota_net_positive_bits = 0;
        size_t total_replay_safe_quota_cancelled_bits = 0;
        size_t total_replay_safe_quota_final_vote_reached = 0;
        size_t total_replay_safe_quota_final_vote_changed = 0;
        size_t total_replay_safe_quota_final_vote_crossed = 0;
        size_t total_replay_safe_quota_decoded_vote_changed = 0;
        size_t total_replay_safe_quota_decoded_threshold_crossed = 0;
        size_t total_replay_safe_quota_lost_at_final_majority = 0;
        size_t total_replay_safe_quota_lost_at_final_rotation_or_expansion = 0;
        size_t total_replay_safe_quota_lost_at_unselected_final_node = 0;
        size_t total_replay_safe_quota_lost_at_segment_vote_margin = 0;
        size_t total_replay_safe_quota_lost_due_to_duplicate_final_segment = 0;
        size_t total_replay_safe_quota_local_votes_selected = 0;
        size_t total_replay_safe_quota_unique_local_votes_changed = 0;
        size_t total_replay_safe_quota_unique_final_condensed_bits_affected = 0;
        size_t total_replay_safe_quota_unique_final_segment_votes_affected = 0;
        size_t total_replay_safe_quota_unique_decoded_bits_affected = 0;
        size_t total_replay_safe_quota_distance_flips = 0;
        size_t total_replay_safe_quota_rejected_local_only = 0;
        size_t total_replay_safe_quota_blocked_bits = 0;
        size_t total_replay_safe_quota_complete_candidates = 0;
        size_t total_replay_safe_quota_incomplete_candidates = 0;
        size_t total_replay_direct_protected_zero_hit_candidates = 0;
        size_t total_replay_reserved_zero_votes = 0;
        size_t total_replay_realization_collateral_false_positive_candidates = 0;
        size_t total_replay_sensitive_mask_flips = 0;
        size_t total_replay_safe_mask_flips = 0;
        size_t total_replay_false_positive_mask_flips = 0;
        size_t total_replay_false_negative_mask_flips = 0;
        size_t total_replay_margin_or_safety_weakening_mask_flips = 0;
        size_t total_replay_collateral_cost = 0;
        size_t total_replay_decomposed_candidates = 0;
        size_t total_replay_decomposed_mask_flips = 0;
        size_t total_replay_decomposed_unsafe_mask_flips = 0;
        size_t total_replay_one_bad_flip_candidates = 0;
        size_t total_replay_mostly_unsafe_candidates = 0;
        size_t total_replay_alternate_mask_flips = 0;
        size_t total_replay_alternate_unsafe_mask_flips = 0;
        size_t total_replay_alternate_collateral_cost = 0;
        size_t total_replay_alternate_better_candidates = 0;
        size_t total_replay_bad_flip_unique = 0;
        size_t total_replay_bad_flip_top_frequency = 0;
        size_t total_replay_bad_flip_top_harmless_uses = 0;
        size_t total_replay_bad_flip_top_damaged_bits = 0;
        size_t total_replay_bad_flip_top_helped_decoded_bit = 0;
        size_t total_replay_bad_flip_top_layer = 0;
        size_t total_replay_bad_flip_top_node = 0;
        size_t total_replay_bad_flip_top_input = 0;
        size_t total_replay_bad_flip_top_bit = 0;
        bool total_replay_bad_flip_top_value = false;
        size_t total_replay_taboo_vote_candidates = 0;
        size_t total_replay_taboo_mask_flips = 0;
        size_t total_replay_taboo_penalty_total = 0;
        size_t total_replay_taboo_rejected_vote_candidates = 0;
        size_t total_replay_minus_one_bad_candidates = 0;
        size_t total_replay_minus_one_bad_safe_candidates = 0;
        size_t total_replay_minus_one_bad_deficit_candidates = 0;
        size_t total_replay_minus_one_bad_strict_candidates = 0;
        size_t total_unsafe_correct_zero_to_false_positive_bits = 0;
        size_t total_unsafe_correct_one_to_false_negative_bits = 0;
        size_t total_unsafe_false_positive_excess_regressions = 0;
        size_t total_unsafe_target_zero_safety_regressions = 0;
        size_t total_unsafe_selected_node_switches = 0;
        size_t total_unsafe_best_decoded_node_switches = 0;
        size_t total_rejected_correct_one_to_false_negative_bits = 0;
        size_t total_replay_unchanged_correct_bits = 0;
        size_t total_current_false_negative_to_correct_one_bits = 0;
        size_t total_current_correct_one_to_false_negative_bits = 0;
        size_t total_current_false_positive_to_correct_zero_bits = 0;
        size_t total_current_correct_zero_to_false_positive_bits = 0;
        size_t total_current_false_negative_to_false_positive_bits = 0;
        size_t total_current_false_positive_to_false_negative_bits = 0;
        size_t total_current_unchanged_wrong_bits = 0;
        size_t total_current_unchanged_correct_bits = 0;
        size_t total_current_helpful_decoded_bits = 0;
        size_t total_current_harmed_decoded_bits = 0;
        size_t total_current_neutral_decoded_bits = 0;
        size_t total_current_candidate_net_positive = 0;
        size_t total_current_candidate_net_zero = 0;
        size_t total_current_candidate_net_negative = 0;
        size_t total_current_batch_cancel_target_one_loss = 0;
        size_t total_current_batch_cancel_target_zero_loss = 0;
        size_t total_current_batch_cancel_selected_node_change = 0;
        size_t total_current_batch_cancel_duplicate_or_overlap_side_effect = 0;

        itty_feed_model_set_decoder (model,
                                     ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        assert (itty_feed_model_randomize_masks (model,
                                                 44291,
                                                 1,
                                                 8));
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &first_summary));
        assert (itty_feed_model_measure_decoder_objective (model,
                                                           first_input,
                                                           first_target,
                                                           &first_after_first));
        assert (first_after_first.selected_distance == 0);
        assert (itty_feed_model_measure_decoder_objective (model,
                                                           second_input,
                                                           second_target,
                                                           &second_before));

        fprintf (stderr,
                 "segment replay scaffold trajectory:\n");
        for (size_t scaffold_round = 0; scaffold_round < 8; scaffold_round++) {
                itty_feed_model_decoder_objective_t first_before_round;
                itty_feed_model_decoder_objective_t first_after_round;
                itty_feed_model_decoder_objective_t second_before_round;
                itty_feed_model_decoder_objective_t second_after_round;

                assert (itty_feed_model_measure_decoder_objective (model,
                                                                   first_input,
                                                                   first_target,
                                                                   &first_before_round));
                assert (itty_feed_model_measure_decoder_objective (model,
                                                                   second_input,
                                                                   second_target,
                                                                   &second_before_round));
                assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                        second_input,
                                                                                        second_target,
                                                                                        &second_options,
                                                                                        &second_summary));
                assert (itty_feed_model_measure_decoder_objective (model,
                                                                   first_input,
                                                                   first_target,
                                                                   &first_after_round));
                assert (itty_feed_model_measure_decoder_objective (model,
                                                                   second_input,
                                                                   second_target,
                                                                   &second_after_round));

                size_t threshold_crossings = second_before_round.selected_distance > second_after_round.selected_distance ?
                                             second_before_round.selected_distance - second_after_round.selected_distance :
                                             0;
                scaffold_threshold_crossings += threshold_crossings;
                scaffold_rounds++;
                total_replay_rejected_batches += second_summary.training.replay_rejected_batches;
                total_replay_bisections += second_summary.training.replay_bisections;
                total_replay_examples_scored += second_summary.training.replay_examples_scored;
                total_replay_safe_candidates += second_summary.training.projected.replay_safe_candidates;
                total_replay_unsafe_candidates += second_summary.training.projected.replay_unsafe_candidates;
                total_replay_best_candidate_unsafe += second_summary.training.projected.replay_best_candidate_unsafe;
                total_replay_safe_strict_candidates += second_summary.training.projected.replay_safe_strict_distance_candidates;
                total_replay_safe_deficit_candidates += second_summary.training.projected.replay_safe_deficit_candidates;
                total_replay_safe_frontier_candidates += second_summary.training.projected.replay_safe_frontier_candidates;
                total_replay_safe_vote_candidates += second_summary.training.projected.replay_safe_vote_movement_candidates;
                total_replay_safe_noop_candidates += second_summary.training.projected.replay_safe_noop_candidates;
                total_replay_safe_irrelevant_candidates += second_summary.training.projected.replay_safe_irrelevant_candidates;
                total_replay_safe_quota_feasible_bits += second_summary.training.projected.replay_safe_quota_feasible_decoded_bits;
                total_replay_safe_quota_selected_bits += second_summary.training.projected.replay_safe_quota_selected_decoded_bits;
                total_replay_safe_quota_accepted_bits += second_summary.training.projected.replay_safe_quota_accepted_decoded_bits;
                total_replay_safe_quota_completed_bits += second_summary.training.projected.replay_safe_quota_completed_decoded_bits;
                total_replay_safe_quota_local_realized_bits += second_summary.training.projected.replay_safe_quota_local_realized_decoded_bits;
                total_replay_safe_quota_local_crossed_bits += second_summary.training.projected.replay_safe_quota_local_crossed_decoded_bits;
                total_replay_safe_quota_net_positive_bits += second_summary.training.projected.replay_safe_quota_net_positive_decoded_bits;
                total_replay_safe_quota_cancelled_bits += second_summary.training.projected.replay_safe_quota_cancelled_decoded_bits;
                total_replay_safe_quota_final_vote_reached += second_summary.training.projected.replay_safe_quota_final_vote_reached;
                total_replay_safe_quota_final_vote_changed += second_summary.training.projected.replay_safe_quota_final_vote_changed;
                total_replay_safe_quota_final_vote_crossed += second_summary.training.projected.replay_safe_quota_final_vote_crossed;
                total_replay_safe_quota_decoded_vote_changed += second_summary.training.projected.replay_safe_quota_decoded_vote_changed;
                total_replay_safe_quota_decoded_threshold_crossed += second_summary.training.projected.replay_safe_quota_decoded_threshold_crossed;
                total_replay_safe_quota_lost_at_final_majority += second_summary.training.projected.replay_safe_quota_lost_at_final_majority;
                total_replay_safe_quota_lost_at_final_rotation_or_expansion += second_summary.training.projected.replay_safe_quota_lost_at_final_rotation_or_expansion;
                total_replay_safe_quota_lost_at_unselected_final_node += second_summary.training.projected.replay_safe_quota_lost_at_unselected_final_node;
                total_replay_safe_quota_lost_at_segment_vote_margin += second_summary.training.projected.replay_safe_quota_lost_at_segment_vote_margin;
                total_replay_safe_quota_lost_due_to_duplicate_final_segment += second_summary.training.projected.replay_safe_quota_lost_due_to_duplicate_final_segment;
                total_replay_safe_quota_local_votes_selected += second_summary.training.projected.replay_safe_quota_local_votes_selected;
                total_replay_safe_quota_unique_local_votes_changed += second_summary.training.projected.replay_safe_quota_unique_local_votes_changed;
                total_replay_safe_quota_unique_final_condensed_bits_affected += second_summary.training.projected.replay_safe_quota_unique_final_condensed_bits_affected;
                total_replay_safe_quota_unique_final_segment_votes_affected += second_summary.training.projected.replay_safe_quota_unique_final_segment_votes_affected;
                total_replay_safe_quota_unique_decoded_bits_affected += second_summary.training.projected.replay_safe_quota_unique_decoded_bits_affected;
                total_replay_safe_quota_distance_flips += second_summary.training.projected.replay_safe_quota_distance_flip_decoded_bits;
                total_replay_safe_quota_rejected_local_only += second_summary.training.projected.replay_safe_quota_rejected_local_only_decoded_bits;
                total_replay_safe_quota_blocked_bits += second_summary.training.projected.replay_safe_quota_blocked_decoded_bits;
                total_replay_safe_quota_complete_candidates += second_summary.training.projected.replay_safe_quota_complete_candidates;
                total_replay_safe_quota_incomplete_candidates += second_summary.training.projected.replay_safe_quota_incomplete_candidates;
                total_replay_direct_protected_zero_hit_candidates += second_summary.training.projected.replay_direct_protected_zero_hit_candidates;
                total_replay_reserved_zero_votes += second_summary.training.projected.replay_reserved_zero_votes;
                total_replay_realization_collateral_false_positive_candidates += second_summary.training.projected.replay_realization_collateral_false_positive_candidates;
                total_replay_sensitive_mask_flips += second_summary.training.projected.replay_sensitive_mask_flips;
                total_replay_safe_mask_flips += second_summary.training.projected.replay_safe_mask_flips;
                total_replay_false_positive_mask_flips += second_summary.training.projected.replay_false_positive_mask_flips;
                total_replay_false_negative_mask_flips += second_summary.training.projected.replay_false_negative_mask_flips;
                total_replay_margin_or_safety_weakening_mask_flips += second_summary.training.projected.replay_margin_or_safety_weakening_mask_flips;
                total_replay_collateral_cost += second_summary.training.projected.replay_collateral_cost;
                total_replay_decomposed_candidates += second_summary.training.projected.replay_decomposed_candidates;
                total_replay_decomposed_mask_flips += second_summary.training.projected.replay_decomposed_mask_flips;
                total_replay_decomposed_unsafe_mask_flips += second_summary.training.projected.replay_decomposed_unsafe_mask_flips;
                total_replay_one_bad_flip_candidates += second_summary.training.projected.replay_one_bad_flip_candidates;
                total_replay_mostly_unsafe_candidates += second_summary.training.projected.replay_mostly_unsafe_candidates;
                total_replay_alternate_mask_flips += second_summary.training.projected.replay_alternate_mask_flips;
                total_replay_alternate_unsafe_mask_flips += second_summary.training.projected.replay_alternate_unsafe_mask_flips;
                total_replay_alternate_collateral_cost += second_summary.training.projected.replay_alternate_collateral_cost;
                total_replay_alternate_better_candidates += second_summary.training.projected.replay_alternate_better_candidates;
                total_replay_bad_flip_unique += second_summary.training.projected.replay_bad_flip_unique;
                total_replay_taboo_vote_candidates += second_summary.training.projected.replay_taboo_vote_candidates;
                total_replay_taboo_mask_flips += second_summary.training.projected.replay_taboo_mask_flips;
                total_replay_taboo_penalty_total += second_summary.training.projected.replay_taboo_penalty_total;
                total_replay_taboo_rejected_vote_candidates += second_summary.training.projected.replay_taboo_rejected_vote_candidates;
                total_replay_minus_one_bad_candidates += second_summary.training.projected.replay_minus_one_bad_candidates;
                total_replay_minus_one_bad_safe_candidates += second_summary.training.projected.replay_minus_one_bad_safe_candidates;
                total_replay_minus_one_bad_deficit_candidates += second_summary.training.projected.replay_minus_one_bad_deficit_candidates;
                total_replay_minus_one_bad_strict_candidates += second_summary.training.projected.replay_minus_one_bad_strict_candidates;
                if (second_summary.training.projected.replay_bad_flip_top_frequency > total_replay_bad_flip_top_frequency ||
                    (second_summary.training.projected.replay_bad_flip_top_frequency == total_replay_bad_flip_top_frequency &&
                     second_summary.training.projected.replay_bad_flip_top_damaged_bits > total_replay_bad_flip_top_damaged_bits)) {
                        total_replay_bad_flip_top_frequency = second_summary.training.projected.replay_bad_flip_top_frequency;
                        total_replay_bad_flip_top_harmless_uses = second_summary.training.projected.replay_bad_flip_top_harmless_uses;
                        total_replay_bad_flip_top_damaged_bits = second_summary.training.projected.replay_bad_flip_top_damaged_bits;
                        total_replay_bad_flip_top_helped_decoded_bit = second_summary.training.projected.replay_bad_flip_top_helped_decoded_bit;
                        total_replay_bad_flip_top_layer = second_summary.training.projected.replay_bad_flip_top_layer;
                        total_replay_bad_flip_top_node = second_summary.training.projected.replay_bad_flip_top_node;
                        total_replay_bad_flip_top_input = second_summary.training.projected.replay_bad_flip_top_input;
                        total_replay_bad_flip_top_bit = second_summary.training.projected.replay_bad_flip_top_bit;
                        total_replay_bad_flip_top_value = second_summary.training.projected.replay_bad_flip_top_value;
                }
                total_unsafe_correct_zero_to_false_positive_bits += second_summary.training.projected.replay_unsafe_transitions.correct_zero_to_false_positive_bits;
                total_unsafe_correct_one_to_false_negative_bits += second_summary.training.projected.replay_unsafe_transitions.correct_one_to_false_negative_bits;
                total_unsafe_false_positive_excess_regressions += second_summary.training.projected.replay_unsafe_false_positive_excess_regressions;
                total_unsafe_target_zero_safety_regressions += second_summary.training.projected.replay_unsafe_target_zero_safety_regressions;
                total_unsafe_selected_node_switches += second_summary.training.projected.replay_unsafe_selected_node_switches;
                total_unsafe_best_decoded_node_switches += second_summary.training.projected.replay_unsafe_best_decoded_node_switches;
                total_rejected_correct_one_to_false_negative_bits += second_summary.training.rejected_round.replay_transitions.correct_one_to_false_negative_bits;
                total_replay_unchanged_correct_bits += second_summary.training.replay_transitions.unchanged_correct_bits;
                total_current_false_negative_to_correct_one_bits += second_summary.training.current_transitions.false_negative_to_correct_one_bits;
                total_current_correct_one_to_false_negative_bits += second_summary.training.current_transitions.correct_one_to_false_negative_bits;
                total_current_false_positive_to_correct_zero_bits += second_summary.training.current_transitions.false_positive_to_correct_zero_bits;
                total_current_correct_zero_to_false_positive_bits += second_summary.training.current_transitions.correct_zero_to_false_positive_bits;
                total_current_false_negative_to_false_positive_bits += second_summary.training.current_transitions.false_negative_to_false_positive_bits;
                total_current_false_positive_to_false_negative_bits += second_summary.training.current_transitions.false_positive_to_false_negative_bits;
                total_current_unchanged_wrong_bits += second_summary.training.current_transitions.unchanged_wrong_bits;
                total_current_unchanged_correct_bits += second_summary.training.current_transitions.unchanged_correct_bits;
                total_current_helpful_decoded_bits += second_summary.training.current_helpful_decoded_bits;
                total_current_harmed_decoded_bits += second_summary.training.current_harmed_decoded_bits;
                total_current_neutral_decoded_bits += second_summary.training.current_neutral_decoded_bits;
                total_current_candidate_net_positive += second_summary.training.current_candidate_net_positive;
                total_current_candidate_net_zero += second_summary.training.current_candidate_net_zero;
                total_current_candidate_net_negative += second_summary.training.current_candidate_net_negative;
                total_current_batch_cancel_target_one_loss += second_summary.training.current_batch_cancel_target_one_loss;
                total_current_batch_cancel_target_zero_loss += second_summary.training.current_batch_cancel_target_zero_loss;
                total_current_batch_cancel_selected_node_change += second_summary.training.current_batch_cancel_selected_node_change;
                total_current_batch_cancel_duplicate_or_overlap_side_effect += second_summary.training.current_batch_cancel_duplicate_or_overlap_side_effect;

                fprintf (stderr,
                         "  round %zu: A distance %zu->%zu excess %zu->%zu zero-safety %zu->%zu; "
                         "B distance %zu->%zu deficit %zu->%zu; safe-quota feasible %zu blocked %zu; "
                         "safe-complete %zu safe-incomplete %zu direct-zero-hit %zu collateral-fp %zu "
                         "reserved-zero %zu; mask-flips safe %zu fp %zu fn %zu weak %zu cost %zu; "
                         "decomp %zu unsafe-frac %zu/%zu alt-unsafe-frac %zu/%zu alt-cost %zu alt-better %zu; "
                         "threshold-crossings %zu; stop %s\n",
                         scaffold_round,
                         first_before_round.selected_distance,
                         first_after_round.selected_distance,
                         first_before_round.false_positive_vote_excess,
                         first_after_round.false_positive_vote_excess,
                         first_before_round.target_zero_safety_min,
                         first_after_round.target_zero_safety_min,
                         second_before_round.selected_distance,
                         second_after_round.selected_distance,
                         second_before_round.false_negative_vote_deficit,
                         second_after_round.false_negative_vote_deficit,
                         second_summary.training.projected.replay_safe_quota_feasible_decoded_bits,
                         second_summary.training.projected.replay_safe_quota_blocked_decoded_bits,
                         second_summary.training.projected.replay_safe_quota_complete_candidates,
                         second_summary.training.projected.replay_safe_quota_incomplete_candidates,
                         second_summary.training.projected.replay_direct_protected_zero_hit_candidates,
                         second_summary.training.projected.replay_realization_collateral_false_positive_candidates,
                         second_summary.training.projected.replay_reserved_zero_votes,
                         second_summary.training.projected.replay_safe_mask_flips,
                         second_summary.training.projected.replay_false_positive_mask_flips,
                         second_summary.training.projected.replay_false_negative_mask_flips,
                         second_summary.training.projected.replay_margin_or_safety_weakening_mask_flips,
                         second_summary.training.projected.replay_collateral_cost,
                         second_summary.training.projected.replay_decomposed_candidates,
                         second_summary.training.projected.replay_decomposed_unsafe_mask_flips,
                         second_summary.training.projected.replay_decomposed_mask_flips,
                         second_summary.training.projected.replay_alternate_unsafe_mask_flips,
                         second_summary.training.projected.replay_alternate_mask_flips,
                         second_summary.training.projected.replay_alternate_collateral_cost,
                         second_summary.training.projected.replay_alternate_better_candidates,
                         threshold_crossings,
                         segment_stop_reason_to_string (second_summary.stop_reason));

                assert (first_after_round.selected_distance == 0);
                if (second_summary.training.projected.accepted_blocks == 0)
                        break;
        }
        assert (itty_feed_model_measure_decoder_objective (model,
                                                           first_input,
                                                           first_target,
                                                           &first_after_second));
        assert (itty_feed_model_measure_decoder_objective (model,
                                                           second_input,
                                                           second_target,
                                                           &second_after));
        assert (itty_feed_model_measure_segment_node_selection (model,
                                                                first_input,
                                                                first_target,
                                                                &first_selection_after_second));
        assert (itty_feed_model_measure_segment_node_selection (model,
                                                                second_input,
                                                                second_target,
                                                                &second_selection_after_second));
        assert (itty_feed_model_measure_segment_node_polarity_selection (model,
                                                                         first_input,
                                                                         first_target,
                                                                         &first_polarity_after_second));
        assert (itty_feed_model_measure_segment_node_polarity_selection (model,
                                                                         second_input,
                                                                         second_target,
                                                                         &second_polarity_after_second));
        fprintf (stderr,
                "segment replay guard: A %zu->%zu, B %zu->%zu, B deficit %zu->%zu, stop %s, "
                "rounds %zu crossings %zu replay rejected %zu bisections %zu scored %zu safe %zu unsafe %zu best-unsafe %zu "
                "safe-strict %zu safe-deficit %zu safe-frontier %zu safe-vote %zu safe-noop %zu safe-irrelevant %zu "
                "safe-quota-feasible %zu safe-quota-selected %zu safe-quota-accepted %zu "
                "safe-quota-completed %zu safe-quota-local-realized %zu safe-quota-local-crossed %zu "
                "safe-quota-net-positive %zu safe-quota-cancelled %zu "
                "safe-quota-final-vote-reached %zu safe-quota-final-vote-changed %zu safe-quota-final-vote-crossed %zu "
                "safe-quota-decoded-vote-changed %zu safe-quota-decoded-threshold-crossed %zu "
                "lost-final-majority %zu lost-final-rotation %zu lost-unselected-final %zu lost-segment-margin %zu lost-duplicate-final %zu "
                "local-votes %zu unique-local %zu unique-final-condensed %zu unique-final-segment %zu unique-decoded %zu "
                "safe-quota-distance-flips %zu safe-quota-rejected-local-only %zu "
                "safe-quota-blocked %zu safe-quota-complete %zu safe-quota-incomplete %zu "
                "direct-zero-hit %zu reserved-zero %zu collateral-fp %zu "
                "mask-flips %zu safe-mask %zu fp-mask %zu fn-mask %zu weak-mask %zu collateral-cost %zu "
                "decomp-candidates %zu unsafe-frac %zu/%zu one-bad %zu mostly-unsafe %zu "
                "alt-unsafe-frac %zu/%zu alt-cost %zu alt-better %zu "
                "bad-flips unique %zu top L%zu N%zu I%zu bit%zu->%zu freq %zu harmless %zu damaged %zu helped-bit %zu "
                "taboo-votes %zu taboo-mask %zu taboo-penalty %zu taboo-rejected %zu "
                "minus-one %zu safe %zu deficit %zu strict %zu "
                "unsafe-c0->fp %zu unsafe-c1->fn %zu unsafe-excess %zu unsafe-zero-safety %zu "
                "unsafe-selected-switch %zu unsafe-best-switch %zu c1->fn %zu unchanged-correct %zu\n",
                first_after_first.selected_distance,
                first_after_second.selected_distance,
                second_before.selected_distance,
                second_after.selected_distance,
                second_before.false_negative_vote_deficit,
                second_after.false_negative_vote_deficit,
                segment_stop_reason_to_string (second_summary.stop_reason),
                scaffold_rounds,
                scaffold_threshold_crossings,
                total_replay_rejected_batches,
                total_replay_bisections,
                total_replay_examples_scored,
                total_replay_safe_candidates,
                total_replay_unsafe_candidates,
                total_replay_best_candidate_unsafe,
                total_replay_safe_strict_candidates,
                total_replay_safe_deficit_candidates,
                total_replay_safe_frontier_candidates,
                total_replay_safe_vote_candidates,
                total_replay_safe_noop_candidates,
                total_replay_safe_irrelevant_candidates,
                total_replay_safe_quota_feasible_bits,
                total_replay_safe_quota_selected_bits,
                total_replay_safe_quota_accepted_bits,
                total_replay_safe_quota_completed_bits,
                total_replay_safe_quota_local_realized_bits,
                total_replay_safe_quota_local_crossed_bits,
                total_replay_safe_quota_net_positive_bits,
                total_replay_safe_quota_cancelled_bits,
                total_replay_safe_quota_final_vote_reached,
                total_replay_safe_quota_final_vote_changed,
                total_replay_safe_quota_final_vote_crossed,
                total_replay_safe_quota_decoded_vote_changed,
                total_replay_safe_quota_decoded_threshold_crossed,
                total_replay_safe_quota_lost_at_final_majority,
                total_replay_safe_quota_lost_at_final_rotation_or_expansion,
                total_replay_safe_quota_lost_at_unselected_final_node,
                total_replay_safe_quota_lost_at_segment_vote_margin,
                total_replay_safe_quota_lost_due_to_duplicate_final_segment,
                total_replay_safe_quota_local_votes_selected,
                total_replay_safe_quota_unique_local_votes_changed,
                total_replay_safe_quota_unique_final_condensed_bits_affected,
                total_replay_safe_quota_unique_final_segment_votes_affected,
                total_replay_safe_quota_unique_decoded_bits_affected,
                total_replay_safe_quota_distance_flips,
                total_replay_safe_quota_rejected_local_only,
                total_replay_safe_quota_blocked_bits,
                total_replay_safe_quota_complete_candidates,
                total_replay_safe_quota_incomplete_candidates,
                total_replay_direct_protected_zero_hit_candidates,
                total_replay_reserved_zero_votes,
                total_replay_realization_collateral_false_positive_candidates,
                total_replay_sensitive_mask_flips,
                total_replay_safe_mask_flips,
                total_replay_false_positive_mask_flips,
                total_replay_false_negative_mask_flips,
                total_replay_margin_or_safety_weakening_mask_flips,
                total_replay_collateral_cost,
                total_replay_decomposed_candidates,
                total_replay_decomposed_unsafe_mask_flips,
                total_replay_decomposed_mask_flips,
                total_replay_one_bad_flip_candidates,
                total_replay_mostly_unsafe_candidates,
                total_replay_alternate_unsafe_mask_flips,
                total_replay_alternate_mask_flips,
                total_replay_alternate_collateral_cost,
                total_replay_alternate_better_candidates,
                total_replay_bad_flip_unique,
                total_replay_bad_flip_top_layer,
                total_replay_bad_flip_top_node,
                total_replay_bad_flip_top_input,
                total_replay_bad_flip_top_bit,
                total_replay_bad_flip_top_value ? (size_t) 1 : (size_t) 0,
                total_replay_bad_flip_top_frequency,
                total_replay_bad_flip_top_harmless_uses,
                total_replay_bad_flip_top_damaged_bits,
                total_replay_bad_flip_top_helped_decoded_bit,
                total_replay_taboo_vote_candidates,
                total_replay_taboo_mask_flips,
                total_replay_taboo_penalty_total,
                total_replay_taboo_rejected_vote_candidates,
                total_replay_minus_one_bad_candidates,
                total_replay_minus_one_bad_safe_candidates,
                total_replay_minus_one_bad_deficit_candidates,
                total_replay_minus_one_bad_strict_candidates,
                total_unsafe_correct_zero_to_false_positive_bits,
                total_unsafe_correct_one_to_false_negative_bits,
                total_unsafe_false_positive_excess_regressions,
                total_unsafe_target_zero_safety_regressions,
                total_unsafe_selected_node_switches,
                total_unsafe_best_decoded_node_switches,
                total_rejected_correct_one_to_false_negative_bits,
                total_replay_unchanged_correct_bits);
        fprintf (stderr,
                "segment replay node diagnostic: "
                "A selected %zu best-distance-node %zu selected-distance %zu best-distance %zu popcount-gap %zu "
                "B selected %zu best-distance-node %zu selected-distance %zu best-distance %zu popcount-gap %zu "
                "B best-deficit-node %zu selected-deficit %zu best-deficit %zu\n",
                first_selection_after_second.selected_by_popcount,
                first_selection_after_second.best_by_target_distance,
                first_selection_after_second.selected_distance,
                first_selection_after_second.best_target_distance,
                first_selection_after_second.popcount_gap,
                second_selection_after_second.selected_by_popcount,
                second_selection_after_second.best_by_target_distance,
                second_selection_after_second.selected_distance,
                second_selection_after_second.best_target_distance,
                second_selection_after_second.popcount_gap,
                second_selection_after_second.best_by_false_negative_deficit,
                second_selection_after_second.selected_false_negative_deficit,
                second_selection_after_second.best_false_negative_deficit);
        fprintf (stderr,
                "B transition audit: "
                "fn->correct %zu correct1->fn %zu fp->correct %zu correct0->fp %zu unchanged-wrong %zu unchanged-correct %zu\n",
                total_current_false_negative_to_correct_one_bits,
                total_current_correct_one_to_false_negative_bits,
                total_current_false_positive_to_correct_zero_bits,
                total_current_correct_zero_to_false_positive_bits,
                total_current_unchanged_wrong_bits,
                total_current_unchanged_correct_bits);
        fprintf (stderr,
                "B per-candidate net: "
                "positive %zu zero %zu negative %zu\n",
                total_current_candidate_net_positive,
                total_current_candidate_net_zero,
                total_current_candidate_net_negative);
        fprintf (stderr,
                "B batch-cancel causes: "
                "target-one-loss %zu target-zero-loss %zu selected-node-change %zu duplicate-overlap %zu\n",
                total_current_batch_cancel_target_one_loss,
                total_current_batch_cancel_target_zero_loss,
                total_current_batch_cancel_selected_node_change,
                total_current_batch_cancel_duplicate_or_overlap_side_effect);
        fprintf (stderr,
                "segment current transitions: "
                "helpful %zu harmed %zu neutral %zu "
                "fn->c1 %zu c1->fn %zu fp->c0 %zu c0->fp %zu fn->fp %zu fp->fn %zu unchanged-wrong %zu unchanged-correct %zu\n",
                total_current_helpful_decoded_bits,
                total_current_harmed_decoded_bits,
                total_current_neutral_decoded_bits,
                total_current_false_negative_to_correct_one_bits,
                total_current_correct_one_to_false_negative_bits,
                total_current_false_positive_to_correct_zero_bits,
                total_current_correct_zero_to_false_positive_bits,
                total_current_false_negative_to_false_positive_bits,
                total_current_false_positive_to_false_negative_bits,
                total_current_unchanged_wrong_bits,
                total_current_unchanged_correct_bits);
        fprintf (stderr,
                "segment replay polarity diagnostic: "
                "A best node %zu %s distance %zu deficit %zu normal %zu/%zu complement %zu/%zu "
                "B best node %zu %s distance %zu deficit %zu normal %zu/%zu complement %zu/%zu\n",
                first_polarity_after_second.best_node,
                first_polarity_after_second.best_complemented ? "complement" : "normal",
                first_polarity_after_second.best_distance,
                first_polarity_after_second.best_false_negative_deficit,
                first_polarity_after_second.best_normal_node,
                first_polarity_after_second.best_normal_distance,
                first_polarity_after_second.best_complement_node,
                first_polarity_after_second.best_complement_distance,
                second_polarity_after_second.best_node,
                second_polarity_after_second.best_complemented ? "complement" : "normal",
                second_polarity_after_second.best_distance,
                second_polarity_after_second.best_false_negative_deficit,
                second_polarity_after_second.best_normal_node,
                second_polarity_after_second.best_normal_distance,
                second_polarity_after_second.best_complement_node,
                second_polarity_after_second.best_complement_distance);
        fprintf (stderr,
                 "segment replay transform diagnostic:\n");
        for (size_t transform_index = 0; transform_index < 4; transform_index++) {
                itty_feed_model_output_transform_t transform = (itty_feed_model_output_transform_t) transform_index;
                itty_feed_model_segment_transform_summary_t first_transform_summary;
                itty_feed_model_segment_transform_summary_t second_transform_summary;

                assert (itty_feed_model_measure_segment_transform (model,
                                                                  first_input,
                                                                  first_target,
                                                                  transform,
                                                                  &first_transform_summary));
                assert (itty_feed_model_measure_segment_transform (model,
                                                                  second_input,
                                                                  second_target,
                                                                  transform,
                                                                  &second_transform_summary));
                fprintf (stderr,
                         "  %s: A distance %zu deficit %zu safety %zu selected %zu best %zu; "
                         "B distance %zu deficit %zu safety %zu selected %zu best %zu\n",
                         output_transform_to_string (transform),
                         first_transform_summary.selected_distance,
                         first_transform_summary.false_negative_vote_deficit,
                         first_transform_summary.target_zero_safety,
                         first_transform_summary.selected_node,
                         first_transform_summary.best_decoded_node,
                         second_transform_summary.selected_distance,
                         second_transform_summary.false_negative_vote_deficit,
                         second_transform_summary.target_zero_safety,
                         second_transform_summary.selected_node,
                         second_transform_summary.best_decoded_node);
        }
        print_replay_final_surface_feasibility (&second_summary.training.projected);
        fprintf (stderr,
                 "segment replay safe-quota effect trace:\n");
        for (size_t trace_index = 0;
             trace_index < second_summary.training.projected.replay_safe_quota_effect_trace_count;
             trace_index++) {
                itty_feed_model_safe_quota_effect_t const *trace =
                        &second_summary.training.projected.replay_safe_quota_effect_traces[trace_index];

                fprintf (stderr,
                         "  bit %zu target %zu penultimate node %zu->%zu ones %zu->%zu threshold %zu "
                         "needed %zu safe-available %zu safe-selected %zu changed %zu "
                         "realized %zu crossed %zu decoded %zu->%zu "
                         "final node %zu->%zu condensed %zu->%zu threshold %zu changed %zu crossed %zu "
                         "segment %zu->%zu threshold %zu changed %zu "
                         "decode %zu->%zu distance %zu->%zu\n",
                         trace->decoded_bit,
                         trace->target_bit,
                         trace->penultimate_selected_node_before,
                         trace->penultimate_selected_node_after,
                         trace->penultimate_ones_before,
                         trace->penultimate_ones_after,
                         trace->penultimate_threshold,
                         trace->needed_before,
                         trace->safe_votes_available,
                         trace->safe_votes_selected,
                         trace->penultimate_votes_changed,
                         trace->quota_realized ? (size_t) 1 : (size_t) 0,
                         trace->penultimate_threshold_crossed ? (size_t) 1 : (size_t) 0,
                         trace->penultimate_decoded_before ? (size_t) 1 : (size_t) 0,
                         trace->penultimate_decoded_after ? (size_t) 1 : (size_t) 0,
                         trace->final_selected_node_before,
                         trace->final_selected_node_after,
                         trace->final_condensed_ones_before,
                         trace->final_condensed_ones_after,
                         trace->final_condensed_threshold,
                         trace->final_condensed_changed ? (size_t) 1 : (size_t) 0,
                         trace->final_condensed_crossed ? (size_t) 1 : (size_t) 0,
                         trace->final_segment_ones_before,
                         trace->final_segment_ones_after,
                         trace->final_segment_threshold,
                         trace->final_segment_changed ? (size_t) 1 : (size_t) 0,
                         trace->final_decoded_before ? (size_t) 1 : (size_t) 0,
                         trace->final_decoded_after ? (size_t) 1 : (size_t) 0,
                         trace->final_distance_contribution_before,
                         trace->final_distance_contribution_after);
        }
        assert (first_after_second.selected_distance == 0);
        assert (total_replay_rejected_batches > 0 ||
                total_replay_examples_scored > 0 ||
                total_replay_safe_quota_feasible_bits == 0);
        run_segment_replay_final_layer_oracle_diagnostic ();

        itty_bit_string_free (first_target);
        itty_bit_string_free (second_target);
        itty_bit_string_list_free (first_input);
        itty_bit_string_list_free (second_input);
       itty_feed_model_free (model);
}

static void
test_itty_feed_model_segment_replay_scaffold_matrix (void)
{
        size_t nodes[] = { 2, 4, 8 };

        printf ("segment replay final-surface matrix:\n");
        printf ("nodes init rotation A-after-B B-distance B-deficit safe-feasible safe-blocked reserved-zero max-safe-feasible rounds crossings selected best-dense best-polarity\n");

        for (size_t node_index = 0; node_index < sizeof (nodes) / sizeof (nodes[0]); node_index++) {
                for (size_t init_mode = 0; init_mode < 2; init_mode++) {
                        for (size_t rotation_mode = 0; rotation_mode < 2; rotation_mode++) {
                                itty_feed_model_t *model = itty_feed_model_new (2,
                                                                                nodes[node_index],
                                                                                1,
                                                                                1);
                                itty_bit_string_list_t *first_input = create_input_with_count (1,
                                                                                               0);
                                itty_bit_string_list_t *second_input = create_input_with_count (1,
                                                                                                1);
                                itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
                                itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
                                itty_feed_model_replay_example_t replay_examples[] = {
                                        {
                                                .input = first_input,
                                                .target = first_target
                                        }
                                };
                                itty_feed_model_refreshed_projected_repair_options_t first_options = {
                                        .batch_size = 64,
                                        .max_rounds = 32,
                                        .max_layer_flips_per_batch = 128
                                };
                                itty_feed_model_refreshed_projected_repair_options_t second_options = {
                                        .batch_size = 64,
                                        .max_rounds = 1,
                                        .max_layer_flips_per_batch = 128,
                                        .top_k_segment_vote_alternatives = 16,
                                        .replay_zero_protection_penalty = 1024,
                                        .replay_one_protection_penalty = 1024,
                                        .replay_taboo_flip_penalty = 1024,
                                        .replay_examples = replay_examples,
                                        .replay_example_count = 1,
                                        .strict_replay_guard = true,
                                        .strict_replay_taboo_rejection = true,
                                        .replay_safe_quota_complete_only = true,
                                        .reserve_replay_protected_zero_votes = true
                                };
                                itty_feed_model_segment_training_summary_t first_summary;
                                itty_feed_model_segment_training_summary_t second_summary;
                                itty_feed_model_decoder_objective_t first_after_first;
                                itty_feed_model_decoder_objective_t first_after_second;
                                itty_feed_model_decoder_objective_t second_after;
                                itty_feed_model_segment_node_selection_summary_t second_selection;
                                itty_feed_model_segment_node_polarity_summary_t second_polarity;
                                size_t scaffold_rounds = 0;
                                size_t threshold_crossings = 0;
                                size_t last_safe_feasible = 0;
                                size_t last_safe_blocked = 0;
                                size_t total_reserved_zero_votes = 0;
                                size_t max_safe_feasible = 0;

                                itty_feed_model_set_decoder (model,
                                                             ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                                if (rotation_mode == 1) {
                                        for (size_t model_layer = 0; model_layer < 2; model_layer++)
                                                itty_feed_model_set_layer_rotation (model,
                                                                                    model_layer,
                                                                                    model_layer + 1);
                                }
                                if (init_mode == 1)
                                        assert (itty_feed_model_randomize_masks (model,
                                                                                 0x7100 + node_index * 16 + rotation_mode,
                                                                                 1,
                                                                                 8));

                                assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                                        first_input,
                                                                                                        first_target,
                                                                                                        &first_options,
                                                                                                        &first_summary));
                                assert (itty_feed_model_measure_decoder_objective (model,
                                                                                   first_input,
                                                                                   first_target,
                                                                                   &first_after_first));
                                if (first_after_first.selected_distance != 0) {
                                        assert (itty_feed_model_measure_decoder_objective (model,
                                                                                           second_input,
                                                                                           second_target,
                                                                                           &second_after));
                                        assert (itty_feed_model_measure_segment_node_selection (model,
                                                                                                second_input,
                                                                                                second_target,
                                                                                                &second_selection));
                                        assert (itty_feed_model_measure_segment_node_polarity_selection (model,
                                                                                                         second_input,
                                                                                                         second_target,
                                                                                                         &second_polarity));

                                        printf ("%zu %s %s %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %s\n",
                                                nodes[node_index],
                                                init_mode == 0 ? "zero" : "sparse",
                                                rotation_mode == 0 ? "none" : "varied",
                                                first_after_first.selected_distance,
                                                second_after.selected_distance,
                                                second_after.false_negative_vote_deficit,
                                                last_safe_feasible,
                                                last_safe_blocked,
                                                total_reserved_zero_votes,
                                                max_safe_feasible,
                                                scaffold_rounds,
                                                threshold_crossings,
                                                second_selection.selected_by_popcount,
                                                second_selection.best_by_target_distance,
                                                second_polarity.best_complemented ? "complement" : "normal");

                                        itty_bit_string_free (first_target);
                                        itty_bit_string_free (second_target);
                                        itty_bit_string_list_free (first_input);
                                        itty_bit_string_list_free (second_input);
                                        itty_feed_model_free (model);
                                        continue;
                                }

                                for (size_t scaffold_round = 0; scaffold_round < 8; scaffold_round++) {
                                        itty_feed_model_decoder_objective_t second_before_round;
                                        itty_feed_model_decoder_objective_t second_after_round;

                                        assert (itty_feed_model_measure_decoder_objective (model,
                                                                                           second_input,
                                                                                           second_target,
                                                                                           &second_before_round));
                                        assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                                                second_input,
                                                                                                                second_target,
                                                                                                                &second_options,
                                                                                                                &second_summary));
                                        assert (itty_feed_model_measure_decoder_objective (model,
                                                                                           second_input,
                                                                                           second_target,
                                                                                           &second_after_round));

                                        if (second_before_round.selected_distance > second_after_round.selected_distance)
                                                threshold_crossings += second_before_round.selected_distance - second_after_round.selected_distance;

                                        last_safe_feasible = second_summary.training.projected.replay_safe_quota_feasible_decoded_bits;
                                        last_safe_blocked = second_summary.training.projected.replay_safe_quota_blocked_decoded_bits;
                                        total_reserved_zero_votes += second_summary.training.projected.replay_reserved_zero_votes;
                                        if (last_safe_feasible > max_safe_feasible)
                                                max_safe_feasible = last_safe_feasible;
                                        scaffold_rounds++;

                                        if (second_summary.training.projected.accepted_blocks == 0)
                                                break;
                                }

                                assert (itty_feed_model_measure_decoder_objective (model,
                                                                                   first_input,
                                                                                   first_target,
                                                                                   &first_after_second));
                                assert (itty_feed_model_measure_decoder_objective (model,
                                                                                   second_input,
                                                                                   second_target,
                                                                                   &second_after));
                                assert (itty_feed_model_measure_segment_node_selection (model,
                                                                                        second_input,
                                                                                        second_target,
                                                                                        &second_selection));
                                assert (itty_feed_model_measure_segment_node_polarity_selection (model,
                                                                                                 second_input,
                                                                                                 second_target,
                                                                                                 &second_polarity));

                                printf ("%zu %s %s %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %s\n",
                                        nodes[node_index],
                                        init_mode == 0 ? "zero" : "sparse",
                                        rotation_mode == 0 ? "none" : "varied",
                                        first_after_second.selected_distance,
                                        second_after.selected_distance,
                                        second_after.false_negative_vote_deficit,
                                        last_safe_feasible,
                                        last_safe_blocked,
                                        total_reserved_zero_votes,
                                        max_safe_feasible,
                                        scaffold_rounds,
                                        threshold_crossings,
                                        second_selection.selected_by_popcount,
                                        second_selection.best_by_target_distance,
                                        second_polarity.best_complemented ? "complement" : "normal");

                                assert (first_after_second.selected_distance == 0);

                                itty_bit_string_free (first_target);
                                itty_bit_string_free (second_target);
                                itty_bit_string_list_free (first_input);
                                itty_bit_string_list_free (second_input);
                                itty_feed_model_free (model);
                        }
                }
        }
}

static void
test_itty_feed_model_segment_replay_transaction_capacity_matrix (void)
{
        size_t nodes[] = { 2, 4, 8 };

        printf ("segment replay transaction matrix:\n");
        printf ("nodes init rotation A-solved B-scaffold-dist B-scaffold-def d1 top4 B-finish B-restore B-replace repl-cand repl-safe repl-dist repl-progress repl-survive selected best-dense polarity\n");

        for (size_t node_index = 0; node_index < sizeof (nodes) / sizeof (nodes[0]); node_index++) {
                for (size_t init_mode = 0; init_mode < 2; init_mode++) {
                        for (size_t rotation_mode = 0; rotation_mode < 2; rotation_mode++) {
                                itty_feed_model_t *model = itty_feed_model_new (2,
                                                                                nodes[node_index],
                                                                                1,
                                                                                1);
                                itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
                                itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
                                itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
                                itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
                                itty_feed_model_refreshed_projected_repair_options_t first_options = {
                                        .batch_size = 64,
                                        .max_rounds = 32,
                                        .max_layer_flips_per_batch = 128
                                };
                                itty_feed_model_train_options_t oracle_options = {
                                        .max_flips = 8,
                                        .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                                        .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
                                };
                                itty_feed_model_segment_training_summary_t first_summary;
                                itty_feed_model_transaction_scaffold_round_t trajectory[ITTY_FEED_MODEL_TRANSACTION_SCAFFOLD_TRAJECTORY_LIMIT] = { 0 };
                                itty_feed_model_transaction_scaffold_summary_t scaffold = { 0 };
                                itty_feed_model_decoder_objective_t a_after = { 0 };
                                itty_feed_model_decoder_objective_t b_after = { 0 };
                                itty_feed_model_segment_node_selection_summary_t selection = { 0 };
                                itty_feed_model_segment_node_polarity_summary_t polarity = { 0 };
                                char best_finish_text[32] = "-";
                                char best_restore_text[32] = "-";
                                char best_replacement_text[32] = "-";
                                size_t best_finish_distance = (size_t) -1;
                                size_t best_restore_distance = (size_t) -1;
                                size_t best_replacement_distance = (size_t) -1;
                                size_t replacement_candidates = 0;
                                size_t replacement_safe = 0;
                                size_t replacement_distance_helpful = 0;
                                size_t replacement_progress_helpful = 0;
                                bool replacement_survivor = false;

                                itty_feed_model_set_decoder (model,
                                                             ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                                if (rotation_mode == 1) {
                                        for (size_t model_layer = 0; model_layer < 2; model_layer++)
                                                itty_feed_model_set_layer_rotation (model,
                                                                                    model_layer,
                                                                                    model_layer + 1);
                                }
                                if (init_mode == 1)
                                        assert (itty_feed_model_randomize_masks (model,
                                                                                 0x9100 + node_index * 16 + rotation_mode,
                                                                                 1,
                                                                                 8));

                                assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                                        first_input,
                                                                                                        first_target,
                                                                                                        &first_options,
                                                                                                        &first_summary));
                                assert (itty_feed_model_train_final_layer_transaction_scaffold (model,
                                                                                                first_input,
                                                                                                first_target,
                                                                                                second_input,
                                                                                                second_target,
                                                                                                &oracle_options,
                                                                                                8,
                                                                                                trajectory,
                                                                                                ITTY_FEED_MODEL_TRANSACTION_SCAFFOLD_TRAJECTORY_LIMIT,
                                                                                                &scaffold));
                                assert (itty_feed_model_measure_decoder_objective (model,
                                                                                   first_input,
                                                                                   first_target,
                                                                                   &a_after));
                                assert (itty_feed_model_measure_decoder_objective (model,
                                                                                   second_input,
                                                                                   second_target,
                                                                                   &b_after));
                                assert (itty_feed_model_measure_segment_node_selection (model,
                                                                                        second_input,
                                                                                        second_target,
                                                                                        &selection));
                                assert (itty_feed_model_measure_segment_node_polarity_selection (model,
                                                                                                 second_input,
                                                                                                 second_target,
                                                                                                 &polarity));

                                for (size_t trace_index = 0;
                                     trace_index < scaffold.finish_trace_count &&
                                     trace_index < ITTY_FEED_MODEL_FINISH_CANDIDATE_TRACE_LIMIT;
                                     trace_index++) {
                                        itty_feed_model_finish_candidate_trace_t const *trace =
                                                &scaffold.finish_traces[trace_index];
                                        replacement_candidates += trace->replacement_candidate_count;
                                        replacement_safe += trace->replacement_a_safe_count;
                                        replacement_distance_helpful += trace->replacement_distance_helpful_count;
                                        replacement_progress_helpful += trace->replacement_progress_helpful_count;
                                        if (trace->family_one_finish) {
                                                if (trace->b_distance_after_finish < best_finish_distance)
                                                        best_finish_distance = trace->b_distance_after_finish;
                                                if (trace->damage_set_restore_restored_a &&
                                                    trace->b_distance_after_restore < best_restore_distance)
                                                        best_restore_distance = trace->b_distance_after_restore;
                                                if (trace->replacement_available) {
                                                        replacement_survivor = true;
                                                        if (trace->b_distance_after_restore < best_replacement_distance)
                                                                best_replacement_distance = trace->b_distance_after_restore;
                                                }
                                        }
                                }

                                if (best_finish_distance != (size_t) -1)
                                        snprintf (best_finish_text,
                                                  sizeof best_finish_text,
                                                  "%zu",
                                                  best_finish_distance);
                                if (best_restore_distance != (size_t) -1)
                                        snprintf (best_restore_text,
                                                  sizeof best_restore_text,
                                                  "%zu",
                                                  best_restore_distance);
                                if (best_replacement_distance != (size_t) -1)
                                        snprintf (best_replacement_text,
                                                  sizeof best_replacement_text,
                                                  "%zu",
                                                  best_replacement_distance);

                                printf ("%zu %s %s %s %zu %zu %zu %zu ",
                                        nodes[node_index],
                                        init_mode == 0 ? "zero" : "sparse",
                                        rotation_mode == 0 ? "none" : "varied",
                                        a_after.selected_distance == 0 ? "yes" : "no",
                                        b_after.selected_distance,
                                        b_after.false_negative_vote_deficit,
                                        scaffold.b_deficit_one_bits_after,
                                        scaffold.b_top_k_completion_cost_after);
                                printf ("%s %s %s %zu %zu %zu %zu %s %zu %zu %s\n",
                                        best_finish_text,
                                        best_restore_text,
                                        best_replacement_text,
                                        replacement_candidates,
                                        replacement_safe,
                                        replacement_distance_helpful,
                                        replacement_progress_helpful,
                                        replacement_survivor ? "yes" : "no",
                                        selection.selected_by_popcount,
                                        selection.best_by_target_distance,
                                        polarity.best_complemented ? "complement" : "normal");

                                itty_bit_string_free (first_target);
                                itty_bit_string_free (second_target);
                                itty_bit_string_list_free (first_input);
                                itty_bit_string_list_free (second_input);
                                itty_feed_model_free (model);
                        }
                }
        }
}

static void
test_itty_feed_model_segment_training_matrix_tracks_bounded_runs (void)
{
        size_t layers[] = { 2, 4, 8, 12 };
        size_t nodes[] = { 1, 2, 4, 8 };
        size_t target_words[] = {
                create_half_populated_word (),
                1,
                create_mixed_word ()
        };
        size_t checked_runs = 0;
        size_t improved_runs = 0;
        size_t converged_runs = 0;
        size_t rejected_runs = 0;
        size_t rejected_distance_runs = 0;
        size_t rejected_excess_runs = 0;
        size_t rejected_zero_safety_runs = 0;
        size_t rejected_one_margin_runs = 0;
        size_t rejected_no_delta_runs = 0;
        size_t rejected_true_noop_runs = 0;
        size_t rejected_local_only_runs = 0;
        size_t rejected_vote_tied_runs = 0;
        size_t rejected_future_cost_runs = 0;
        size_t rejected_realization_runs = 0;
        size_t rejected_interaction_runs = 0;
        size_t max_round_runs = 0;
        size_t no_repair_runs = 0;
        size_t no_effective_candidate_runs = 0;
        size_t no_progress_runs = 0;
        size_t total_flips = 0;
        size_t total_distance_before = 0;
        size_t total_distance_after = 0;
        size_t total_deficit_before = 0;
        size_t total_deficit_after = 0;
        size_t no_effect_candidates = 0;
        size_t no_effect_already_satisfied = 0;
        size_t no_effect_no_majority_crossing = 0;
        size_t no_effect_unselected_node = 0;
        size_t no_effect_irrelevant_segment = 0;
        size_t no_effect_vote_tied = 0;
        size_t replay_safe_candidates = 0;
        size_t replay_unsafe_candidates = 0;
        size_t replay_best_candidate_unsafe = 0;

        printf ("segment training matrix:\n");
        printf ("layers nodes rotation init target distance flips rounds deficit excess zero-safety entropy stop\n");

        for (size_t layer_index = 0; layer_index < sizeof (layers) / sizeof (layers[0]); layer_index++) {
                for (size_t node_index = 0; node_index < sizeof (nodes) / sizeof (nodes[0]); node_index++) {
                        for (size_t rotation_mode = 0; rotation_mode < 2; rotation_mode++) {
                                for (size_t init_mode = 0; init_mode < 2; init_mode++) {
                                        for (size_t target_index = 0; target_index < sizeof (target_words) / sizeof (target_words[0]); target_index++) {
                                                itty_feed_model_t *model = itty_feed_model_new (layers[layer_index],
                                                                                                nodes[node_index],
                                                                                                1,
                                                                                                1);
                                                itty_bit_string_list_t *input = create_input_with_count (1,
                                                                                                         target_index);
                                                itty_bit_string_t *target = create_bit_string (target_words[target_index]);
                                                itty_feed_model_refreshed_projected_repair_options_t repair_options = {
                                                        .batch_size = 8,
                                                        .max_rounds = 1,
                                                        .max_layer_flips_per_batch = 32
                                                };
                                                itty_feed_model_segment_training_summary_t summary;
                                                itty_feed_model_test_segment_training_row_t row;

                                                if (rotation_mode == 1) {
                                                        for (size_t model_layer = 0; model_layer < layers[layer_index]; model_layer++)
                                                                itty_feed_model_set_layer_rotation (model,
                                                                                                    model_layer,
                                                                                                    model_layer + 1);
                                                }
                                                if (init_mode == 1)
                                                        assert (itty_feed_model_randomize_masks (model,
                                                                                                 0xc0ffee + checked_runs,
                                                                                                 1,
                                                                                                 8));

                                                assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                                                        input,
                                                                                                                        target,
                                                                                                                        &repair_options,
                                                                                                                        &summary));
                                                capture_segment_training_row (&summary,
                                                                              &row);

                                                assert (row.final_distance <= ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
                                                assert (row.deficit_after <= row.deficit_before ||
                                                        row.final_distance <= row.initial_distance);
                                                assert (row.excess_after <= row.excess_before ||
                                                        row.final_distance <= row.initial_distance);
                                                assert (row.rounds <= repair_options.max_rounds);
                                                assert (row.stop_reason >= ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_CONVERGED);
                                                assert (row.stop_reason <= ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_PROGRESS);
                                                if (row.final_distance < row.initial_distance)
                                                        improved_runs++;
                                                switch (row.stop_reason) {
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_CONVERGED:
                                                        converged_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_BATCH:
                                                        rejected_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_DISTANCE_REGRESSION:
                                                        rejected_runs++;
                                                        rejected_distance_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_EXCESS_REGRESSION:
                                                        rejected_runs++;
                                                        rejected_excess_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_ZERO_SAFETY_REGRESSION:
                                                        rejected_runs++;
                                                        rejected_zero_safety_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_TARGET_ONE_MARGIN_LOSS:
                                                        rejected_runs++;
                                                        rejected_one_margin_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_NO_OBJECTIVE_DELTA:
                                                        rejected_runs++;
                                                        rejected_no_delta_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_TRUE_NOOP:
                                                        rejected_runs++;
                                                        rejected_true_noop_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_LOCAL_ACTIVATION_ONLY:
                                                        rejected_runs++;
                                                        rejected_local_only_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_VOTE_MOVEMENT_TIED:
                                                        rejected_runs++;
                                                        rejected_vote_tied_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_FUTURE_COST_IMPROVEMENT:
                                                        rejected_runs++;
                                                        rejected_future_cost_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_REALIZATION_MISMATCH:
                                                        rejected_runs++;
                                                        rejected_realization_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_BATCH_INTERACTION:
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REJECTED_REPLAY_GUARD:
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_PREFIX_ORDERING_FAILURE:
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_SINGLE_CANDIDATE_CONFLICT:
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_CAPACITY_CONFLICT:
                                                        rejected_runs++;
                                                        rejected_interaction_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_BLOCKED_ALL_CANDIDATES:
                                                        no_effective_candidate_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_REPLAY_SAFE_NO_CURRENT_GAIN:
                                                        no_progress_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_MAX_ROUNDS:
                                                        max_round_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_REPAIRS:
                                                        no_repair_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_EFFECTIVE_CANDIDATES:
                                                        no_effective_candidate_runs++;
                                                        break;
                                                case ITTY_FEED_MODEL_SEGMENT_TRAIN_STOP_NO_PROGRESS:
                                                        no_progress_runs++;
                                                        break;
                                                }
                                                total_flips += row.flips;
                                                total_distance_before += row.initial_distance;
                                                total_distance_after += row.final_distance;
                                                total_deficit_before += row.deficit_before;
                                                total_deficit_after += row.deficit_after;
                                                no_effect_candidates += summary.training.projected.no_effect_candidates;
                                                no_effect_already_satisfied += summary.training.projected.no_effect_candidate_already_satisfied;
                                                no_effect_no_majority_crossing += summary.training.projected.no_effect_candidate_no_majority_crossing;
                                                no_effect_unselected_node += summary.training.projected.no_effect_candidate_unselected_node;
                                                no_effect_irrelevant_segment += summary.training.projected.no_effect_candidate_irrelevant_segment;
                                                no_effect_vote_tied += summary.training.projected.no_effect_candidate_vote_tied;
                                                replay_safe_candidates += summary.training.projected.replay_safe_candidates;
                                                replay_unsafe_candidates += summary.training.projected.replay_unsafe_candidates;
                                                replay_best_candidate_unsafe += summary.training.projected.replay_best_candidate_unsafe;
                                                printf ("%zu %zu %s %s %s %zu->%zu %zu %zu %zu->%zu %zu->%zu %zu %.4f->%.4f %s\n",
                                                        layers[layer_index],
                                                        nodes[node_index],
                                                        rotation_mode == 0 ? "none" : "varied",
                                                        init_mode == 0 ? "zero" : "sparse",
                                                        segment_target_name (target_index),
                                                        row.initial_distance,
                                                        row.final_distance,
                                                        row.flips,
                                                        row.rounds,
                                                        row.deficit_before,
                                                        row.deficit_after,
                                                        row.excess_before,
                                                        row.excess_after,
                                                        row.zero_safety,
                                                        row.entropy_before,
                                                        row.entropy_after,
                                                        segment_stop_reason_to_string (row.stop_reason));
                                                if (summary.training.rejected_rounds > 0) {
                                                        itty_feed_model_refreshed_projected_repair_round_t *rejected = &summary.training.rejected_round;
                                                        printf ("  rejected-batch detail: proposed %zu selected %zu "
                                                                "estimated-distance-delta %td actual-distance-delta %td "
                                                                "distance %zu->%zu deficit %zu->%zu excess %zu->%zu "
                                                                "one-margin %zu->%zu zero-safety %zu->%zu "
                                                                "constraints %zu->%zu collateral %zu "
                                                                "activation-changed %d vote-changed %d "
                                                                "selected-node %zu->%zu best-node %zu->%zu reason %s\n",
                                                                rejected->candidate_blocks_proposed,
                                                                rejected->candidate_blocks_selected,
                                                                rejected->estimated_distance_delta,
                                                                rejected->actual_distance_delta,
                                                                rejected->before_distance,
                                                                rejected->after_distance,
                                                                rejected->before_false_negative_deficit,
                                                                rejected->after_false_negative_deficit,
                                                                rejected->before_false_positive_excess,
                                                                rejected->after_false_positive_excess,
                                                                rejected->before_target_one_margin,
                                                                rejected->after_target_one_margin,
                                                                rejected->before_target_zero_safety,
                                                                rejected->after_target_zero_safety,
                                                                rejected->requested_constraints,
                                                                rejected->realized_constraints,
                                                                rejected->collateral_changed_bits,
                                                                rejected->selected_activation_changed,
                                                                rejected->segment_votes_changed,
                                                                rejected->selected_node,
                                                                rejected->selected_node_after,
                                                                rejected->best_decoded_node,
                                                                rejected->best_decoded_node_after,
                                                                segment_stop_reason_to_string (rejected->rejection_reason));
                                                }
                                                checked_runs++;

                                                itty_bit_string_free (target);
                                                itty_bit_string_list_free (input);
                                                itty_feed_model_free (model);
                                        }
                                }
                        }
                }
        }

        assert (checked_runs == 4 * 4 * 2 * 2 * 3);
        assert (improved_runs > 0);
        printf ("segment training matrix summary: runs %zu, improved %zu, converged %zu, "
                "rejected %zu, max-rounds %zu, no-repairs %zu, no-effective %zu, no-progress %zu, "
                "distance %zu->%zu, deficit %zu->%zu, flips %zu\n",
                checked_runs,
                improved_runs,
                converged_runs,
                rejected_runs,
                max_round_runs,
                no_repair_runs,
                no_effective_candidate_runs,
                no_progress_runs,
                total_distance_before,
                total_distance_after,
                total_deficit_before,
                total_deficit_after,
                total_flips);
        printf ("segment training rejection breakdown: distance %zu, excess %zu, "
                "zero-safety %zu, one-margin %zu, no-delta %zu, realization %zu, interaction %zu\n",
                rejected_distance_runs,
                rejected_excess_runs,
                rejected_zero_safety_runs,
                rejected_one_margin_runs,
                rejected_no_delta_runs,
                rejected_realization_runs,
                rejected_interaction_runs);
        printf ("segment training no-delta split: true-noop %zu, local-only %zu, "
                "vote-tied %zu, future-cost %zu\n",
                rejected_true_noop_runs,
                rejected_local_only_runs,
                rejected_vote_tied_runs,
                rejected_future_cost_runs);
        printf ("segment training no-effect candidates: total %zu, already-satisfied %zu, "
                "no-majority-crossing %zu, unselected-node %zu, irrelevant-segment %zu, vote-tied %zu, "
                "replay-safe %zu, replay-unsafe %zu, best-unsafe %zu\n",
                no_effect_candidates,
                no_effect_already_satisfied,
                no_effect_no_majority_crossing,
                no_effect_unselected_node,
                no_effect_irrelevant_segment,
                no_effect_vote_tied,
                replay_safe_candidates,
                replay_unsafe_candidates,
                replay_best_candidate_unsafe);
}

static void
test_itty_feed_model_segment_replay_forced_route_diagnostic (void)
{
        struct replay_route_case {
                size_t      nodes;
                bool        sparse_init;
                bool        varied_rotation;
                char const *label;
        } cases[] = {
                { 2, false, false, "2-zero-none" },
                { 4, false, false, "4-zero-none" },
                { 8, true,  false, "8-sparse-none" }
        };

        printf ("segment replay forced routes:\n");
        printf ("case route A-dist A-fp A-def A-pop A-sel A-best B-dist B-fp B-def B-pop B-sel B-best\n");

        for (size_t case_index = 0; case_index < sizeof (cases) / sizeof (cases[0]); case_index++) {
                struct replay_route_case const *cfg = &cases[case_index];
                itty_feed_model_t *model = itty_feed_model_new (2,
                                                                cfg->nodes,
                                                                1,
                                                                1);
                itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
                itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
                itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
                itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
                itty_feed_model_refreshed_projected_repair_options_t first_options = {
                        .batch_size = 64,
                        .max_rounds = 32,
                        .max_layer_flips_per_batch = 128
                };
                itty_feed_model_train_options_t oracle_options = {
                        .max_flips = 8,
                        .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                        .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
                };
                itty_feed_model_segment_training_summary_t first_summary;
                itty_feed_model_transaction_scaffold_summary_t scaffold = { 0 };
                itty_feed_model_segment_node_selection_summary_t a_selection = { 0 };
                itty_feed_model_segment_node_selection_summary_t b_selection = { 0 };

                itty_feed_model_set_decoder (model,
                                             ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                if (cfg->varied_rotation) {
                        for (size_t model_layer = 0; model_layer < 2; model_layer++)
                                itty_feed_model_set_layer_rotation (model,
                                                                    model_layer,
                                                                    model_layer + 1);
                }
                if (cfg->sparse_init)
                        assert (itty_feed_model_randomize_masks (model,
                                                                 0xa100 + case_index,
                                                                 1,
                                                                 8));

                assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                        first_input,
                                                                                        first_target,
                                                                                        &first_options,
                                                                                        &first_summary));
                assert (itty_feed_model_train_final_layer_transaction_scaffold (model,
                                                                                first_input,
                                                                                first_target,
                                                                                second_input,
                                                                                second_target,
                                                                                &oracle_options,
                                                                                8,
                                                                                NULL,
                                                                                0,
                                                                                &scaffold));
                assert (itty_feed_model_measure_segment_node_selection (model,
                                                                        first_input,
                                                                        first_target,
                                                                        &a_selection));
                assert (itty_feed_model_measure_segment_node_selection (model,
                                                                        second_input,
                                                                        second_target,
                                                                        &b_selection));

                for (size_t route = 0; route < cfg->nodes; route++) {
                        itty_feed_model_decoder_objective_t a_forced = { 0 };
                        itty_feed_model_decoder_objective_t b_forced = { 0 };

                        assert (itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                    first_input,
                                                                                    first_target,
                                                                                    route,
                                                                                    &a_forced));
                        assert (itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                    second_input,
                                                                                    second_target,
                                                                                    route,
                                                                                    &b_forced));

                        printf ("%s %zu %zu %zu %zu %zu %s %s %zu %zu %zu %zu %s %s\n",
                                cfg->label,
                                route,
                                a_forced.selected_distance,
                                a_forced.false_positive_vote_excess,
                                a_forced.false_negative_vote_deficit,
                                a_forced.selected_popcount,
                                a_selection.selected_by_popcount == route ? "sel" : "-",
                                a_selection.best_by_target_distance == route ? "best" : "-",
                                b_forced.selected_distance,
                                b_forced.false_positive_vote_excess,
                                b_forced.false_negative_vote_deficit,
                                b_forced.selected_popcount,
                                b_selection.selected_by_popcount == route ? "sel" : "-",
                                b_selection.best_by_target_distance == route ? "best" : "-");
                }

                itty_bit_string_free (first_target);
                itty_bit_string_free (second_target);
                itty_bit_string_list_free (first_input);
                itty_bit_string_list_free (second_input);
                itty_feed_model_free (model);
        }
}

typedef enum {
        REPLAY_ROUTE_DEAD,
        REPLAY_ROUTE_UNSAFE,
        REPLAY_ROUTE_LATENT,
        REPLAY_ROUTE_STABLE,
} replay_route_class_t;

static char const *
replay_route_class_name (replay_route_class_t route_class)
{
        switch (route_class) {
        case REPLAY_ROUTE_STABLE:
                return "stable";
        case REPLAY_ROUTE_LATENT:
                return "latent";
        case REPLAY_ROUTE_UNSAFE:
                return "unsafe";
        case REPLAY_ROUTE_DEAD:
        default:
                return "dead";
        }
}

static bool
replay_route_is_a_safe (itty_feed_model_decoder_objective_t const *objective)
{
        return objective->selected_distance == 0 &&
               objective->false_positive_vote_excess == 0 &&
               objective->false_negative_vote_deficit == 0;
}

static replay_route_class_t
replay_classify_route (itty_feed_model_decoder_objective_t const                 *a_objective,
                       itty_feed_model_decoder_objective_t const                 *b_objective,
                       itty_feed_model_segment_node_selection_summary_t const    *b_selection,
                       size_t                                                     route,
                       size_t                                                     selector_margin)
{
        bool a_safe = replay_route_is_a_safe (a_objective);
        bool b_selected = b_selection->selected_by_popcount == route;
        bool b_best = b_selection->best_by_target_distance == route;
        bool b_helpful = b_objective->selected_distance < b_selection->selected_distance ||
                         b_objective->false_negative_vote_deficit < b_selection->selected_false_negative_deficit;
        size_t selected_popcount = b_selected ? b_objective->selected_popcount : b_selection->selected_popcount;
        bool selector_margin_met = b_selected && b_objective->selected_popcount >= selected_popcount + selector_margin;

        if (a_safe && b_selected && b_best && selector_margin_met)
                return REPLAY_ROUTE_STABLE;
        if (a_safe && (b_best || b_helpful))
                return REPLAY_ROUTE_LATENT;
        if (!a_safe && (b_best || b_helpful))
                return REPLAY_ROUTE_UNSAFE;
        return REPLAY_ROUTE_DEAD;
}

static ptrdiff_t
replay_route_margin_for_objectives (itty_feed_model_decoder_objective_t const *objectives,
                                    size_t                                      count,
                                    size_t                                      route)
{
        size_t route_popcount = objectives[route].selected_popcount;
        size_t max_other = 0;

        for (size_t other = 0; other < count; other++) {
                if (other == route)
                        continue;
                if (objectives[other].selected_popcount > max_other)
                        max_other = objectives[other].selected_popcount;
        }

        return (ptrdiff_t) route_popcount - (ptrdiff_t) max_other;
}

static bool
replay_options_have_lane_split (itty_feed_model_train_options_t const *options)
{
        return options &&
               (options->selector_lane_bit_count > 0 ||
                options->decoder_lane_bit_count > 0);
}

static bool
replay_measure_decoder_objective_with_options (itty_feed_model_t                     *model,
                                               itty_bit_string_list_t                *input,
                                               itty_bit_string_t                     *target,
                                               itty_feed_model_train_options_t const *options,
                                               itty_feed_model_decoder_objective_t   *objective)
{
        if (replay_options_have_lane_split (options))
                return itty_feed_model_measure_decoder_objective_with_lane_split (model,
                                                                                  input,
                                                                                  target,
                                                                                  options->selector_lane_bit_offset,
                                                                                  options->selector_lane_bit_count,
                                                                                  options->decoder_lane_bit_offset,
                                                                                  options->decoder_lane_bit_count,
                                                                                  objective);
        return itty_feed_model_measure_decoder_objective (model, input, target, objective);
}

static bool
replay_measure_decoder_objective_for_node_with_options (itty_feed_model_t                     *model,
                                                        itty_bit_string_list_t                *input,
                                                        itty_bit_string_t                     *target,
                                                        size_t                                 route,
                                                        itty_feed_model_train_options_t const *options,
                                                        itty_feed_model_decoder_objective_t   *objective)
{
        if (replay_options_have_lane_split (options))
                return itty_feed_model_measure_decoder_objective_for_node_with_lane_split (model,
                                                                                           input,
                                                                                           target,
                                                                                           route,
                                                                                           options->selector_lane_bit_offset,
                                                                                           options->selector_lane_bit_count,
                                                                                           options->decoder_lane_bit_offset,
                                                                                           options->decoder_lane_bit_count,
                                                                                           objective);
        return itty_feed_model_measure_decoder_objective_for_node (model, input, target, route, objective);
}

static bool
replay_measure_segment_node_selection_with_options (itty_feed_model_t                                *model,
                                                    itty_bit_string_list_t                           *input,
                                                    itty_bit_string_t                                *target,
                                                    itty_feed_model_train_options_t const            *options,
                                                    itty_feed_model_segment_node_selection_summary_t *summary)
{
        if (replay_options_have_lane_split (options))
                return itty_feed_model_measure_segment_node_selection_with_lane_split (model,
                                                                                       input,
                                                                                       target,
                                                                                       options->selector_lane_bit_offset,
                                                                                       options->selector_lane_bit_count,
                                                                                       options->decoder_lane_bit_offset,
                                                                                       options->decoder_lane_bit_count,
                                                                                       summary);
        return itty_feed_model_measure_segment_node_selection (model, input, target, summary);
}

static bool
replay_measure_shadow_selected_decode (itty_feed_model_t                                *decoder_model,
                                       itty_feed_model_t                                *selector_model,
                                       itty_bit_string_list_t                           *input,
                                       itty_bit_string_t                                *target,
                                       itty_feed_model_segment_node_selection_summary_t *selector_selection,
                                       itty_feed_model_decoder_objective_t              *decoder_objective)
{
        if (!itty_feed_model_measure_segment_node_selection (selector_model,
                                                             input,
                                                             target,
                                                             selector_selection))
                return false;
        return itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                   input,
                                                                   target,
                                                                   selector_selection->selected_by_popcount,
                                                                   decoder_objective);
}

typedef struct {
        size_t    accepted_repairs;
        size_t    owner_lifts;
        size_t    competitor_suppresses;
        ptrdiff_t margin_before;
        ptrdiff_t margin_after;
        bool      owner_route_safe;
        bool      global_safe;
} replay_selector_block_summary_t;

typedef struct {
        itty_feed_model_segment_node_selection_summary_t a_selector;
        itty_feed_model_segment_node_selection_summary_t b_selector;
        itty_feed_model_decoder_objective_t              a_shadow;
        itty_feed_model_decoder_objective_t              a_forced;
        itty_feed_model_decoder_objective_t              b_forced;
        ptrdiff_t                                        a_margin;
        ptrdiff_t                                        b_margin;
        ptrdiff_t                                        a_margin_shortfall;
        ptrdiff_t                                        b_margin_shortfall;
        ptrdiff_t                                        b_decoder_distance_shortfall;
        ptrdiff_t                                        b_decoder_deficit_shortfall;
        bool                                             a_safe;
        bool                                             b_selecting;
} replay_selector_joint_state_t;

typedef struct {
        size_t                                    b_desired_popcount;
        size_t                                    b_winner_route;
        size_t                                    b_winner_popcount;
        size_t                                    b_gap;
        size_t                                    b_needed;
        ptrdiff_t                                 b_margin;
        itty_feed_model_selector_protection_summary_t b_selector_inventory;
        itty_feed_model_selector_protection_summary_t a_compensation_inventory;
} replay_shadow_selector_inventory_t;

typedef struct {
        size_t    a_selected_route;
        size_t    b_selected_route;
        ptrdiff_t a_margin;
        ptrdiff_t b_margin;
        ptrdiff_t a_margin_shortfall;
        ptrdiff_t b_margin_shortfall;
        ptrdiff_t b_decoder_distance_shortfall;
        ptrdiff_t b_decoder_deficit_shortfall;
        size_t    depth_remaining;
} replay_selector_joint_visited_t;

typedef struct {
        itty_feed_model_layer_state_snapshot_t *snapshot;
        replay_selector_joint_state_t           state;
        size_t                                 depth_remaining;
        size_t                                 steps_taken;
} replay_selector_joint_queue_entry_t;

typedef struct {
        bool                                    strict_mode;
        ptrdiff_t                               owner_margin_reserve;
        ptrdiff_t                               initial_b_margin;
        ptrdiff_t                               best_margin_after_b;
        ptrdiff_t                               best_margin_after_repair;
        ptrdiff_t                               best_a_margin_after_repair;
        size_t                                  a_safe_blocks;
        size_t                                  b_selecting_blocks;
        size_t                                  accepted_blocks;
        size_t                                  reject_no_candidates;
        size_t                                  reject_a_unsafe;
        size_t                                  reject_comp_erases;
        size_t                                  reject_insufficient_margin;
        size_t                                  best_steps;
        size_t                                  current_depth_limit;
        bool                                    found_any_candidate;
        bool                                    found_solution;
        itty_feed_model_layer_state_snapshot_t *best_snapshot;
        replay_selector_joint_state_t           best_state;
        replay_selector_joint_visited_t        visited[4096];
        size_t                                  visited_count;
        replay_selector_joint_queue_entry_t    queue[4096];
        size_t                                  queue_head;
        size_t                                  queue_tail;
} replay_selector_joint_search_t;

static bool
replay_measure_shadow_selector_inventory (itty_feed_model_t                      *selector_model,
                                          itty_bit_string_list_t                 *first_input,
                                          itty_bit_string_t                      *first_target,
                                          size_t                                  a_route,
                                          itty_bit_string_list_t                 *second_input,
                                          itty_bit_string_t                      *second_target,
                                          size_t                                  b_route,
                                          size_t                                  nodes,
                                          itty_feed_model_train_options_t const  *options,
                                          replay_shadow_selector_inventory_t     *inventory)
{
        itty_feed_model_decoder_objective_t *selector_routes = calloc (nodes, sizeof *selector_routes);

        if (!selector_routes)
                return false;

        *inventory = (replay_shadow_selector_inventory_t) { 0 };

        for (size_t route = 0; route < nodes; route++) {
                if (!itty_feed_model_measure_decoder_objective_for_node (selector_model,
                                                                        second_input,
                                                                        second_target,
                                                                        route,
                                                                        &selector_routes[route])) {
                        free (selector_routes);
                        return false;
                }
        }

        inventory->b_winner_route = 0;
        inventory->b_winner_popcount = selector_routes[0].selected_popcount;
        inventory->b_desired_popcount = selector_routes[b_route].selected_popcount;
        for (size_t route = 1; route < nodes; route++) {
                if (selector_routes[route].selected_popcount > inventory->b_winner_popcount) {
                        inventory->b_winner_route = route;
                        inventory->b_winner_popcount = selector_routes[route].selected_popcount;
                }
        }
        inventory->b_gap = inventory->b_winner_popcount > inventory->b_desired_popcount ?
                           inventory->b_winner_popcount - inventory->b_desired_popcount :
                           0;
        inventory->b_needed = inventory->b_gap + 1;
        inventory->b_margin = replay_route_margin_for_objectives (selector_routes, nodes, b_route);

        free (selector_routes);

        return itty_feed_model_measure_final_layer_selector_protection_for_node_with_guard (selector_model,
                                                                                             second_input,
                                                                                             second_target,
                                                                                             b_route,
                                                                                             first_input,
                                                                                             first_target,
                                                                                             a_route,
                                                                                             options,
                                                                                             &inventory->b_selector_inventory) &&
               itty_feed_model_measure_final_layer_selector_protection_for_node_with_guard (selector_model,
                                                                                             first_input,
                                                                                             first_target,
                                                                                             a_route,
                                                                                             second_input,
                                                                                             second_target,
                                                                                             b_route,
                                                                                             options,
                                                                                             &inventory->a_compensation_inventory);
}

static bool
replay_selector_joint_state_dominates (replay_selector_joint_state_t const *lhs,
                                       replay_selector_joint_state_t const *rhs)
{
        return lhs->a_margin_shortfall <= rhs->a_margin_shortfall &&
               lhs->b_margin_shortfall <= rhs->b_margin_shortfall &&
               lhs->b_decoder_distance_shortfall <= rhs->b_decoder_distance_shortfall &&
               lhs->b_decoder_deficit_shortfall <= rhs->b_decoder_deficit_shortfall;
}

static bool
replay_selector_joint_state_seen (replay_selector_joint_search_t const *search,
                                  replay_selector_joint_state_t const  *state,
                                  size_t                                depth_remaining)
{
        for (size_t index = 0; index < search->visited_count; index++) {
                replay_selector_joint_state_t seen_state = {
                        .a_selector.selected_by_popcount = search->visited[index].a_selected_route,
                        .b_selector.selected_by_popcount = search->visited[index].b_selected_route,
                        .a_margin_shortfall = search->visited[index].a_margin_shortfall,
                        .b_margin_shortfall = search->visited[index].b_margin_shortfall,
                        .b_decoder_distance_shortfall = search->visited[index].b_decoder_distance_shortfall,
                        .b_decoder_deficit_shortfall = search->visited[index].b_decoder_deficit_shortfall,
                };

                if (search->visited[index].a_selected_route == state->a_selector.selected_by_popcount &&
                    search->visited[index].b_selected_route == state->b_selector.selected_by_popcount &&
                    replay_selector_joint_state_dominates (&seen_state, state) &&
                    search->visited[index].depth_remaining >= depth_remaining)
                        return true;
        }

        return false;
}

static void
replay_selector_joint_state_mark_seen (replay_selector_joint_search_t const *search_in,
                                       replay_selector_joint_state_t const  *state,
                                       size_t                                depth_remaining)
{
        replay_selector_joint_search_t *search = (replay_selector_joint_search_t *) search_in;
        size_t replacement_index = (size_t) -1;

        for (size_t index = 0; index < search->visited_count; index++) {
                if (search->visited[index].a_selected_route != state->a_selector.selected_by_popcount ||
                    search->visited[index].b_selected_route != state->b_selector.selected_by_popcount)
                        continue;

                if (state->a_margin_shortfall <= search->visited[index].a_margin_shortfall &&
                    state->b_margin_shortfall <= search->visited[index].b_margin_shortfall &&
                    state->b_decoder_distance_shortfall <= search->visited[index].b_decoder_distance_shortfall &&
                    state->b_decoder_deficit_shortfall <= search->visited[index].b_decoder_deficit_shortfall &&
                    depth_remaining >= search->visited[index].depth_remaining) {
                        replacement_index = index;
                        break;
                }
        }

        if (replacement_index == (size_t) -1) {
                if (search->visited_count >= sizeof (search->visited) / sizeof (search->visited[0]))
                        return;
                replacement_index = search->visited_count++;
        }

        search->visited[replacement_index] = (replay_selector_joint_visited_t) {
                .a_selected_route = state->a_selector.selected_by_popcount,
                .b_selected_route = state->b_selector.selected_by_popcount,
                .a_margin = state->a_margin,
                .b_margin = state->b_margin,
                .a_margin_shortfall = state->a_margin_shortfall,
                .b_margin_shortfall = state->b_margin_shortfall,
                .b_decoder_distance_shortfall = state->b_decoder_distance_shortfall,
                .b_decoder_deficit_shortfall = state->b_decoder_deficit_shortfall,
                .depth_remaining = depth_remaining,
        };
}

static bool
replay_measure_selector_joint_state (itty_feed_model_t                             *decoder_model,
                                     itty_feed_model_t                             *selector_model,
                                     itty_bit_string_list_t                        *first_input,
                                     itty_bit_string_t                             *first_target,
                                     size_t                                         a_route,
                                     itty_bit_string_list_t                        *second_input,
                                     itty_bit_string_t                             *second_target,
                                     size_t                                         b_route,
                                     size_t                                         nodes,
                                     replay_selector_joint_state_t                 *state)
{
        itty_feed_model_decoder_objective_t *a_routes = calloc (nodes, sizeof *a_routes);
        itty_feed_model_decoder_objective_t *b_routes = calloc (nodes, sizeof *b_routes);

        if (!a_routes || !b_routes) {
                free (a_routes);
                free (b_routes);
                return false;
        }

        if (!itty_feed_model_measure_segment_node_selection (selector_model,
                                                             first_input,
                                                             first_target,
                                                             &state->a_selector) ||
            !itty_feed_model_measure_segment_node_selection (selector_model,
                                                             second_input,
                                                             second_target,
                                                             &state->b_selector) ||
            !replay_measure_shadow_selected_decode (decoder_model,
                                                   selector_model,
                                                   first_input,
                                                   first_target,
                                                   &state->a_selector,
                                                   &state->a_shadow) ||
            !itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                 first_input,
                                                                 first_target,
                                                                 a_route,
                                                                 &state->a_forced) ||
            !itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                 second_input,
                                                                 second_target,
                                                                 b_route,
                                                                 &state->b_forced)) {
                free (a_routes);
                free (b_routes);
                return false;
        }

        for (size_t route = 0; route < nodes; route++) {
                if (!itty_feed_model_measure_decoder_objective_for_node (selector_model,
                                                                        first_input,
                                                                        first_target,
                                                                        route,
                                                                        &a_routes[route]) ||
                    !itty_feed_model_measure_decoder_objective_for_node (selector_model,
                                                                        second_input,
                                                                        second_target,
                                                                        route,
                                                                        &b_routes[route])) {
                        free (a_routes);
                        free (b_routes);
                        return false;
                }
        }

        state->a_margin = replay_route_margin_for_objectives (a_routes, nodes, a_route);
        state->b_margin = replay_route_margin_for_objectives (b_routes, nodes, b_route);
        state->a_safe = state->a_shadow.selected_distance == 0 &&
                        state->a_forced.selected_distance == 0 &&
                        state->a_selector.selected_by_popcount == a_route;
        state->b_selecting = state->b_selector.selected_by_popcount == b_route &&
                             state->b_selector.popcount_gap >= 2;
        state->a_margin_shortfall = 0;
        state->b_margin_shortfall = 0;
        state->b_decoder_distance_shortfall = 0;
        state->b_decoder_deficit_shortfall = 0;

        free (a_routes);
        free (b_routes);
        return true;
}

static void
replay_selector_joint_state_compute_slack (replay_selector_joint_search_t const *search,
                                           replay_selector_joint_state_t        *state)
{
        ptrdiff_t b_margin_target = search->strict_mode ? 2 : search->initial_b_margin + 1;

        if (b_margin_target < 1)
                b_margin_target = 1;

        state->a_margin_shortfall = search->owner_margin_reserve > state->a_margin
                ? search->owner_margin_reserve - state->a_margin
                : 0;
        state->b_margin_shortfall = b_margin_target > state->b_margin
                ? b_margin_target - state->b_margin
                : 0;
        state->b_decoder_distance_shortfall = (ptrdiff_t) state->b_forced.selected_distance;
        state->b_decoder_deficit_shortfall = (ptrdiff_t) state->b_forced.false_negative_vote_deficit;
}

static bool
replay_selector_joint_state_better (replay_selector_joint_search_t const *search,
                                    replay_selector_joint_state_t const  *candidate,
                                    size_t                                steps)
{
        if (search->strict_mode) {
                if (!candidate->b_selecting)
                        return false;
                if (!search->found_solution)
                        return true;
                if (candidate->a_margin_shortfall != search->best_state.a_margin_shortfall)
                        return candidate->a_margin_shortfall < search->best_state.a_margin_shortfall;
                if (candidate->b_margin_shortfall != search->best_state.b_margin_shortfall)
                        return candidate->b_margin_shortfall < search->best_state.b_margin_shortfall;
                if (candidate->b_decoder_distance_shortfall != search->best_state.b_decoder_distance_shortfall)
                        return candidate->b_decoder_distance_shortfall < search->best_state.b_decoder_distance_shortfall;
                if (candidate->b_decoder_deficit_shortfall != search->best_state.b_decoder_deficit_shortfall)
                        return candidate->b_decoder_deficit_shortfall < search->best_state.b_decoder_deficit_shortfall;
                if (candidate->b_selector.popcount_gap != search->best_state.b_selector.popcount_gap)
                        return candidate->b_selector.popcount_gap > search->best_state.b_selector.popcount_gap;
                if (candidate->b_margin != search->best_state.b_margin)
                        return candidate->b_margin > search->best_state.b_margin;
                return steps < search->best_steps;
        }

        if (candidate->b_margin <= search->initial_b_margin)
                return false;
        if (!search->found_solution)
                return true;
        if (candidate->a_margin_shortfall != search->best_state.a_margin_shortfall)
                return candidate->a_margin_shortfall < search->best_state.a_margin_shortfall;
        if (candidate->b_margin_shortfall != search->best_state.b_margin_shortfall)
                return candidate->b_margin_shortfall < search->best_state.b_margin_shortfall;
        if (candidate->b_decoder_distance_shortfall != search->best_state.b_decoder_distance_shortfall)
                return candidate->b_decoder_distance_shortfall < search->best_state.b_decoder_distance_shortfall;
        if (candidate->b_decoder_deficit_shortfall != search->best_state.b_decoder_deficit_shortfall)
                return candidate->b_decoder_deficit_shortfall < search->best_state.b_decoder_deficit_shortfall;
        if (candidate->b_margin != search->best_state.b_margin)
                return candidate->b_margin > search->best_state.b_margin;
        if (candidate->b_selector.popcount_gap != search->best_state.b_selector.popcount_gap)
                return candidate->b_selector.popcount_gap > search->best_state.b_selector.popcount_gap;
        return steps < search->best_steps;
}

static bool
replay_selector_joint_search_enqueue (replay_selector_joint_search_t            *search,
                                      itty_feed_model_layer_state_snapshot_t    *snapshot,
                                      replay_selector_joint_state_t const       *state,
                                      size_t                                     depth_remaining,
                                      size_t                                     steps_taken)
{
        if (search->queue_tail >= sizeof (search->queue) / sizeof (search->queue[0])) {
                if (snapshot)
                        itty_feed_model_free_final_layer_state_snapshot (NULL, snapshot);
                return false;
        }

        search->queue[search->queue_tail++] = (replay_selector_joint_queue_entry_t) {
                .snapshot = snapshot,
                .state = *state,
                .depth_remaining = depth_remaining,
                .steps_taken = steps_taken,
        };
        return true;
}

static void
replay_selector_joint_search_consider_state (itty_feed_model_t              *selector_model,
                                             replay_selector_joint_search_t *search,
                                             replay_selector_joint_state_t const *state,
                                             size_t                          steps_taken)
{
        if (state->b_margin > search->best_margin_after_b)
                search->best_margin_after_b = state->b_margin;
        if (state->a_safe && state->a_margin >= search->owner_margin_reserve) {
                search->a_safe_blocks++;
                if (state->b_margin > search->best_margin_after_repair)
                        search->best_margin_after_repair = state->b_margin;
                if (state->a_margin > search->best_a_margin_after_repair)
                        search->best_a_margin_after_repair = state->a_margin;
                if (state->b_selecting)
                        search->b_selecting_blocks++;
                if (replay_selector_joint_state_better (search, state, steps_taken)) {
                        itty_feed_model_layer_state_snapshot_t *candidate_snapshot =
                                itty_feed_model_snapshot_final_layer_state (selector_model);

                        if (candidate_snapshot) {
                                if (search->best_snapshot)
                                        itty_feed_model_free_final_layer_state_snapshot (selector_model,
                                                                                          search->best_snapshot);
                                search->best_snapshot = candidate_snapshot;
                                search->best_state = *state;
                                search->best_steps = steps_taken;
                                search->found_solution = true;
                        }
                }
        }
}

static void
replay_search_selector_joint_blocks_internal (itty_feed_model_t               *decoder_model,
                                              itty_feed_model_t               *selector_model,
                                              itty_bit_string_list_t          *first_input,
                                              itty_bit_string_t               *first_target,
                                              size_t                           a_route,
                                              itty_bit_string_list_t          *second_input,
                                              itty_bit_string_t               *second_target,
                                              size_t                           b_route,
                                              size_t                           nodes,
                                              itty_feed_model_train_options_t *options,
                                              replay_selector_joint_search_t  *search)
{
        while (search->queue_head < search->queue_tail) {
                replay_selector_joint_queue_entry_t entry = search->queue[search->queue_head++];

                if (!entry.snapshot)
                        continue;

                itty_feed_model_restore_final_layer_state_snapshot (selector_model, entry.snapshot);
                entry.snapshot = NULL;

                replay_selector_joint_search_consider_state (selector_model,
                                                             search,
                                                             &entry.state,
                                                             entry.steps_taken);

                if (entry.depth_remaining == 0)
                        continue;

                for (size_t action = 0; action < 2; action++) {
                        itty_feed_model_layer_state_snapshot_t *snapshot =
                                itty_feed_model_snapshot_final_layer_state (selector_model);
                        itty_feed_model_selector_protection_summary_t step = { 0 };
                        replay_selector_joint_state_t child_state = { 0 };
                        bool trained = false;

                        if (!snapshot)
                                continue;

                        if (action == 0) {
                                trained = itty_feed_model_train_final_layer_selector_margin_for_node (selector_model,
                                                                                                       second_input,
                                                                                                       second_target,
                                                                                                       b_route,
                                                                                                       options,
                                                                                                       &step);
                        } else {
                                trained = itty_feed_model_train_final_layer_selector_margin_for_node (selector_model,
                                                                                                       first_input,
                                                                                                       first_target,
                                                                                                       a_route,
                                                                                                       options,
                                                                                                       &step);
                        }

                        if (trained && step.accepted) {
                                search->found_any_candidate = true;
                                if (replay_measure_selector_joint_state (decoder_model,
                                                                        selector_model,
                                                                        first_input,
                                                                        first_target,
                                                                        a_route,
                                                                        second_input,
                                                                        second_target,
                                                                        b_route,
                                                                        nodes,
                                                                        &child_state)) {
                                        replay_selector_joint_state_compute_slack (search, &child_state);
                                }
                                if (trained && step.accepted &&
                                    !replay_selector_joint_state_seen (search,
                                                                       &child_state,
                                                                       entry.depth_remaining - 1)) {
                                        itty_feed_model_layer_state_snapshot_t *child_snapshot =
                                                itty_feed_model_snapshot_final_layer_state (selector_model);

                                        replay_selector_joint_state_mark_seen (search,
                                                                              &child_state,
                                                                              entry.depth_remaining - 1);
                                        if (child_snapshot) {
                                                replay_selector_joint_search_enqueue (search,
                                                                                      child_snapshot,
                                                                                      &child_state,
                                                                                      entry.depth_remaining - 1,
                                                                                      entry.steps_taken + 1);
                                        }
                                }
                        }

                        itty_feed_model_restore_final_layer_state_snapshot (selector_model, snapshot);
                }
        }
}

static bool
replay_search_selector_joint_blocks (itty_feed_model_t                     *decoder_model,
                                     itty_feed_model_t                     *selector_model,
                                     itty_bit_string_list_t                *first_input,
                                     itty_bit_string_t                     *first_target,
                                     size_t                                 a_route,
                                     itty_bit_string_list_t                *second_input,
                                     itty_bit_string_t                     *second_target,
                                     size_t                                 b_route,
                                     size_t                                 nodes,
                                     ptrdiff_t                              owner_margin_reserve,
                                     ptrdiff_t                              initial_b_margin,
                                     size_t                                 max_depth,
                                     bool                                   strict_mode,
                                     itty_feed_model_train_options_t       *options,
                                     replay_selector_joint_search_t       *search)
{
        replay_selector_joint_state_t root_state = { 0 };
        itty_feed_model_layer_state_snapshot_t *root_snapshot = NULL;

        if (search)
                *search = (replay_selector_joint_search_t) {
                        .strict_mode = strict_mode,
                        .owner_margin_reserve = owner_margin_reserve,
                        .initial_b_margin = initial_b_margin,
                        .best_margin_after_b = -9999,
                        .best_margin_after_repair = -9999,
                        .best_a_margin_after_repair = -9999,
                        .current_depth_limit = max_depth,
                };

        if (!replay_measure_selector_joint_state (decoder_model,
                                                 selector_model,
                                                 first_input,
                                                 first_target,
                                                 a_route,
                                                 second_input,
                                                 second_target,
                                                 b_route,
                                                 nodes,
                                                 &root_state))
                return false;
        replay_selector_joint_state_compute_slack (search, &root_state);

        root_snapshot = itty_feed_model_snapshot_final_layer_state (selector_model);
        if (!root_snapshot)
                return false;

        replay_selector_joint_state_mark_seen (search, &root_state, max_depth);
        if (!replay_selector_joint_search_enqueue (search,
                                                  root_snapshot,
                                                  &root_state,
                                                  max_depth,
                                                  0))
                return false;

        replay_search_selector_joint_blocks_internal (decoder_model,
                                                      selector_model,
                                                      first_input,
                                                      first_target,
                                                      a_route,
                                                      second_input,
                                                      second_target,
                                                      b_route,
                                                      nodes,
                                                      options,
                                                      search);

        if (search && search->found_solution && search->best_snapshot) {
                itty_feed_model_restore_final_layer_state_snapshot (selector_model,
                                                                     search->best_snapshot);
                search->best_snapshot = NULL;
                search->accepted_blocks = 1;
                return true;
        }

        return true;
}

static bool
replay_apply_selector_protection_block (itty_feed_model_t                     *model,
                                        itty_bit_string_list_t                *input,
                                        itty_bit_string_t                     *target,
                                        size_t                                 owner_route,
                                        itty_bit_string_list_t                *guard_input,
                                        itty_bit_string_t                     *guard_target,
                                        size_t                                 guard_route,
                                        size_t                                 route_count,
                                        ptrdiff_t                              owner_margin_reserve,
                                        size_t                                 max_repairs,
                                        itty_feed_model_train_options_t const *options,
                                        replay_selector_block_summary_t       *summary)
{
        itty_feed_model_decoder_objective_t *routes = calloc (route_count, sizeof *routes);
        itty_feed_model_decoder_objective_t owner_objective = { 0 };
        itty_feed_model_decoder_objective_t global_objective = { 0 };

        if (!routes)
                return false;
        if (summary)
                *summary = (replay_selector_block_summary_t) { 0 };

        for (size_t route = 0; route < route_count; route++) {
                if (!replay_measure_decoder_objective_for_node_with_options (model,
                                                                             input,
                                                                             target,
                                                                             route,
                                                                             options,
                                                                             &routes[route])) {
                        free (routes);
                        return false;
                }
        }
        if (!replay_measure_decoder_objective_for_node_with_options (model,
                                                                     input,
                                                                     target,
                                                                     owner_route,
                                                                     options,
                                                                     &owner_objective) ||
            !replay_measure_decoder_objective_with_options (model,
                                                            input,
                                                            target,
                                                            options,
                                                            &global_objective)) {
                free (routes);
                return false;
        }

        if (summary) {
                summary->margin_before = replay_route_margin_for_objectives (routes, route_count, owner_route);
                summary->margin_after = summary->margin_before;
                summary->owner_route_safe = replay_route_is_a_safe (&owner_objective);
                summary->global_safe = global_objective.selected_distance == 0;
        }

        for (size_t repair = 0; repair < max_repairs; repair++) {
                ptrdiff_t current_margin = replay_route_margin_for_objectives (routes, route_count, owner_route);

                if (current_margin >= owner_margin_reserve &&
                    replay_route_is_a_safe (&owner_objective) &&
                    global_objective.selected_distance == 0)
                        break;

                itty_feed_model_selector_protection_summary_t step = { 0 };
                bool trained = guard_input && guard_target ?
                               itty_feed_model_train_final_layer_selector_protection_for_node_with_guard (model,
                                                                                                           input,
                                                                                                           target,
                                                                                                           owner_route,
                                                                                                           guard_input,
                                                                                                           guard_target,
                                                                                                           guard_route,
                                                                                                           options,
                                                                                                           &step) :
                               itty_feed_model_train_final_layer_selector_protection_for_node (model,
                                                                                              input,
                                                                                              target,
                                                                                              owner_route,
                                                                                              options,
                                                                                              &step);
                if (!trained)
                        break;
                if (!step.accepted)
                        break;

                if (summary) {
                        summary->accepted_repairs++;
                        if (step.kind == ITTY_FEED_MODEL_SELECTOR_PROTECTION_OWNER_LIFT)
                                summary->owner_lifts++;
                        else if (step.kind == ITTY_FEED_MODEL_SELECTOR_PROTECTION_COMPETITOR_SUPPRESS)
                                summary->competitor_suppresses++;
                }

                for (size_t route = 0; route < route_count; route++) {
                        if (!replay_measure_decoder_objective_for_node_with_options (model,
                                                                                     input,
                                                                                     target,
                                                                                     route,
                                                                                     options,
                                                                                     &routes[route])) {
                                free (routes);
                                return false;
                        }
                }
                if (!replay_measure_decoder_objective_for_node_with_options (model,
                                                                             input,
                                                                             target,
                                                                             owner_route,
                                                                             options,
                                                                             &owner_objective) ||
                    !replay_measure_decoder_objective_with_options (model,
                                                                    input,
                                                                    target,
                                                                    options,
                                                                    &global_objective)) {
                        free (routes);
                        return false;
                }
                if (summary) {
                        summary->margin_after = replay_route_margin_for_objectives (routes, route_count, owner_route);
                        summary->owner_route_safe = replay_route_is_a_safe (&owner_objective);
                        summary->global_safe = global_objective.selected_distance == 0;
                }
        }

        free (routes);
        return true;
}

static bool
replay_apply_selector_margin_block (itty_feed_model_t                     *model,
                                    itty_bit_string_list_t                *input,
                                    itty_bit_string_t                     *target,
                                    size_t                                 owner_route,
                                    size_t                                 route_count,
                                    ptrdiff_t                              owner_margin_target,
                                    size_t                                 max_repairs,
                                    itty_feed_model_train_options_t const *options,
                                    replay_selector_block_summary_t       *summary)
{
        itty_feed_model_decoder_objective_t *routes = calloc (route_count, sizeof *routes);
        if (!routes)
                return false;
        if (summary)
                *summary = (replay_selector_block_summary_t) { 0 };

        for (size_t route = 0; route < route_count; route++) {
                if (!replay_measure_decoder_objective_for_node_with_options (model,
                                                                             input,
                                                                             target,
                                                                             route,
                                                                             options,
                                                                             &routes[route])) {
                        free (routes);
                        return false;
                }
        }

        if (summary) {
                summary->margin_before = replay_route_margin_for_objectives (routes, route_count, owner_route);
                summary->margin_after = summary->margin_before;
        }

        for (size_t repair = 0; repair < max_repairs; repair++) {
                ptrdiff_t current_margin = replay_route_margin_for_objectives (routes, route_count, owner_route);
                if (current_margin >= owner_margin_target)
                        break;

                itty_feed_model_selector_protection_summary_t step = { 0 };
                bool trained = itty_feed_model_train_final_layer_selector_margin_for_node (model,
                                                                                           input,
                                                                                           target,
                                                                                           owner_route,
                                                                                           options,
                                                                                           &step);
                if (!trained || !step.accepted)
                        break;

                if (summary) {
                        summary->accepted_repairs++;
                        if (step.kind == ITTY_FEED_MODEL_SELECTOR_PROTECTION_OWNER_LIFT)
                                summary->owner_lifts++;
                        else if (step.kind == ITTY_FEED_MODEL_SELECTOR_PROTECTION_COMPETITOR_SUPPRESS)
                                summary->competitor_suppresses++;
                }

                for (size_t route = 0; route < route_count; route++) {
                        if (!replay_measure_decoder_objective_for_node_with_options (model,
                                                                                     input,
                                                                                     target,
                                                                                     route,
                                                                                     options,
                                                                                     &routes[route])) {
                                free (routes);
                                return false;
                        }
                }
                if (summary)
                        summary->margin_after = replay_route_margin_for_objectives (routes, route_count, owner_route);
        }

        free (routes);
        return true;
}

static size_t
replay_choose_a_route (itty_feed_model_decoder_objective_t const              *objectives,
                       size_t                                                   count,
                       itty_feed_model_segment_node_selection_summary_t const  *selection)
{
        size_t best = 0;

        for (size_t route = 1; route < count; route++) {
                itty_feed_model_decoder_objective_t const *candidate = &objectives[route];
                itty_feed_model_decoder_objective_t const *current = &objectives[best];

                if (candidate->selected_distance < current->selected_distance) {
                        best = route;
                        continue;
                }
                if (candidate->selected_distance > current->selected_distance)
                        continue;
                if (candidate->false_positive_vote_excess < current->false_positive_vote_excess) {
                        best = route;
                        continue;
                }
                if (candidate->false_positive_vote_excess > current->false_positive_vote_excess)
                        continue;
                if (candidate->false_negative_vote_deficit < current->false_negative_vote_deficit) {
                        best = route;
                        continue;
                }
                if (candidate->false_negative_vote_deficit > current->false_negative_vote_deficit)
                        continue;
                if (selection->selected_by_popcount == route &&
                    selection->selected_by_popcount != best)
                        best = route;
        }

        return best;
}

static size_t
replay_choose_b_route (itty_feed_model_decoder_objective_t const              *objectives,
                       size_t                                                   count,
                       size_t                                                   a_route,
                       itty_feed_model_segment_node_selection_summary_t const  *selection)
{
        size_t best = (size_t) -1;

        for (size_t route = 0; route < count; route++) {
                if (route == a_route)
                        continue;
                if (best == (size_t) -1) {
                        best = route;
                        continue;
                }

                itty_feed_model_decoder_objective_t const *candidate = &objectives[route];
                itty_feed_model_decoder_objective_t const *current = &objectives[best];

                if (candidate->selected_distance < current->selected_distance) {
                        best = route;
                        continue;
                }
                if (candidate->selected_distance > current->selected_distance)
                        continue;
                if (candidate->false_negative_vote_deficit < current->false_negative_vote_deficit) {
                        best = route;
                        continue;
                }
                if (candidate->false_negative_vote_deficit > current->false_negative_vote_deficit)
                        continue;
                if (selection->best_by_target_distance == route &&
                    selection->best_by_target_distance != best)
                        best = route;
        }

        return best == (size_t) -1 ? a_route : best;
}

static size_t
replay_choose_c_route (itty_feed_model_decoder_objective_t const *objectives,
                       size_t                                      count,
                       size_t                                      a_route,
                       size_t                                      b_route)
{
        size_t best = (size_t) -1;

        for (size_t route = 0; route < count; route++) {
                if (route == a_route || route == b_route)
                        continue;
                if (best == (size_t) -1) {
                        best = route;
                        continue;
                }

                itty_feed_model_decoder_objective_t const *candidate = &objectives[route];
                itty_feed_model_decoder_objective_t const *current = &objectives[best];

                if (candidate->selected_distance < current->selected_distance) {
                        best = route;
                        continue;
                }
                if (candidate->selected_distance > current->selected_distance)
                        continue;
                if (candidate->false_negative_vote_deficit < current->false_negative_vote_deficit) {
                        best = route;
                        continue;
                }
                if (candidate->false_negative_vote_deficit > current->false_negative_vote_deficit)
                        continue;
                if (candidate->false_positive_vote_excess < current->false_positive_vote_excess)
                        best = route;
        }

        return best == (size_t) -1 ? a_route : best;
}

static void
test_itty_feed_model_segment_replay_route_assignment_diagnostic (void)
{
        struct replay_route_case {
                size_t      nodes;
                bool        sparse_init;
                bool        varied_rotation;
                char const *label;
        } cases[] = {
                { 2, false, false, "2-zero-none" },
                { 4, false, false, "4-zero-none" },
                { 8, true,  false, "8-sparse-none" },
                { 8, false, false, "8-zero-none" },
                { 4, true,  false, "4-sparse-none" }
        };
        size_t selector_margin = 2;

        printf ("segment replay route assignments:\n");
        printf ("case a-route a-class b-route b-class b-dist b-def b-selected b-best selector-margin margin-met\n");

        for (size_t case_index = 0; case_index < sizeof (cases) / sizeof (cases[0]); case_index++) {
                struct replay_route_case const *cfg = &cases[case_index];
                itty_feed_model_t *model = itty_feed_model_new (2,
                                                                cfg->nodes,
                                                                1,
                                                                1);
                itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
                itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
                itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
                itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
                itty_feed_model_refreshed_projected_repair_options_t first_options = {
                        .batch_size = 64,
                        .max_rounds = 32,
                        .max_layer_flips_per_batch = 128
                };
                itty_feed_model_train_options_t oracle_options = {
                        .max_flips = 8,
                        .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                        .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
                };
                itty_feed_model_segment_training_summary_t first_summary;
                itty_feed_model_segment_node_selection_summary_t a_selection = { 0 };
                itty_feed_model_segment_node_selection_summary_t b_selection = { 0 };
                itty_feed_model_decoder_objective_t *a_routes = calloc (cfg->nodes,
                                                                        sizeof *a_routes);
                itty_feed_model_decoder_objective_t *b_routes = calloc (cfg->nodes,
                                                                        sizeof *b_routes);

                assert (a_routes);
                assert (b_routes);

                itty_feed_model_set_decoder (model,
                                             ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                if (cfg->varied_rotation) {
                        for (size_t model_layer = 0; model_layer < 2; model_layer++)
                                itty_feed_model_set_layer_rotation (model,
                                                                    model_layer,
                                                                    model_layer + 1);
                }
                if (cfg->sparse_init)
                        assert (itty_feed_model_randomize_masks (model,
                                                                 0xa100 + case_index,
                                                                 1,
                                                                 8));

                assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                        first_input,
                                                                                        first_target,
                                                                                        &first_options,
                                                                                        &first_summary));
                assert (itty_feed_model_train_final_layer_transaction_scaffold (model,
                                                                                first_input,
                                                                                first_target,
                                                                                second_input,
                                                                                second_target,
                                                                                &oracle_options,
                                                                                8,
                                                                                NULL,
                                                                                0,
                                                                                &(itty_feed_model_transaction_scaffold_summary_t){0}));
                assert (itty_feed_model_measure_segment_node_selection (model,
                                                                        first_input,
                                                                        first_target,
                                                                        &a_selection));
                assert (itty_feed_model_measure_segment_node_selection (model,
                                                                        second_input,
                                                                        second_target,
                                                                        &b_selection));

                for (size_t route = 0; route < cfg->nodes; route++) {
                        assert (itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                    first_input,
                                                                                    first_target,
                                                                                    route,
                                                                                    &a_routes[route]));
                        assert (itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                    second_input,
                                                                                    second_target,
                                                                                    route,
                                                                                    &b_routes[route]));
                }

                size_t a_route = replay_choose_a_route (a_routes,
                                                       cfg->nodes,
                                                       &a_selection);
                size_t b_route = replay_choose_b_route (b_routes,
                                                       cfg->nodes,
                                                       a_route,
                                                       &b_selection);
                replay_route_class_t a_class = replay_route_is_a_safe (&a_routes[a_route]) ?
                                               REPLAY_ROUTE_STABLE :
                                               REPLAY_ROUTE_UNSAFE;
                replay_route_class_t b_class = replay_classify_route (&a_routes[b_route],
                                                                      &b_routes[b_route],
                                                                      &b_selection,
                                                                      b_route,
                                                                      selector_margin);
                bool margin_met = b_selection.selected_by_popcount == b_route &&
                                  b_selection.popcount_gap >= selector_margin;

                printf ("%s %zu %s %zu %s %zu %zu %s %s %zu %s\n",
                        cfg->label,
                        a_route,
                        replay_route_class_name (a_class),
                        b_route,
                        replay_route_class_name (b_class),
                        b_routes[b_route].selected_distance,
                        b_routes[b_route].false_negative_vote_deficit,
                        b_selection.selected_by_popcount == b_route ? "yes" : "no",
                        b_selection.best_by_target_distance == b_route ? "yes" : "no",
                        selector_margin,
                        margin_met ? "yes" : "no");

                free (a_routes);
                free (b_routes);
                itty_bit_string_free (first_target);
                itty_bit_string_free (second_target);
                itty_bit_string_list_free (first_input);
                itty_bit_string_list_free (second_input);
                itty_feed_model_free (model);
        }
}

static void
test_itty_feed_model_segment_replay_route_aware_training_diagnostic (void)
{
        struct replay_route_case {
                size_t      nodes;
                bool        sparse_init;
                bool        varied_rotation;
                char const *label;
        } cases[] = {
                { 2, false, false, "2-zero-none" },
                { 4, false, false, "4-zero-none" },
                { 8, true,  false, "8-sparse-none" },
                { 8, false, false, "8-zero-none" },
                { 4, true,  false, "4-sparse-none" }
        };
        size_t selector_margin = 2;

        printf ("segment replay route-aware training:\n");
        printf ("case a-route b-route steps A-forced A-global B-forced B-global B-sel margin-meets flips\n");

        for (size_t case_index = 0; case_index < sizeof (cases) / sizeof (cases[0]); case_index++) {
                struct replay_route_case const *cfg = &cases[case_index];
                itty_feed_model_t *model = itty_feed_model_new (2,
                                                                cfg->nodes,
                                                                1,
                                                                1);
                itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
                itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
                itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
                itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
                itty_feed_model_refreshed_projected_repair_options_t first_options = {
                        .batch_size = 64,
                        .max_rounds = 32,
                        .max_layer_flips_per_batch = 128
                };
                itty_feed_model_train_options_t oracle_options = {
                        .max_flips = 8,
                        .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                        .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
                };
                itty_feed_model_segment_training_summary_t first_summary;
                itty_feed_model_segment_node_selection_summary_t a_selection = { 0 };
                itty_feed_model_segment_node_selection_summary_t b_selection = { 0 };
                itty_feed_model_decoder_objective_t *a_routes = calloc (cfg->nodes,
                                                                        sizeof *a_routes);
                itty_feed_model_decoder_objective_t *b_routes = calloc (cfg->nodes,
                                                                        sizeof *b_routes);
                itty_feed_model_decoder_objective_t a_global = { 0 };
                itty_feed_model_decoder_objective_t b_global = { 0 };
                size_t total_flips = 0;
                size_t accepted_steps = 0;

                assert (a_routes);
                assert (b_routes);

                itty_feed_model_set_decoder (model,
                                             ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                if (cfg->varied_rotation) {
                        for (size_t model_layer = 0; model_layer < 2; model_layer++)
                                itty_feed_model_set_layer_rotation (model,
                                                                    model_layer,
                                                                    model_layer + 1);
                }
                if (cfg->sparse_init)
                        assert (itty_feed_model_randomize_masks (model,
                                                                 0xa100 + case_index,
                                                                 1,
                                                                 8));

                assert (itty_feed_model_train_segment_condense_quota_repair_projection (model,
                                                                                        first_input,
                                                                                        first_target,
                                                                                        &first_options,
                                                                                        &first_summary));

                assert (itty_feed_model_measure_segment_node_selection (model,
                                                                        first_input,
                                                                        first_target,
                                                                        &a_selection));
                assert (itty_feed_model_measure_segment_node_selection (model,
                                                                        second_input,
                                                                        second_target,
                                                                        &b_selection));
                for (size_t route = 0; route < cfg->nodes; route++) {
                        assert (itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                    first_input,
                                                                                    first_target,
                                                                                    route,
                                                                                    &a_routes[route]));
                        assert (itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                    second_input,
                                                                                    second_target,
                                                                                    route,
                                                                                    &b_routes[route]));
                }

                size_t a_route = replay_choose_a_route (a_routes,
                                                       cfg->nodes,
                                                       &a_selection);
                size_t b_route = replay_choose_b_route (b_routes,
                                                       cfg->nodes,
                                                       a_route,
                                                       &b_selection);

                for (size_t step = 0; step < 8; step++) {
                        itty_feed_model_train_stats_t step_stats = { 0 };
                        itty_feed_model_decoder_objective_t a_forced_before = { 0 };
                        itty_feed_model_decoder_objective_t a_global_before = { 0 };
                        itty_feed_model_decoder_objective_t b_forced_before = { 0 };
                        itty_feed_model_decoder_objective_t b_global_before = { 0 };
                        itty_feed_model_decoder_objective_t a_forced_after = { 0 };
                        itty_feed_model_decoder_objective_t a_global_after = { 0 };
                        itty_feed_model_decoder_objective_t b_forced_after = { 0 };
                        itty_feed_model_decoder_objective_t b_global_after = { 0 };

                        assert (itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                    first_input,
                                                                                    first_target,
                                                                                    a_route,
                                                                                    &a_forced_before));
                        assert (itty_feed_model_measure_decoder_objective (model,
                                                                          first_input,
                                                                          first_target,
                                                                          &a_global_before));
                        assert (itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                    second_input,
                                                                                    second_target,
                                                                                    b_route,
                                                                                    &b_forced_before));
                        assert (itty_feed_model_measure_decoder_objective (model,
                                                                          second_input,
                                                                          second_target,
                                                                          &b_global_before));

                        if (!itty_feed_model_train_final_layer_with_suffix_oracle_for_node (model,
                                                                                             second_input,
                                                                                             second_target,
                                                                                             b_route,
                                                                                             &oracle_options,
                                                                                             &step_stats))
                                break;
                        if (step_stats.flips == 0)
                                break;

                        assert (itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                    first_input,
                                                                                    first_target,
                                                                                    a_route,
                                                                                    &a_forced_after));
                        assert (itty_feed_model_measure_decoder_objective (model,
                                                                          first_input,
                                                                          first_target,
                                                                          &a_global_after));
                        assert (itty_feed_model_measure_decoder_objective_for_node (model,
                                                                                    second_input,
                                                                                    second_target,
                                                                                    b_route,
                                                                                    &b_forced_after));
                        assert (itty_feed_model_measure_decoder_objective (model,
                                                                          second_input,
                                                                          second_target,
                                                                          &b_global_after));

                        if (!replay_route_is_a_safe (&a_forced_after) ||
                            a_global_after.selected_distance > 0)
                                break;

                        if (b_forced_after.selected_distance < b_forced_before.selected_distance ||
                            b_forced_after.false_negative_vote_deficit < b_forced_before.false_negative_vote_deficit) {
                                accepted_steps++;
                                total_flips += step_stats.flips;
                        } else {
                                break;
                        }
                }

                itty_feed_model_t *report_model = itty_feed_model_new (2,
                                                                       cfg->nodes,
                                                                       1,
                                                                       1);
                itty_bit_string_list_t *report_first_input = create_input_with_count (1, 0);
                itty_bit_string_list_t *report_second_input = create_input_with_count (1, 1);
                itty_bit_string_t *report_first_target = create_bit_string ((size_t) 1 << 3);
                itty_bit_string_t *report_second_target = create_bit_string (create_half_populated_word ());
                itty_feed_model_segment_training_summary_t report_first_summary;

                itty_feed_model_set_decoder (report_model,
                                             ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                if (cfg->varied_rotation) {
                        for (size_t model_layer = 0; model_layer < 2; model_layer++)
                                itty_feed_model_set_layer_rotation (report_model,
                                                                    model_layer,
                                                                    model_layer + 1);
                }
                if (cfg->sparse_init)
                        assert (itty_feed_model_randomize_masks (report_model,
                                                                 0xa100 + case_index,
                                                                 1,
                                                                 8));
                assert (itty_feed_model_train_segment_condense_quota_repair_projection (report_model,
                                                                                        report_first_input,
                                                                                        report_first_target,
                                                                                        &first_options,
                                                                                        &report_first_summary));
                for (size_t step = 0; step < accepted_steps; step++) {
                        itty_feed_model_train_stats_t replayed_step = { 0 };
                        assert (itty_feed_model_train_final_layer_with_suffix_oracle_for_node (report_model,
                                                                                               report_second_input,
                                                                                               report_second_target,
                                                                                               b_route,
                                                                                               &oracle_options,
                                                                                               &replayed_step));
                }

                assert (itty_feed_model_measure_segment_node_selection (report_model,
                                                                        report_second_input,
                                                                        report_second_target,
                                                                        &b_selection));
                assert (itty_feed_model_measure_decoder_objective_for_node (report_model,
                                                                            report_first_input,
                                                                            report_first_target,
                                                                            a_route,
                                                                            &a_routes[a_route]));
                assert (itty_feed_model_measure_decoder_objective (report_model,
                                                                  report_first_input,
                                                                  report_first_target,
                                                                  &a_global));
                assert (itty_feed_model_measure_decoder_objective_for_node (report_model,
                                                                            report_second_input,
                                                                            report_second_target,
                                                                            b_route,
                                                                            &b_routes[b_route]));
                assert (itty_feed_model_measure_decoder_objective (report_model,
                                                                  report_second_input,
                                                                  report_second_target,
                                                                  &b_global));

                printf ("%s %zu %zu %zu %zu/%zu %zu %zu/%zu %zu/%zu %s %s %zu\n",
                        cfg->label,
                        a_route,
                        b_route,
                        accepted_steps,
                        a_routes[a_route].selected_distance,
                        a_routes[a_route].false_positive_vote_excess,
                        a_global.selected_distance,
                        b_routes[b_route].selected_distance,
                        b_routes[b_route].false_negative_vote_deficit,
                        b_global.selected_distance,
                        b_global.false_negative_vote_deficit,
                        b_selection.selected_by_popcount == b_route ? "yes" : "no",
                        b_selection.popcount_gap >= selector_margin ? "yes" : "no",
                        total_flips);

                itty_bit_string_free (report_first_target);
                itty_bit_string_free (report_second_target);
                itty_bit_string_list_free (report_first_input);
                itty_bit_string_list_free (report_second_input);
                itty_feed_model_free (report_model);
                free (a_routes);
                free (b_routes);
                itty_bit_string_free (first_target);
                itty_bit_string_free (second_target);
                itty_bit_string_list_free (first_input);
                itty_bit_string_list_free (second_input);
                itty_feed_model_free (model);
        }
}

static void
test_itty_feed_model_segment_replay_route_owned_training_diagnostic (void)
{
        struct replay_route_case {
                size_t      nodes;
                bool        sparse_init;
                bool        varied_rotation;
                char const *label;
        } cases[] = {
                { 2, false, false, "2-zero-none" },
                { 4, false, false, "4-zero-none" },
                { 8, true,  false, "8-sparse-none" },
                { 8, false, false, "8-zero-none" },
                { 4, true,  false, "4-sparse-none" }
        };
        size_t selector_margin = 2;
        ptrdiff_t owner_margin_reserve = 2;

        printf ("segment replay route-owned training:\n");
        printf ("case a-route b-route s0-kind s0-acc s0-margin s1-acc s1-topup s1-rej A-sel before->after A-margin before->after B-margin before->after A-fr/global B-fr/global committed-A/B probe-A/B probe-margins s2-acc B-sel margin\n");

        for (size_t case_index = 0; case_index < sizeof (cases) / sizeof (cases[0]); case_index++) {
                struct replay_route_case const *cfg = &cases[case_index];
                itty_feed_model_t *seed_model = itty_feed_model_new (2,
                                                                     cfg->nodes,
                                                                     1,
                                                                     1);
                itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
                itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
                itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
                itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
                itty_feed_model_refreshed_projected_repair_options_t first_options = {
                        .batch_size = 64,
                        .max_rounds = 32,
                        .max_layer_flips_per_batch = 128
                };
                itty_feed_model_train_options_t oracle_options = {
                        .max_flips = 8,
                        .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                        .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
                };
                itty_feed_model_segment_training_summary_t first_summary;
                itty_feed_model_segment_node_selection_summary_t a_selection = { 0 };
                itty_feed_model_segment_node_selection_summary_t b_selection = { 0 };
                itty_feed_model_decoder_objective_t *a_routes = calloc (cfg->nodes, sizeof *a_routes);
                itty_feed_model_decoder_objective_t *b_routes = calloc (cfg->nodes, sizeof *b_routes);
                size_t a_route;
                size_t b_route;
                size_t stage1_accepted = 0;
                size_t stage1_topups = 0;
                char const *stage1_reject = "none";
                size_t stage2_accepted = 0;
                replay_selector_block_summary_t stage0_summary = { 0 };
                size_t stage1_a_selected_before = 0;
                size_t stage1_a_selected_after = 0;
                ptrdiff_t stage1_a_margin_before = 0;
                ptrdiff_t stage1_a_margin_after = 0;
                ptrdiff_t stage1_b_margin_before = 0;
                ptrdiff_t stage1_b_margin_after = 0;
                itty_feed_model_decoder_objective_t stage1_committed_a_forced = { 0 };
                itty_feed_model_decoder_objective_t stage1_committed_a_global = { 0 };
                itty_feed_model_decoder_objective_t stage1_committed_b_forced = { 0 };
                itty_feed_model_decoder_objective_t stage1_committed_b_global = { 0 };
                ptrdiff_t stage1_committed_a_margin = 0;
                ptrdiff_t stage1_committed_b_margin = 0;
                itty_feed_model_decoder_objective_t stage1_probe_b_after_step = { 0 };
                itty_feed_model_decoder_objective_t stage1_probe_b_after_restore = { 0 };
                itty_feed_model_decoder_objective_t stage1_probe_a_after_restore = { 0 };
                ptrdiff_t stage1_probe_a_margin_after_step = 0;
                ptrdiff_t stage1_probe_a_margin_after_restore = 0;
                itty_feed_model_decoder_objective_t final_a_forced = { 0 };
                itty_feed_model_decoder_objective_t final_a_global = { 0 };
                itty_feed_model_decoder_objective_t final_b_forced = { 0 };
                itty_feed_model_decoder_objective_t final_b_global = { 0 };

                assert (a_routes && b_routes);

                itty_feed_model_set_decoder (seed_model,
                                             ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                if (cfg->varied_rotation) {
                        for (size_t model_layer = 0; model_layer < 2; model_layer++)
                                itty_feed_model_set_layer_rotation (seed_model,
                                                                    model_layer,
                                                                    model_layer + 1);
                }
                if (cfg->sparse_init)
                        assert (itty_feed_model_randomize_masks (seed_model,
                                                                 0xa100 + case_index,
                                                                 1,
                                                                 8));
                assert (itty_feed_model_train_segment_condense_quota_repair_projection (seed_model,
                                                                                        first_input,
                                                                                        first_target,
                                                                                        &first_options,
                                                                                        &first_summary));
                assert (itty_feed_model_measure_segment_node_selection (seed_model,
                                                                        first_input,
                                                                        first_target,
                                                                        &a_selection));
                assert (itty_feed_model_measure_segment_node_selection (seed_model,
                                                                        second_input,
                                                                        second_target,
                                                                        &b_selection));
                for (size_t route = 0; route < cfg->nodes; route++) {
                        assert (itty_feed_model_measure_decoder_objective_for_node (seed_model,
                                                                                    first_input,
                                                                                    first_target,
                                                                                    route,
                                                                                    &a_routes[route]));
                        assert (itty_feed_model_measure_decoder_objective_for_node (seed_model,
                                                                                    second_input,
                                                                                    second_target,
                                                                                    route,
                                                                                    &b_routes[route]));
                }
                a_route = replay_choose_a_route (a_routes, cfg->nodes, &a_selection);
                b_route = replay_choose_b_route (b_routes, cfg->nodes, a_route, &b_selection);

                itty_feed_model_t *stage1_model = itty_feed_model_new (2,
                                                                       cfg->nodes,
                                                                       1,
                                                                       1);
                itty_bit_string_list_t *s1_first_input = create_input_with_count (1, 0);
                itty_bit_string_list_t *s1_second_input = create_input_with_count (1, 1);
                itty_bit_string_t *s1_first_target = create_bit_string ((size_t) 1 << 3);
                itty_bit_string_t *s1_second_target = create_bit_string (create_half_populated_word ());
                itty_feed_model_segment_training_summary_t s1_first_summary;

                itty_feed_model_set_decoder (stage1_model,
                                             ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                if (cfg->varied_rotation) {
                        for (size_t model_layer = 0; model_layer < 2; model_layer++)
                                itty_feed_model_set_layer_rotation (stage1_model,
                                                                    model_layer,
                                                                    model_layer + 1);
                }
                if (cfg->sparse_init)
                        assert (itty_feed_model_randomize_masks (stage1_model,
                                                                 0xa100 + case_index,
                                                                 1,
                                                                 8));
                assert (itty_feed_model_train_segment_condense_quota_repair_projection (stage1_model,
                                                                                        s1_first_input,
                                                                                        s1_first_target,
                                                                                        &first_options,
                                                                                        &s1_first_summary));

                {
                        assert (replay_apply_selector_protection_block (stage1_model,
                                                                        s1_first_input,
                                                                        s1_first_target,
                                                                        a_route,
                                                                        NULL,
                                                                        NULL,
                                                                        0,
                                                                        cfg->nodes,
                                                                        owner_margin_reserve,
                                                                        8,
                                                                        &oracle_options,
                                                                        &stage0_summary));
                        assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model, s1_first_input, s1_first_target, a_route, &stage1_committed_a_forced));
                        assert (itty_feed_model_measure_decoder_objective (stage1_model, s1_first_input, s1_first_target, &stage1_committed_a_global));
                        assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model, s1_second_input, s1_second_target, b_route, &stage1_committed_b_forced));
                        assert (itty_feed_model_measure_decoder_objective (stage1_model, s1_second_input, s1_second_target, &stage1_committed_b_global));
                        stage1_committed_a_margin = stage0_summary.margin_after;
                        stage1_committed_b_margin = 0;
                }

                for (size_t step = 0; step < 8; step++) {
                        itty_feed_model_train_stats_t step_stats = { 0 };
                        itty_feed_model_decoder_objective_t a_forced_before = { 0 };
                        itty_feed_model_decoder_objective_t a_global_before = { 0 };
                        itty_feed_model_decoder_objective_t b_forced_before = { 0 };
                        itty_feed_model_decoder_objective_t a_forced_after = { 0 };
                        itty_feed_model_decoder_objective_t a_global_after = { 0 };
                        itty_feed_model_decoder_objective_t b_forced_after = { 0 };
                        itty_feed_model_decoder_objective_t *a_route_before = calloc (cfg->nodes, sizeof *a_route_before);
                        itty_feed_model_decoder_objective_t *a_route_after = calloc (cfg->nodes, sizeof *a_route_after);
                        itty_feed_model_decoder_objective_t *b_route_before = calloc (cfg->nodes, sizeof *b_route_before);
                        itty_feed_model_decoder_objective_t *b_route_after = calloc (cfg->nodes, sizeof *b_route_after);
                        itty_feed_model_segment_node_selection_summary_t a_stage1_selection_before = { 0 };
                        itty_feed_model_segment_node_selection_summary_t a_stage1_selection_after = { 0 };

                        assert (a_route_before && a_route_after && b_route_before && b_route_after);
                        assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model, s1_first_input, s1_first_target, a_route, &a_forced_before));
                        assert (itty_feed_model_measure_decoder_objective (stage1_model, s1_first_input, s1_first_target, &a_global_before));
                        assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model, s1_second_input, s1_second_target, b_route, &b_forced_before));
                        assert (itty_feed_model_measure_segment_node_selection (stage1_model, s1_first_input, s1_first_target, &a_stage1_selection_before));
                        stage1_a_selected_before = a_stage1_selection_before.selected_by_popcount;
                        for (size_t route = 0; route < cfg->nodes; route++) {
                                assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model,
                                                                                            s1_first_input,
                                                                                            s1_first_target,
                                                                                            route,
                                                                                            &a_route_before[route]));
                                assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model,
                                                                                            s1_second_input,
                                                                                            s1_second_target,
                                                                                            route,
                                                                                            &b_route_before[route]));
                        }
                        stage1_a_margin_before = replay_route_margin_for_objectives (a_route_before, cfg->nodes, a_route);
                        stage1_b_margin_before = replay_route_margin_for_objectives (b_route_before, cfg->nodes, b_route);

                        if (!itty_feed_model_train_final_layer_with_suffix_oracle_for_node (stage1_model,
                                                                                             s1_second_input,
                                                                                             s1_second_target,
                                                                                             b_route,
                                                                                             &oracle_options,
                                                                                             &step_stats)) {
                                stage1_reject = "train-fail";
                                break;
                        }
                        if (step_stats.flips == 0) {
                                stage1_reject = "no-flips";
                                break;
                        }

                        assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model, s1_first_input, s1_first_target, a_route, &a_forced_after));
                        assert (itty_feed_model_measure_decoder_objective (stage1_model, s1_first_input, s1_first_target, &a_global_after));
                        assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model, s1_second_input, s1_second_target, b_route, &b_forced_after));
                        assert (itty_feed_model_measure_segment_node_selection (stage1_model, s1_first_input, s1_first_target, &a_stage1_selection_after));
                        stage1_a_selected_after = a_stage1_selection_after.selected_by_popcount;
                        for (size_t route = 0; route < cfg->nodes; route++) {
                                assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model,
                                                                                            s1_first_input,
                                                                                            s1_first_target,
                                                                                            route,
                                                                                            &a_route_after[route]));
                                assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model,
                                                                                            s1_second_input,
                                                                                            s1_second_target,
                                                                                            route,
                                                                                            &b_route_after[route]));
                        }
                        stage1_a_margin_after = replay_route_margin_for_objectives (a_route_after, cfg->nodes, a_route);
                        stage1_b_margin_after = replay_route_margin_for_objectives (b_route_after, cfg->nodes, b_route);
                        stage1_probe_b_after_step = b_forced_after;
                        stage1_probe_a_after_restore = a_forced_after;
                        stage1_probe_a_margin_after_step = stage1_a_margin_after;
                        stage1_probe_b_after_restore = b_forced_after;
                        stage1_probe_a_margin_after_restore = stage1_a_margin_after;

                        if (!replay_route_is_a_safe (&a_forced_after)) {
                                stage1_reject = "a-owner-route";
                                free (a_route_before);
                                free (a_route_after);
                                free (b_route_before);
                                free (b_route_after);
                                break;
                        }
                        if (stage1_a_margin_after < owner_margin_reserve ||
                            a_global_after.selected_distance != 0) {
                                replay_selector_block_summary_t topup_summary = { 0 };

                                assert (replay_apply_selector_protection_block (stage1_model,
                                                                                s1_first_input,
                                                                                s1_first_target,
                                                                                a_route,
                                                                                s1_second_input,
                                                                                s1_second_target,
                                                                                b_route,
                                                                                cfg->nodes,
                                                                                owner_margin_reserve,
                                                                                8,
                                                                                &oracle_options,
                                                                                &topup_summary));
                                stage1_topups += topup_summary.accepted_repairs;

                                assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model, s1_first_input, s1_first_target, a_route, &a_forced_after));
                                assert (itty_feed_model_measure_decoder_objective (stage1_model, s1_first_input, s1_first_target, &a_global_after));
                                assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model, s1_second_input, s1_second_target, b_route, &b_forced_after));
                                assert (itty_feed_model_measure_segment_node_selection (stage1_model, s1_first_input, s1_first_target, &a_stage1_selection_after));
                                stage1_a_selected_after = a_stage1_selection_after.selected_by_popcount;
                                for (size_t route = 0; route < cfg->nodes; route++) {
                                        assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model,
                                                                                                    s1_first_input,
                                                                                                    s1_first_target,
                                                                                                    route,
                                                                                                    &a_route_after[route]));
                                        assert (itty_feed_model_measure_decoder_objective_for_node (stage1_model,
                                                                                                    s1_second_input,
                                                                                                    s1_second_target,
                                                                                                    route,
                                                                                                    &b_route_after[route]));
                                }
                                stage1_a_margin_after = replay_route_margin_for_objectives (a_route_after, cfg->nodes, a_route);
                                stage1_b_margin_after = replay_route_margin_for_objectives (b_route_after, cfg->nodes, b_route);
                                stage1_probe_a_after_restore = a_forced_after;
                                stage1_probe_b_after_restore = b_forced_after;
                                stage1_probe_a_margin_after_restore = stage1_a_margin_after;
                        }

                        if (stage1_a_margin_after < owner_margin_reserve) {
                                stage1_reject = "a-owner-margin";
                                free (a_route_before);
                                free (a_route_after);
                                free (b_route_before);
                                free (b_route_after);
                                break;
                        }
                        if (a_global_after.selected_distance != 0) {
                                stage1_reject = a_stage1_selection_after.selected_by_popcount != a_route ?
                                                "a-global-selector" :
                                                "a-global-same-route";
                                free (a_route_before);
                                free (a_route_after);
                                free (b_route_before);
                                free (b_route_after);
                                break;
                        }
                        if (!(b_forced_after.selected_distance < b_forced_before.selected_distance ||
                              (b_forced_after.selected_distance == b_forced_before.selected_distance &&
                               b_forced_after.false_negative_vote_deficit < b_forced_before.false_negative_vote_deficit))) {
                                if (stage1_probe_b_after_step.selected_distance > b_forced_before.selected_distance ||
                                    (stage1_probe_b_after_step.selected_distance == b_forced_before.selected_distance &&
                                     stage1_probe_b_after_step.false_negative_vote_deficit > b_forced_before.false_negative_vote_deficit))
                                        stage1_reject = "b-route-regressed-step";
                                else
                                        stage1_reject = "b-route-regressed-topup";
                                free (a_route_before);
                                free (a_route_after);
                                free (b_route_before);
                                free (b_route_after);
                                break;
                        }

                        stage1_accepted++;
                        stage1_committed_a_forced = a_forced_after;
                        stage1_committed_a_global = a_global_after;
                        stage1_committed_b_forced = b_forced_after;
                        stage1_committed_b_global = b_forced_after;
                        stage1_committed_a_margin = stage1_a_margin_after;
                        stage1_committed_b_margin = stage1_b_margin_after;
                        free (a_route_before);
                        free (a_route_after);
                        free (b_route_before);
                        free (b_route_after);
                }

                itty_feed_model_t *stage2_model = itty_feed_model_new (2,
                                                                       cfg->nodes,
                                                                       1,
                                                                       1);
                itty_bit_string_list_t *s2_first_input = create_input_with_count (1, 0);
                itty_bit_string_list_t *s2_second_input = create_input_with_count (1, 1);
                itty_bit_string_t *s2_first_target = create_bit_string ((size_t) 1 << 3);
                itty_bit_string_t *s2_second_target = create_bit_string (create_half_populated_word ());
                itty_feed_model_segment_training_summary_t s2_first_summary;

                itty_feed_model_set_decoder (stage2_model,
                                             ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                if (cfg->varied_rotation) {
                        for (size_t model_layer = 0; model_layer < 2; model_layer++)
                                itty_feed_model_set_layer_rotation (stage2_model,
                                                                    model_layer,
                                                                    model_layer + 1);
                }
                if (cfg->sparse_init)
                        assert (itty_feed_model_randomize_masks (stage2_model,
                                                                 0xa100 + case_index,
                                                                 1,
                                                                 8));
                assert (itty_feed_model_train_segment_condense_quota_repair_projection (stage2_model,
                                                                                        s2_first_input,
                                                                                        s2_first_target,
                                                                                        &first_options,
                                                                                        &s2_first_summary));
                if (stage0_summary.accepted_repairs > 0)
                        assert (replay_apply_selector_protection_block (stage2_model,
                                                                        s2_first_input,
                                                                        s2_first_target,
                                                                        a_route,
                                                                        NULL,
                                                                        NULL,
                                                                        0,
                                                                        cfg->nodes,
                                                                        owner_margin_reserve,
                                                                        stage0_summary.accepted_repairs,
                                                                        &oracle_options,
                                                                        NULL));
                for (size_t step = 0; step < stage1_accepted; step++) {
                        itty_feed_model_train_stats_t replayed_step = { 0 };
                        itty_feed_model_decoder_objective_t replay_a_forced = { 0 };
                        itty_feed_model_decoder_objective_t replay_a_global = { 0 };
                        itty_feed_model_decoder_objective_t replay_b_forced = { 0 };
                        itty_feed_model_decoder_objective_t *replay_a_routes = calloc (cfg->nodes, sizeof *replay_a_routes);

                        assert (replay_a_routes);
                        assert (itty_feed_model_train_final_layer_with_suffix_oracle_for_node (stage2_model,
                                                                                               s2_second_input,
                                                                                               s2_second_target,
                                                                                               b_route,
                                                                                               &oracle_options,
                                                                                               &replayed_step));
                        assert (itty_feed_model_measure_decoder_objective_for_node (stage2_model, s2_first_input, s2_first_target, a_route, &replay_a_forced));
                        assert (itty_feed_model_measure_decoder_objective (stage2_model, s2_first_input, s2_first_target, &replay_a_global));
                        assert (itty_feed_model_measure_decoder_objective_for_node (stage2_model, s2_second_input, s2_second_target, b_route, &replay_b_forced));
                        for (size_t route = 0; route < cfg->nodes; route++) {
                                assert (itty_feed_model_measure_decoder_objective_for_node (stage2_model,
                                                                                            s2_first_input,
                                                                                            s2_first_target,
                                                                                            route,
                                                                                            &replay_a_routes[route]));
                        }
                        if (replay_route_margin_for_objectives (replay_a_routes, cfg->nodes, a_route) < owner_margin_reserve ||
                            replay_a_global.selected_distance != 0) {
                                assert (replay_apply_selector_protection_block (stage2_model,
                                                                                s2_first_input,
                                                                                s2_first_target,
                                                                                a_route,
                                                                                s2_second_input,
                                                                                s2_second_target,
                                                                                b_route,
                                                                                cfg->nodes,
                                                                                owner_margin_reserve,
                                                                                8,
                                                                                &oracle_options,
                                                                                NULL));
                        }
                        free (replay_a_routes);
                }

                for (size_t step = 0; step < 8; step++) {
                        itty_feed_model_train_stats_t step_stats = { 0 };
                        itty_feed_model_decoder_objective_t a_forced_after = { 0 };
                        itty_feed_model_decoder_objective_t a_global_after = { 0 };
                        itty_feed_model_decoder_objective_t b_forced_after = { 0 };
                        itty_feed_model_decoder_objective_t b_global_before = { 0 };
                        itty_feed_model_decoder_objective_t b_global_after = { 0 };
                        itty_feed_model_segment_node_selection_summary_t step_selection = { 0 };

                        assert (itty_feed_model_measure_decoder_objective (stage2_model, s2_second_input, s2_second_target, &b_global_before));
                        if (!itty_feed_model_train_final_layer_with_suffix_oracle (stage2_model,
                                                                                   s2_second_input,
                                                                                   s2_second_target,
                                                                                   &oracle_options,
                                                                                   &step_stats)) {
                                break;
                        }
                        if (step_stats.flips == 0) {
                                break;
                        }

                        assert (itty_feed_model_measure_decoder_objective_for_node (stage2_model, s2_first_input, s2_first_target, a_route, &a_forced_after));
                        assert (itty_feed_model_measure_decoder_objective (stage2_model, s2_first_input, s2_first_target, &a_global_after));
                        assert (itty_feed_model_measure_decoder_objective_for_node (stage2_model, s2_second_input, s2_second_target, b_route, &b_forced_after));
                        assert (itty_feed_model_measure_decoder_objective (stage2_model, s2_second_input, s2_second_target, &b_global_after));
                        assert (itty_feed_model_measure_segment_node_selection (stage2_model, s2_second_input, s2_second_target, &step_selection));

                        if (!replay_route_is_a_safe (&a_forced_after)) {
                                break;
                        }
                        if (a_global_after.selected_distance != 0) {
                                break;
                        }
                        if (b_forced_after.selected_distance > final_b_forced.selected_distance &&
                            final_b_forced.selected_distance != 0) {
                                break;
                        }
                        if (step_selection.selected_by_popcount == b_route &&
                            step_selection.popcount_gap >= selector_margin) {
                                stage2_accepted++;
                                break;
                        }
                        break;
                }

                assert (itty_feed_model_measure_decoder_objective_for_node (stage2_model, s2_first_input, s2_first_target, a_route, &final_a_forced));
                assert (itty_feed_model_measure_decoder_objective (stage2_model, s2_first_input, s2_first_target, &final_a_global));
                assert (itty_feed_model_measure_decoder_objective_for_node (stage2_model, s2_second_input, s2_second_target, b_route, &final_b_forced));
                assert (itty_feed_model_measure_decoder_objective (stage2_model, s2_second_input, s2_second_target, &final_b_global));
                assert (itty_feed_model_measure_segment_node_selection (stage2_model, s2_second_input, s2_second_target, &b_selection));

                char const *stage0_kind = "none";
                if (stage0_summary.owner_lifts > 0 && stage0_summary.competitor_suppresses > 0)
                        stage0_kind = "mixed";
                else if (stage0_summary.owner_lifts > 0)
                        stage0_kind = "lift";
                else if (stage0_summary.competitor_suppresses > 0)
                        stage0_kind = "suppress";

                printf ("%s %zu %zu %s %s %td->%td %zu %zu %s %zu->%zu %td->%td %td->%td %zu/%zu %zu/%zu %zu/%zu %zu/%zu committed %zu/%zu %td %zu/%zu %td probe %zu/%zu %zu/%zu->%zu/%zu %td->%td %zu %s %s\n",
                        cfg->label,
                        a_route,
                        b_route,
                        stage0_kind,
                        stage0_summary.accepted_repairs > 0 ? "yes" : "no",
                        stage0_summary.margin_before,
                        stage0_summary.margin_after,
                        stage1_accepted,
                        stage1_topups,
                        stage1_reject,
                        stage1_a_selected_before,
                        stage1_a_selected_after,
                        stage1_a_margin_before,
                        stage1_a_margin_after,
                        stage1_b_margin_before,
                        stage1_b_margin_after,
                        final_a_forced.selected_distance,
                        final_a_forced.false_positive_vote_excess,
                        final_a_global.selected_distance,
                        final_a_global.false_positive_vote_excess,
                        final_b_forced.selected_distance,
                        final_b_forced.false_negative_vote_deficit,
                        final_b_global.selected_distance,
                        final_b_global.false_negative_vote_deficit,
                        stage1_committed_a_global.selected_distance,
                        stage1_committed_a_global.false_positive_vote_excess,
                        stage1_committed_a_margin,
                        stage1_committed_b_forced.selected_distance,
                        stage1_committed_b_forced.false_negative_vote_deficit,
                        stage1_committed_b_margin,
                        stage1_probe_a_after_restore.selected_distance,
                        stage1_probe_a_after_restore.false_positive_vote_excess,
                        stage1_probe_b_after_step.selected_distance,
                        stage1_probe_b_after_step.false_negative_vote_deficit,
                        stage1_probe_b_after_restore.selected_distance,
                        stage1_probe_b_after_restore.false_negative_vote_deficit,
                        stage1_probe_a_margin_after_step,
                        stage1_probe_a_margin_after_restore,
                        stage2_accepted,
                        b_selection.selected_by_popcount == b_route ? "yes" : "no",
                        b_selection.popcount_gap >= selector_margin ? "yes" : "no");

                free (a_routes);
                free (b_routes);
                itty_feed_model_free (seed_model);
                itty_feed_model_free (stage1_model);
                itty_feed_model_free (stage2_model);
                itty_bit_string_free (first_target);
                itty_bit_string_free (second_target);
                itty_bit_string_list_free (first_input);
                itty_bit_string_list_free (second_input);
                itty_bit_string_free (s1_first_target);
                itty_bit_string_free (s1_second_target);
                itty_bit_string_list_free (s1_first_input);
                itty_bit_string_list_free (s1_second_input);
                itty_bit_string_free (s2_first_target);
                itty_bit_string_free (s2_second_target);
                itty_bit_string_list_free (s2_first_input);
                itty_bit_string_list_free (s2_second_input);
        }
}

static void
test_itty_feed_model_segment_replay_route_owned_lane_split_diagnostic (void)
{
        ptrdiff_t owner_margin_reserve = 2;

        printf ("segment replay route-owned lane-split:\n");
        for (size_t selector_bits = 4; selector_bits <= 16; selector_bits *= 2) {
                size_t case_index = 2;
                size_t nodes = 8;
                itty_feed_model_t *seed_model = itty_feed_model_new (2, nodes, 1, 1);
                itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
                itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
                itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
                itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
                itty_feed_model_refreshed_projected_repair_options_t first_options = {
                        .batch_size = 64,
                        .max_rounds = 32,
                        .max_layer_flips_per_batch = 128
                };
                size_t target_bits = itty_bit_string_get_bit_capacity (first_target);
                size_t effective_selector_bits = selector_bits < target_bits ? selector_bits : target_bits;
                itty_feed_model_train_options_t lane_options = {
                        .max_flips = 8,
                        .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                        .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE,
                        .selector_lane_bit_offset = target_bits - effective_selector_bits,
                        .selector_lane_bit_count = effective_selector_bits,
                        .decoder_lane_bit_offset = 0,
                        .decoder_lane_bit_count = target_bits - effective_selector_bits,
                };
                itty_feed_model_segment_training_summary_t first_summary;
                itty_feed_model_segment_node_selection_summary_t a_selection = { 0 };
                itty_feed_model_segment_node_selection_summary_t b_selection = { 0 };
                itty_feed_model_decoder_objective_t *a_routes = calloc (nodes, sizeof *a_routes);
                itty_feed_model_decoder_objective_t *b_routes = calloc (nodes, sizeof *b_routes);
                size_t a_route;
                size_t b_route;

                itty_feed_model_t *stage1_model = NULL;
                itty_bit_string_list_t *s1_first_input = NULL;
                itty_bit_string_list_t *s1_second_input = NULL;
                itty_bit_string_t *s1_first_target = NULL;
                itty_bit_string_t *s1_second_target = NULL;
                itty_feed_model_segment_training_summary_t s1_first_summary;
                replay_selector_block_summary_t stage0_summary = { 0 };
                itty_feed_model_decoder_objective_t initial_a_global = { 0 };
                itty_feed_model_decoder_objective_t initial_b_forced = { 0 };
                ptrdiff_t initial_a_margin = 0;
                ptrdiff_t initial_b_margin = 0;
                size_t stage1_selector_flips = 0;
                size_t stage1_decoder_flips = 0;
                char const *stage1_reject = "none";
                itty_feed_model_decoder_objective_t committed_a_global = { 0 };
                itty_feed_model_decoder_objective_t committed_b_forced = { 0 };
                ptrdiff_t committed_a_margin = 0;
                ptrdiff_t committed_b_margin = 0;
                itty_feed_model_decoder_objective_t probe_b_after_step = { 0 };
                itty_feed_model_decoder_objective_t probe_b_after_restore = { 0 };
                ptrdiff_t probe_a_margin_after_step = 0;
                ptrdiff_t probe_a_margin_after_restore = 0;

                assert (a_routes && b_routes);

                itty_feed_model_set_decoder (seed_model, ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                assert (itty_feed_model_randomize_masks (seed_model, 0xa100 + case_index, 1, 8));
                assert (itty_feed_model_train_segment_condense_quota_repair_projection (seed_model,
                                                                                        first_input,
                                                                                        first_target,
                                                                                        &first_options,
                                                                                        &first_summary));
                assert (replay_measure_segment_node_selection_with_options (seed_model,
                                                                           first_input,
                                                                           first_target,
                                                                           &lane_options,
                                                                           &a_selection));
                assert (replay_measure_segment_node_selection_with_options (seed_model,
                                                                           second_input,
                                                                           second_target,
                                                                           &lane_options,
                                                                           &b_selection));
                for (size_t route = 0; route < nodes; route++) {
                        assert (replay_measure_decoder_objective_for_node_with_options (seed_model,
                                                                                        first_input,
                                                                                        first_target,
                                                                                        route,
                                                                                        &lane_options,
                                                                                        &a_routes[route]));
                        assert (replay_measure_decoder_objective_for_node_with_options (seed_model,
                                                                                        second_input,
                                                                                        second_target,
                                                                                        route,
                                                                                        &lane_options,
                                                                                        &b_routes[route]));
                }
                a_route = replay_choose_a_route (a_routes, nodes, &a_selection);
                b_route = replay_choose_b_route (b_routes, nodes, a_route, &b_selection);
                assert (replay_measure_decoder_objective_with_options (seed_model,
                                                                       first_input,
                                                                       first_target,
                                                                       &lane_options,
                                                                       &initial_a_global));
                initial_b_forced = b_routes[b_route];
                initial_a_margin = replay_route_margin_for_objectives (a_routes, nodes, a_route);
                initial_b_margin = replay_route_margin_for_objectives (b_routes, nodes, b_route);

                stage1_model = itty_feed_model_new (2, nodes, 1, 1);
                s1_first_input = create_input_with_count (1, 0);
                s1_second_input = create_input_with_count (1, 1);
                s1_first_target = create_bit_string ((size_t) 1 << 3);
                s1_second_target = create_bit_string (create_half_populated_word ());

                itty_feed_model_set_decoder (stage1_model, ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                assert (itty_feed_model_randomize_masks (stage1_model, 0xa100 + case_index, 1, 8));
                assert (itty_feed_model_train_segment_condense_quota_repair_projection (stage1_model,
                                                                                        s1_first_input,
                                                                                        s1_first_target,
                                                                                        &first_options,
                                                                                        &s1_first_summary));

                assert (replay_apply_selector_protection_block (stage1_model,
                                                                s1_first_input,
                                                                s1_first_target,
                                                                a_route,
                                                                NULL,
                                                                NULL,
                                                                0,
                                                                nodes,
                                                                owner_margin_reserve,
                                                                8,
                                                                &lane_options,
                                                                &stage0_summary));
                stage1_selector_flips += stage0_summary.accepted_repairs;
                assert (replay_measure_decoder_objective_with_options (stage1_model, s1_first_input, s1_first_target, &lane_options, &committed_a_global));
                assert (replay_measure_decoder_objective_for_node_with_options (stage1_model, s1_second_input, s1_second_target, b_route, &lane_options, &committed_b_forced));
                committed_a_margin = stage0_summary.margin_after;
                committed_b_margin = 0;

                for (size_t step = 0; step < 8; step++) {
                        itty_feed_model_train_stats_t step_stats = { 0 };
                        itty_feed_model_decoder_objective_t a_forced_after = { 0 };
                        itty_feed_model_decoder_objective_t a_global_after = { 0 };
                        itty_feed_model_decoder_objective_t b_forced_before = { 0 };
                        itty_feed_model_decoder_objective_t b_forced_after = { 0 };
                        replay_selector_block_summary_t topup_summary = { 0 };
                        itty_feed_model_decoder_objective_t *a_route_after = calloc (nodes, sizeof *a_route_after);
                        itty_feed_model_decoder_objective_t *b_route_after = calloc (nodes, sizeof *b_route_after);

                        assert (a_route_after && b_route_after);
                        assert (replay_measure_decoder_objective_for_node_with_options (stage1_model, s1_second_input, s1_second_target, b_route, &lane_options, &b_forced_before));

                        if (!itty_feed_model_train_final_layer_with_suffix_oracle_for_node (stage1_model,
                                                                                            s1_second_input,
                                                                                            s1_second_target,
                                                                                            b_route,
                                                                                            &lane_options,
                                                                                            &step_stats)) {
                                stage1_reject = "train-fail";
                                free (a_route_after);
                                free (b_route_after);
                                break;
                        }
                        if (step_stats.flips == 0) {
                                stage1_reject = "no-flips";
                                free (a_route_after);
                                free (b_route_after);
                                break;
                        }
                        stage1_decoder_flips += step_stats.flips;

                        assert (replay_measure_decoder_objective_for_node_with_options (stage1_model, s1_second_input, s1_second_target, b_route, &lane_options, &probe_b_after_step));
                        for (size_t route = 0; route < nodes; route++) {
                                assert (replay_measure_decoder_objective_for_node_with_options (stage1_model,
                                                                                                s1_first_input,
                                                                                                s1_first_target,
                                                                                                route,
                                                                                                &lane_options,
                                                                                                &a_route_after[route]));
                                assert (replay_measure_decoder_objective_for_node_with_options (stage1_model,
                                                                                                s1_second_input,
                                                                                                s1_second_target,
                                                                                                route,
                                                                                                &lane_options,
                                                                                                &b_route_after[route]));
                        }
                        probe_a_margin_after_step = replay_route_margin_for_objectives (a_route_after, nodes, a_route);

                        if (probe_a_margin_after_step < owner_margin_reserve) {
                                assert (replay_apply_selector_protection_block (stage1_model,
                                                                                s1_first_input,
                                                                                s1_first_target,
                                                                                a_route,
                                                                                s1_second_input,
                                                                                s1_second_target,
                                                                                b_route,
                                                                                nodes,
                                                                                owner_margin_reserve,
                                                                                8,
                                                                                &lane_options,
                                                                                &topup_summary));
                                stage1_selector_flips += topup_summary.accepted_repairs;
                        }

                        assert (replay_measure_decoder_objective_for_node_with_options (stage1_model, s1_first_input, s1_first_target, a_route, &lane_options, &a_forced_after));
                        assert (replay_measure_decoder_objective_with_options (stage1_model, s1_first_input, s1_first_target, &lane_options, &a_global_after));
                        assert (replay_measure_decoder_objective_for_node_with_options (stage1_model, s1_second_input, s1_second_target, b_route, &lane_options, &b_forced_after));
                        probe_b_after_restore = b_forced_after;
                        for (size_t route = 0; route < nodes; route++) {
                                assert (replay_measure_decoder_objective_for_node_with_options (stage1_model,
                                                                                                s1_first_input,
                                                                                                s1_first_target,
                                                                                                route,
                                                                                                &lane_options,
                                                                                                &a_route_after[route]));
                        }
                        probe_a_margin_after_restore = replay_route_margin_for_objectives (a_route_after, nodes, a_route);

                        if (a_forced_after.selected_distance != 0 ||
                            a_global_after.selected_distance != 0 ||
                            probe_a_margin_after_restore < owner_margin_reserve) {
                                stage1_reject = "a-owner-margin";
                                free (a_route_after);
                                free (b_route_after);
                                break;
                        }
                        if (!(b_forced_after.selected_distance < b_forced_before.selected_distance ||
                              (b_forced_after.selected_distance == b_forced_before.selected_distance &&
                               b_forced_after.false_negative_vote_deficit < b_forced_before.false_negative_vote_deficit))) {
                                stage1_reject = topup_summary.accepted_repairs > 0 ?
                                                "b-route-regressed-topup" :
                                                "b-route-regressed-step";
                                free (a_route_after);
                                free (b_route_after);
                                break;
                        }

                        committed_a_global = a_global_after;
                        committed_b_forced = b_forced_after;
                        committed_a_margin = probe_a_margin_after_restore;
                        committed_b_margin = replay_route_margin_for_objectives (b_route_after, nodes, b_route);
                        free (a_route_after);
                        free (b_route_after);
                }

                printf ("  sel=%zu dec=%zu routes %zu/%zu  A %td %zu/%zu -> %td %zu/%zu  B %td %zu/%zu -> %td %zu/%zu  probe %zu/%zu->%zu/%zu  A-sm %td->%td  sflip %zu dflip %zu xlane 0 reject %s\n",
                        effective_selector_bits,
                        lane_options.decoder_lane_bit_count,
                        a_route,
                        b_route,
                        initial_a_margin,
                        initial_a_global.selected_distance,
                        initial_a_global.false_positive_vote_excess,
                        committed_a_margin,
                        committed_a_global.selected_distance,
                        committed_a_global.false_positive_vote_excess,
                        initial_b_margin,
                        initial_b_forced.selected_distance,
                        initial_b_forced.false_negative_vote_deficit,
                        committed_b_margin,
                        committed_b_forced.selected_distance,
                        committed_b_forced.false_negative_vote_deficit,
                        probe_b_after_step.selected_distance,
                        probe_b_after_step.false_negative_vote_deficit,
                        probe_b_after_restore.selected_distance,
                        probe_b_after_restore.false_negative_vote_deficit,
                        probe_a_margin_after_step,
                        probe_a_margin_after_restore,
                        stage1_selector_flips,
                        stage1_decoder_flips,
                        stage1_reject);

                free (a_routes);
                free (b_routes);
                itty_feed_model_free (seed_model);
                itty_feed_model_free (stage1_model);
                itty_bit_string_free (first_target);
                itty_bit_string_free (second_target);
                itty_bit_string_list_free (first_input);
                itty_bit_string_list_free (second_input);
                itty_bit_string_free (s1_first_target);
                itty_bit_string_free (s1_second_target);
                itty_bit_string_list_free (s1_first_input);
                itty_bit_string_list_free (s1_second_input);
        }
}

static void
test_itty_feed_model_segment_replay_shadow_selector_diagnostic_with_selector_config (size_t selector_words,
                                                                                     size_t selector_numerator,
                                                                                     size_t selector_denominator)
{
        size_t case_index = 2;
        size_t nodes = 8;
        size_t projection_batch_size = selector_words > 1 ? 32 : 64;
        size_t projection_max_rounds = selector_words > 1 ? 12 : 32;
        size_t projection_max_layer_flips = selector_words > 1 ? 64 : 128;
        size_t oracle_max_flips = selector_words > 1 ? 4 : 8;
        size_t stage1_step_limit = selector_words > 1 ? 4 : 8;
        size_t stage2_block_budget = selector_words > 1 ? 4 : 8;
        size_t stage2_round_limit = selector_words > 1 ? 4 : 8;
        ptrdiff_t owner_margin_reserve = 2;
        itty_feed_model_t *decoder_model = itty_feed_model_new (2, nodes, 1, 1);
        itty_feed_model_t *selector_model = itty_feed_model_new (2, nodes, 1, selector_words);
        itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
        itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
        itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
        itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
        itty_feed_model_refreshed_projected_repair_options_t first_options = {
                .batch_size = projection_batch_size,
                .max_rounds = projection_max_rounds,
                .max_layer_flips_per_batch = projection_max_layer_flips
        };
        itty_feed_model_train_options_t oracle_options = {
                .max_flips = oracle_max_flips,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_segment_training_summary_t first_summary;
        itty_feed_model_segment_training_summary_t selector_first_summary;
        itty_feed_model_segment_node_selection_summary_t a_selection = { 0 };
        itty_feed_model_segment_node_selection_summary_t b_selection = { 0 };
        itty_feed_model_decoder_objective_t *a_routes = calloc (nodes, sizeof *a_routes);
        itty_feed_model_decoder_objective_t *b_routes = calloc (nodes, sizeof *b_routes);
        size_t a_route;
        size_t b_route;
        replay_selector_block_summary_t selector_stage0 = { 0 };
        itty_feed_model_decoder_objective_t initial_a_shadow = { 0 };
        itty_feed_model_decoder_objective_t initial_b_forced = { 0 };
        ptrdiff_t initial_a_margin = 0;
        size_t accepted_steps = 0;
        size_t decoder_flips = 0;
        size_t stage2_strict_accepted = 0;
        size_t stage2_scaffold_accepted = 0;
        size_t stage2_reject_no_candidates[2] = { 0 };
        size_t stage2_reject_a_unsafe[2] = { 0 };
        size_t stage2_reject_compensation_erases_gain[2] = { 0 };
        size_t stage2_reject_insufficient_margin[2] = { 0 };
        ptrdiff_t stage2_best_margin_after_b[2] = { -9999, -9999 };
        ptrdiff_t stage2_best_margin_after_repair[2] = { -9999, -9999 };
        ptrdiff_t stage2_best_a_margin_after_repair[2] = { -9999, -9999 };
        size_t stage2_a_safe_combined_blocks[2] = { 0 };
        size_t stage2_b_selecting_combined_blocks[2] = { 0 };
        size_t stage2_scaffold_rounds = 0;
        size_t stage2_scaffold_total_rows = 0;
        ptrdiff_t stage2_scaffold_round_margins[8] = { 0 };
        ptrdiff_t stage2_scaffold_round_a_margins[8] = { 0 };
        size_t stage2_scaffold_round_b_selected_route[8] = { 0 };
        size_t stage2_scaffold_round_desired_pop[8] = { 0 };
        size_t stage2_scaffold_round_winner_route[8] = { 0 };
        size_t stage2_scaffold_round_winner_pop[8] = { 0 };
        size_t stage2_scaffold_round_gap[8] = { 0 };
        size_t stage2_scaffold_round_need[8] = { 0 };
        size_t stage2_scaffold_round_a_safe_b_effective[8] = { 0 };
        size_t stage2_scaffold_round_a_comp_candidates[8] = { 0 };
        ptrdiff_t stage2_scaffold_round_best_a_safe_margin[8] = { 0 };
        bool stage2_scaffold_round_accepted[8] = { 0 };
        char const *stage2_scaffold_round_reason[8] = { 0 };
        ptrdiff_t initial_b_selector_margin = 0;
        size_t initial_b_selector_winner_route = 0;
        size_t initial_b_selector_winner_popcount = 0;
        size_t initial_b_selector_desired_popcount = 0;
        size_t initial_b_selector_gap = 0;
        size_t initial_b_selector_needed = 0;
        replay_shadow_selector_inventory_t current_inventory = { 0 };
        ptrdiff_t committed_b_selector_margin = 0;
        itty_feed_model_decoder_objective_t committed_a_shadow = { 0 };
        itty_feed_model_decoder_objective_t committed_a_forced = { 0 };
        itty_feed_model_decoder_objective_t committed_b_forced = { 0 };
        ptrdiff_t committed_a_margin = 0;
        itty_feed_model_segment_node_selection_summary_t committed_a_selector = { 0 };
        itty_feed_model_segment_node_selection_summary_t committed_b_selector = { 0 };
        itty_feed_model_decoder_objective_t probe_a_shadow = { 0 };
        itty_feed_model_decoder_objective_t probe_a_forced = { 0 };
        itty_feed_model_decoder_objective_t probe_b_before = { 0 };
        itty_feed_model_decoder_objective_t probe_b_after = { 0 };
        itty_feed_model_segment_node_selection_summary_t probe_a_selector = { 0 };
        ptrdiff_t probe_a_margin = 0;

        assert (a_routes && b_routes);

        itty_feed_model_set_decoder (decoder_model, ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        itty_feed_model_set_decoder (selector_model, ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        assert (itty_feed_model_randomize_masks (decoder_model, 0xa100 + case_index, 1, 8));
        assert (itty_feed_model_randomize_masks (selector_model,
                                                 0xa100 + case_index,
                                                 selector_numerator,
                                                 selector_denominator));

        assert (itty_feed_model_train_segment_condense_quota_repair_projection (decoder_model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &first_summary));
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (selector_model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &selector_first_summary));

        assert (itty_feed_model_measure_segment_node_selection (decoder_model,
                                                                first_input,
                                                                first_target,
                                                                &a_selection));
        assert (itty_feed_model_measure_segment_node_selection (decoder_model,
                                                                second_input,
                                                                second_target,
                                                                &b_selection));
        for (size_t route = 0; route < nodes; route++) {
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            first_input,
                                                                            first_target,
                                                                            route,
                                                                            &a_routes[route]));
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            second_input,
                                                                            second_target,
                                                                            route,
                                                                            &b_routes[route]));
        }
        a_route = replay_choose_a_route (a_routes, nodes, &a_selection);
        b_route = replay_choose_b_route (b_routes, nodes, a_route, &b_selection);

        assert (replay_apply_selector_protection_block (selector_model,
                                                        first_input,
                                                        first_target,
                                                        a_route,
                                                        NULL,
                                                        NULL,
                                                        0,
                                                        nodes,
                                                        owner_margin_reserve,
                                                        8,
                                                        &oracle_options,
                                                        &selector_stage0));
        assert (replay_measure_shadow_selected_decode (decoder_model,
                                                       selector_model,
                                                       first_input,
                                                       first_target,
                                                       &committed_a_selector,
                                                       &initial_a_shadow));
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    second_input,
                                                                    second_target,
                                                                    b_route,
                                                                    &initial_b_forced));
        initial_a_margin = selector_stage0.margin_after;
        assert (replay_measure_shadow_selector_inventory (selector_model,
                                                          first_input,
                                                          first_target,
                                                          a_route,
                                                          second_input,
                                                          second_target,
                                                          b_route,
                                                          nodes,
                                                          &oracle_options,
                                                          &current_inventory));
        assert (itty_feed_model_measure_segment_node_selection (selector_model,
                                                                second_input,
                                                                second_target,
                                                                &committed_b_selector));
        assert (replay_measure_shadow_selector_inventory (selector_model,
                                                          first_input,
                                                          first_target,
                                                          a_route,
                                                          second_input,
                                                          second_target,
                                                          b_route,
                                                          nodes,
                                                          &oracle_options,
                                                          &current_inventory));
        initial_b_selector_winner_route = current_inventory.b_winner_route;
        initial_b_selector_winner_popcount = current_inventory.b_winner_popcount;
        initial_b_selector_desired_popcount = current_inventory.b_desired_popcount;
        initial_b_selector_gap = current_inventory.b_gap;
        initial_b_selector_needed = current_inventory.b_needed;
        initial_b_selector_margin = current_inventory.b_margin;

        committed_a_shadow = initial_a_shadow;
        committed_a_margin = initial_a_margin;
        committed_b_forced = initial_b_forced;
        committed_b_selector_margin = initial_b_selector_margin;
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    first_input,
                                                                    first_target,
                                                                    a_route,
                                                                    &committed_a_forced));

        for (size_t step = 0; step < stage1_step_limit; step++) {
                itty_feed_model_train_stats_t step_stats = { 0 };
                itty_feed_model_decoder_objective_t selector_routes[nodes];

                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            second_input,
                                                                            second_target,
                                                                            b_route,
                                                                            &probe_b_before));

                if (!itty_feed_model_train_final_layer_with_suffix_oracle_for_node (decoder_model,
                                                                                    second_input,
                                                                                    second_target,
                                                                                    b_route,
                                                                                    &oracle_options,
                                                                                    &step_stats)) {
                        break;
                }
                if (step_stats.flips == 0) {
                        break;
                }
                decoder_flips += step_stats.flips;

                assert (replay_measure_shadow_selected_decode (decoder_model,
                                                               selector_model,
                                                               first_input,
                                                               first_target,
                                                               &probe_a_selector,
                                                               &probe_a_shadow));
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            first_input,
                                                                            first_target,
                                                                            a_route,
                                                                            &probe_a_forced));
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            second_input,
                                                                            second_target,
                                                                            b_route,
                                                                            &probe_b_after));
                for (size_t route = 0; route < nodes; route++) {
                        assert (itty_feed_model_measure_decoder_objective_for_node (selector_model,
                                                                                    first_input,
                                                                                    first_target,
                                                                                    route,
                                                                                    &selector_routes[route]));
                }
                probe_a_margin = replay_route_margin_for_objectives (selector_routes, nodes, a_route);

                if (!replay_route_is_a_safe (&probe_a_forced)) {
                        break;
                }
                if (probe_a_shadow.selected_distance != 0) {
                        break;
                }
                if (probe_a_margin < owner_margin_reserve) {
                        break;
                }
                if (!(probe_b_after.selected_distance < probe_b_before.selected_distance ||
                      (probe_b_after.selected_distance == probe_b_before.selected_distance &&
                       probe_b_after.false_negative_vote_deficit < probe_b_before.false_negative_vote_deficit))) {
                        break;
                }

                accepted_steps++;
                committed_a_selector = probe_a_selector;
                committed_a_shadow = probe_a_shadow;
                committed_a_forced = probe_a_forced;
                committed_b_forced = probe_b_after;
                committed_a_margin = probe_a_margin;
        }

        {
                itty_feed_model_layer_state_snapshot_t *stage2_base_snapshot =
                        itty_feed_model_snapshot_final_layer_state (selector_model);
                replay_selector_joint_search_t *strict_search =
                        calloc (1, sizeof *strict_search);

                assert (stage2_base_snapshot);
                assert (strict_search);

                assert (replay_search_selector_joint_blocks (decoder_model,
                                                            selector_model,
                                                            first_input,
                                                            first_target,
                                                            a_route,
                                                            second_input,
                                                            second_target,
                                                            b_route,
                                                            nodes,
                                                            owner_margin_reserve,
                                                            committed_b_selector_margin,
                                                            stage2_block_budget,
                                                            true,
                                                            &oracle_options,
                                                            strict_search));

                stage2_strict_accepted = strict_search->accepted_blocks;
                stage2_reject_no_candidates[0] = !strict_search->found_any_candidate;
                stage2_reject_a_unsafe[0] = strict_search->found_any_candidate && strict_search->a_safe_blocks == 0;
                stage2_reject_compensation_erases_gain[0] = strict_search->reject_comp_erases;
                stage2_reject_insufficient_margin[0] = strict_search->found_any_candidate && !strict_search->found_solution;
                stage2_best_margin_after_b[0] = strict_search->best_margin_after_b;
                stage2_best_margin_after_repair[0] = strict_search->best_margin_after_repair;
                stage2_best_a_margin_after_repair[0] = strict_search->best_a_margin_after_repair;
                stage2_a_safe_combined_blocks[0] = strict_search->a_safe_blocks;
                stage2_b_selecting_combined_blocks[0] = strict_search->b_selecting_blocks;

                if (strict_search->found_solution) {
                        itty_feed_model_restore_final_layer_state_snapshot (selector_model, stage2_base_snapshot);
                        stage2_base_snapshot = NULL;
                        committed_a_shadow = strict_search->best_state.a_shadow;
                        committed_a_forced = strict_search->best_state.a_forced;
                        committed_b_forced = strict_search->best_state.b_forced;
                        committed_b_selector = strict_search->best_state.b_selector;
                        committed_b_selector_margin = strict_search->best_state.b_margin;
                        committed_a_margin = strict_search->best_state.a_margin;
                } else {
                        itty_feed_model_restore_final_layer_state_snapshot (selector_model, stage2_base_snapshot);
                        stage2_base_snapshot = NULL;
                }
                if (stage2_base_snapshot)
                        itty_feed_model_free_final_layer_state_snapshot (selector_model, stage2_base_snapshot);
                free (strict_search);

                for (size_t round = 0; round < stage2_round_limit; round++) {
                        itty_feed_model_layer_state_snapshot_t *round_snapshot =
                                itty_feed_model_snapshot_final_layer_state (selector_model);
                        replay_selector_joint_search_t *scaffold_search =
                                calloc (1, sizeof *scaffold_search);

                        assert (round_snapshot);
                        assert (scaffold_search);
                        assert (replay_search_selector_joint_blocks (decoder_model,
                                                                    selector_model,
                                                                    first_input,
                                                                    first_target,
                                                                    a_route,
                                                                    second_input,
                                                                    second_target,
                                                                    b_route,
                                                                    nodes,
                                                                    owner_margin_reserve,
                                                                    committed_b_selector_margin,
                                                                    stage2_block_budget,
                                                                    false,
                                                                    &oracle_options,
                                                                    scaffold_search));

                        stage2_reject_no_candidates[1] += !scaffold_search->found_any_candidate;
                        stage2_reject_a_unsafe[1] += scaffold_search->found_any_candidate && scaffold_search->a_safe_blocks == 0;
                        stage2_reject_compensation_erases_gain[1] += scaffold_search->reject_comp_erases;
                        stage2_reject_insufficient_margin[1] += scaffold_search->found_any_candidate && !scaffold_search->found_solution;
                        if (scaffold_search->best_margin_after_b > stage2_best_margin_after_b[1])
                                stage2_best_margin_after_b[1] = scaffold_search->best_margin_after_b;
                        if (scaffold_search->best_margin_after_repair > stage2_best_margin_after_repair[1])
                                stage2_best_margin_after_repair[1] = scaffold_search->best_margin_after_repair;
                        if (scaffold_search->best_a_margin_after_repair > stage2_best_a_margin_after_repair[1])
                                stage2_best_a_margin_after_repair[1] = scaffold_search->best_a_margin_after_repair;
                        stage2_a_safe_combined_blocks[1] += scaffold_search->a_safe_blocks;
                        stage2_b_selecting_combined_blocks[1] += scaffold_search->b_selecting_blocks;

                        if (!scaffold_search->found_solution) {
                                if (stage2_scaffold_total_rows < 8) {
                                        char const *reason = "candidate-block-search-budget-exhausted";
                                        size_t b_effective =
                                                current_inventory.b_selector_inventory.owner_effective_candidates +
                                                current_inventory.b_selector_inventory.competitor_effective_candidates;
                                        size_t a_compensation =
                                                current_inventory.a_compensation_inventory.owner_candidate_bits +
                                                current_inventory.a_compensation_inventory.competitor_candidate_bits;

                                        if (b_effective == 0)
                                                reason = "no-b-effective-selector-moves-remain";
                                        else if (a_compensation == 0)
                                                reason = "a-compensation-unavailable";
                                        else if (scaffold_search->reject_comp_erases > 0)
                                                reason = "a-compensation-cancels-all-b-movement";
                                        else if (scaffold_search->best_margin_after_repair == 0 &&
                                                 committed_b_selector.selected_by_popcount != b_route)
                                                reason = "b-margin-zero-but-tie-still-selects-route2";

                                        stage2_scaffold_round_a_margins[stage2_scaffold_total_rows] = committed_a_margin;
                                        stage2_scaffold_round_b_selected_route[stage2_scaffold_total_rows] = committed_b_selector.selected_by_popcount;
                                        stage2_scaffold_round_margins[stage2_scaffold_total_rows] = committed_b_selector_margin;
                                        stage2_scaffold_round_desired_pop[stage2_scaffold_total_rows] = current_inventory.b_desired_popcount;
                                        stage2_scaffold_round_winner_route[stage2_scaffold_total_rows] = current_inventory.b_winner_route;
                                        stage2_scaffold_round_winner_pop[stage2_scaffold_total_rows] = current_inventory.b_winner_popcount;
                                        stage2_scaffold_round_gap[stage2_scaffold_total_rows] = current_inventory.b_gap;
                                        stage2_scaffold_round_need[stage2_scaffold_total_rows] = current_inventory.b_needed;
                                        stage2_scaffold_round_a_safe_b_effective[stage2_scaffold_total_rows] =
                                                current_inventory.b_selector_inventory.owner_safe_candidates +
                                                current_inventory.b_selector_inventory.competitor_safe_candidates;
                                        stage2_scaffold_round_a_comp_candidates[stage2_scaffold_total_rows] = a_compensation;
                                        stage2_scaffold_round_best_a_safe_margin[stage2_scaffold_total_rows] =
                                                scaffold_search->best_margin_after_repair;
                                        stage2_scaffold_round_accepted[stage2_scaffold_total_rows] = false;
                                        stage2_scaffold_round_reason[stage2_scaffold_total_rows] = reason;
                                        stage2_scaffold_total_rows++;
                                }
                                itty_feed_model_restore_final_layer_state_snapshot (selector_model, round_snapshot);
                                free (scaffold_search);
                                break;
                        }

                        stage2_scaffold_accepted++;
                        committed_a_shadow = scaffold_search->best_state.a_shadow;
                        committed_a_forced = scaffold_search->best_state.a_forced;
                        committed_b_forced = scaffold_search->best_state.b_forced;
                        committed_b_selector = scaffold_search->best_state.b_selector;
                        committed_b_selector_margin = scaffold_search->best_state.b_margin;
                        committed_a_margin = scaffold_search->best_state.a_margin;

                        assert (replay_measure_shadow_selector_inventory (selector_model,
                                                                          first_input,
                                                                          first_target,
                                                                          a_route,
                                                                          second_input,
                                                                          second_target,
                                                                          b_route,
                                                                          nodes,
                                                                          &oracle_options,
                                                                          &current_inventory));
                        stage2_scaffold_round_a_margins[stage2_scaffold_total_rows] = committed_a_margin;
                        stage2_scaffold_round_b_selected_route[stage2_scaffold_total_rows] = committed_b_selector.selected_by_popcount;
                        stage2_scaffold_round_margins[stage2_scaffold_total_rows] = committed_b_selector_margin;
                        stage2_scaffold_round_desired_pop[stage2_scaffold_total_rows] = current_inventory.b_desired_popcount;
                        stage2_scaffold_round_winner_route[stage2_scaffold_total_rows] = current_inventory.b_winner_route;
                        stage2_scaffold_round_winner_pop[stage2_scaffold_total_rows] = current_inventory.b_winner_popcount;
                        stage2_scaffold_round_gap[stage2_scaffold_total_rows] = current_inventory.b_gap;
                        stage2_scaffold_round_need[stage2_scaffold_total_rows] = current_inventory.b_needed;
                        stage2_scaffold_round_a_safe_b_effective[stage2_scaffold_total_rows] =
                                current_inventory.b_selector_inventory.owner_safe_candidates +
                                current_inventory.b_selector_inventory.competitor_safe_candidates;
                        stage2_scaffold_round_a_comp_candidates[stage2_scaffold_total_rows] =
                                current_inventory.a_compensation_inventory.owner_candidate_bits +
                                current_inventory.a_compensation_inventory.competitor_candidate_bits;
                        stage2_scaffold_round_best_a_safe_margin[stage2_scaffold_total_rows] =
                                scaffold_search->best_margin_after_repair;
                        stage2_scaffold_round_accepted[stage2_scaffold_total_rows] = true;
                        stage2_scaffold_round_reason[stage2_scaffold_total_rows] = "accepted-scaffold";
                        stage2_scaffold_total_rows++;
                        stage2_scaffold_rounds++;

                        if (committed_b_selector.selected_by_popcount == b_route &&
                            committed_b_selector.popcount_gap >= 2) {
                                itty_feed_model_free_final_layer_state_snapshot (selector_model, round_snapshot);
                                free (scaffold_search);
                                break;
                        }

                        itty_feed_model_free_final_layer_state_snapshot (selector_model, round_snapshot);
                        free (scaffold_search);
                }
        }

        assert (itty_feed_model_measure_segment_node_selection (selector_model,
                                                                second_input,
                                                                second_target,
                                                                &committed_b_selector));

        printf ("---shadow-selector-begin\n");
        printf ("---selector_words=%zu\n", selector_words);
        printf ("---selector_density=%zu/%zu\n", selector_numerator, selector_denominator);
        printf ("---a_route=%zu\n", a_route);
        printf ("---b_route=%zu\n", b_route);
        printf ("---b_desired_pop=%zu\n", initial_b_selector_desired_popcount);
        printf ("---b_winner_route=%zu\n", initial_b_selector_winner_route);
        printf ("---b_winner_pop=%zu\n", initial_b_selector_winner_popcount);
        printf ("---b_gap=%zu\n", initial_b_selector_gap);
        printf ("---b_need=%zu\n", initial_b_selector_needed);
        printf ("---inv_lift=%zu\n", current_inventory.b_selector_inventory.owner_candidate_bits);
        printf ("---inv_suppress=%zu\n", current_inventory.b_selector_inventory.competitor_candidate_bits);
        printf ("---inv_mixed=%zu\n", current_inventory.b_selector_inventory.mixed_candidate_pairs);
        printf ("---a_safe_lift=%zu\n", current_inventory.b_selector_inventory.owner_safe_candidates);
        printf ("---a_safe_suppress=%zu\n", current_inventory.b_selector_inventory.competitor_safe_candidates);
        printf ("---b_eff_lift=%zu\n", current_inventory.b_selector_inventory.owner_effective_candidates);
        printf ("---b_eff_suppress=%zu\n", current_inventory.b_selector_inventory.competitor_effective_candidates);
        printf ("---a_comp_lift=%zu\n", current_inventory.a_compensation_inventory.owner_candidate_bits);
        printf ("---a_comp_suppress=%zu\n", current_inventory.a_compensation_inventory.competitor_candidate_bits);
        printf ("---a_comp_eff_lift=%zu\n", current_inventory.a_compensation_inventory.owner_effective_candidates);
        printf ("---a_comp_eff_suppress=%zu\n", current_inventory.a_compensation_inventory.competitor_effective_candidates);
        printf ("---a_sm_before=%td\n", initial_a_margin);
        printf ("---a_sm_after=%td\n", committed_a_margin);
        printf ("---a_shadow_dist=%zu\n", committed_a_shadow.selected_distance);
        printf ("---a_shadow_fp=%zu\n", committed_a_shadow.false_positive_vote_excess);
        printf ("---a_forced_dist=%zu\n", committed_a_forced.selected_distance);
        printf ("---a_forced_fp=%zu\n", committed_a_forced.false_positive_vote_excess);
        printf ("---b_forced_before_dist=%zu\n", initial_b_forced.selected_distance);
        printf ("---b_forced_before_def=%zu\n", initial_b_forced.false_negative_vote_deficit);
        printf ("---b_forced_after_dist=%zu\n", committed_b_forced.selected_distance);
        printf ("---b_forced_after_def=%zu\n", committed_b_forced.false_negative_vote_deficit);
        printf ("---b_sm_before=%td\n", initial_b_selector_margin);
        printf ("---b_sm_after=%td\n", committed_b_selector_margin);
        printf ("---b_sel=%zu\n", committed_b_selector.selected_by_popcount);
        printf ("---b_sel_gap=%zu\n", committed_b_selector.popcount_gap);
        printf ("---stage2_strict_acc=%zu\n", stage2_strict_accepted);
        printf ("---stage2_strict_rej_no_candidates=%zu\n", stage2_reject_no_candidates[0]);
        printf ("---stage2_strict_rej_a_unsafe=%zu\n", stage2_reject_a_unsafe[0]);
        printf ("---stage2_strict_rej_comp_erases=%zu\n", stage2_reject_compensation_erases_gain[0]);
        printf ("---stage2_strict_rej_insufficient_margin=%zu\n", stage2_reject_insufficient_margin[0]);
        printf ("---stage2_strict_best_margin_after_b=%td\n", stage2_best_margin_after_b[0]);
        printf ("---stage2_strict_best_margin_after_repair=%td\n", stage2_best_margin_after_repair[0]);
        printf ("---stage2_strict_best_a_margin_after_repair=%td\n", stage2_best_a_margin_after_repair[0]);
        printf ("---stage2_strict_a_safe_blocks=%zu\n", stage2_a_safe_combined_blocks[0]);
        printf ("---stage2_strict_b_selecting_blocks=%zu\n", stage2_b_selecting_combined_blocks[0]);
        printf ("---stage2_scaffold_acc=%zu\n", stage2_scaffold_accepted);
        printf ("---stage2_scaffold_rej_no_candidates=%zu\n", stage2_reject_no_candidates[1]);
        printf ("---stage2_scaffold_rej_a_unsafe=%zu\n", stage2_reject_a_unsafe[1]);
        printf ("---stage2_scaffold_rej_comp_erases=%zu\n", stage2_reject_compensation_erases_gain[1]);
        printf ("---stage2_scaffold_rej_insufficient_margin=%zu\n", stage2_reject_insufficient_margin[1]);
        printf ("---stage2_scaffold_best_margin_after_b=%td\n", stage2_best_margin_after_b[1]);
        printf ("---stage2_scaffold_best_margin_after_repair=%td\n", stage2_best_margin_after_repair[1]);
        printf ("---stage2_scaffold_best_a_margin_after_repair=%td\n", stage2_best_a_margin_after_repair[1]);
        printf ("---stage2_scaffold_a_safe_blocks=%zu\n", stage2_a_safe_combined_blocks[1]);
        printf ("---stage2_scaffold_b_selecting_blocks=%zu\n", stage2_b_selecting_combined_blocks[1]);
        printf ("---stage2_scaffold_rounds=%zu\n", stage2_scaffold_rounds);
        printf ("---stage2_scaffold_total_rows=%zu\n", stage2_scaffold_total_rows);
        for (size_t round = 0; round < stage2_scaffold_total_rows; round++) {
                printf ("---stage2_scaffold_round_%zu_a_margin=%td\n", round, stage2_scaffold_round_a_margins[round]);
                printf ("---stage2_scaffold_round_%zu_b_selected_route=%zu\n", round, stage2_scaffold_round_b_selected_route[round]);
                printf ("---stage2_scaffold_round_%zu_b_sm=%td\n", round, stage2_scaffold_round_margins[round]);
                printf ("---stage2_scaffold_round_%zu_b_desired_pop=%zu\n", round, stage2_scaffold_round_desired_pop[round]);
                printf ("---stage2_scaffold_round_%zu_b_winner_pop=%zu\n", round, stage2_scaffold_round_winner_pop[round]);
                printf ("---stage2_scaffold_round_%zu_b_gap=%zu\n", round, stage2_scaffold_round_gap[round]);
                printf ("---stage2_scaffold_round_%zu_b_need=%zu\n", round, stage2_scaffold_round_need[round]);
                printf ("---stage2_scaffold_round_%zu_best_a_safe_margin=%td\n", round, stage2_scaffold_round_best_a_safe_margin[round]);
                printf ("---stage2_scaffold_round_%zu_a_safe_b_effective=%zu\n", round, stage2_scaffold_round_a_safe_b_effective[round]);
                printf ("---stage2_scaffold_round_%zu_a_comp_candidates=%zu\n", round, stage2_scaffold_round_a_comp_candidates[round]);
                printf ("---stage2_scaffold_round_%zu_accepted=%s\n", round, stage2_scaffold_round_accepted[round] ? "yes" : "no");
                printf ("---stage2_scaffold_round_%zu_reject_reason=%s\n", round, stage2_scaffold_round_reason[round]);
        }
        printf ("---shadow-selector-end\n");

        free (a_routes);
        free (b_routes);
        itty_feed_model_free (decoder_model);
        itty_feed_model_free (selector_model);
        itty_bit_string_free (first_target);
        itty_bit_string_free (second_target);
        itty_bit_string_list_free (first_input);
        itty_bit_string_list_free (second_input);
}

static void
test_itty_feed_model_segment_replay_shadow_selector_diagnostic (void)
{
        test_itty_feed_model_segment_replay_shadow_selector_diagnostic_with_selector_config (1, 1, 8);
}

static void
test_itty_feed_model_segment_replay_shadow_selector_density_sweep (void)
{
        size_t configs[][2] = {
                { 1, 16 },
                { 1, 8 },
                { 1, 4 },
        };

        for (size_t index = 0; index < sizeof (configs) / sizeof (configs[0]); index++)
                test_itty_feed_model_segment_replay_shadow_selector_diagnostic_with_selector_config (1,
                                                                                                      configs[index][0],
                                                                                                      configs[index][1]);
}

static void
test_itty_feed_model_segment_replay_shadow_selector_frozen_decoder_sweep_with_config (size_t selector_words_filter,
                                                                                      size_t selector_numerator_filter,
                                                                                      size_t selector_denominator_filter,
                                                                                      size_t selector_seed_filter,
                                                                                      bool   single_seed,
                                                                                      bool   single_config)
{
        size_t case_index = 2;
        size_t nodes = 8;
        size_t projection_batch_size = 64;
        size_t projection_max_rounds = 32;
        size_t projection_max_layer_flips = 128;
        size_t oracle_max_flips = 8;
        size_t stage1_step_limit = 8;
        ptrdiff_t owner_margin_reserve = 2;
        size_t selector_words_list[] = { 1, 2, 4 };
        size_t selector_density_list[][2] = {
                { 1, 16 },
                { 1, 8 },
                { 1, 4 },
        };
        itty_feed_model_t *decoder_model = itty_feed_model_new (2, nodes, 1, 1);
        itty_feed_model_t *baseline_selector_model = itty_feed_model_new (2, nodes, 1, 1);
        itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
        itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
        itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
        itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
        itty_feed_model_refreshed_projected_repair_options_t first_options = {
                .batch_size = projection_batch_size,
                .max_rounds = projection_max_rounds,
                .max_layer_flips_per_batch = projection_max_layer_flips
        };
        itty_feed_model_train_options_t oracle_options = {
                .max_flips = oracle_max_flips,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_segment_training_summary_t first_summary;
        itty_feed_model_segment_training_summary_t selector_first_summary;
        itty_feed_model_segment_node_selection_summary_t a_selection = { 0 };
        itty_feed_model_segment_node_selection_summary_t b_selection = { 0 };
        itty_feed_model_decoder_objective_t *a_routes = calloc (nodes, sizeof *a_routes);
        itty_feed_model_decoder_objective_t *b_routes = calloc (nodes, sizeof *b_routes);
        size_t a_route;
        size_t b_route;
        replay_selector_block_summary_t selector_stage0 = { 0 };
        itty_feed_model_decoder_objective_t committed_a_shadow = { 0 };
        itty_feed_model_decoder_objective_t committed_a_forced = { 0 };
        itty_feed_model_decoder_objective_t committed_b_forced = { 0 };
        ptrdiff_t committed_a_margin = 0;

        assert (decoder_model && baseline_selector_model);
        assert (a_routes && b_routes);

        itty_feed_model_set_decoder (decoder_model, ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        itty_feed_model_set_decoder (baseline_selector_model, ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        assert (itty_feed_model_randomize_masks (decoder_model, 0xa100 + case_index, 1, 8));
        assert (itty_feed_model_randomize_masks (baseline_selector_model, 0xa100 + case_index, 1, 8));
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (decoder_model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &first_summary));
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (baseline_selector_model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &selector_first_summary));
        assert (itty_feed_model_measure_segment_node_selection (decoder_model,
                                                                first_input,
                                                                first_target,
                                                                &a_selection));
        assert (itty_feed_model_measure_segment_node_selection (decoder_model,
                                                                second_input,
                                                                second_target,
                                                                &b_selection));
        for (size_t route = 0; route < nodes; route++) {
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            first_input,
                                                                            first_target,
                                                                            route,
                                                                            &a_routes[route]));
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            second_input,
                                                                            second_target,
                                                                            route,
                                                                            &b_routes[route]));
        }
        a_route = replay_choose_a_route (a_routes, nodes, &a_selection);
        b_route = replay_choose_b_route (b_routes, nodes, a_route, &b_selection);

        assert (replay_apply_selector_protection_block (baseline_selector_model,
                                                        first_input,
                                                        first_target,
                                                        a_route,
                                                        NULL,
                                                        NULL,
                                                        0,
                                                        nodes,
                                                        owner_margin_reserve,
                                                        8,
                                                        &oracle_options,
                                                        &selector_stage0));
        assert (replay_measure_shadow_selected_decode (decoder_model,
                                                       baseline_selector_model,
                                                       first_input,
                                                       first_target,
                                                       &a_selection,
                                                       &committed_a_shadow));
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    first_input,
                                                                    first_target,
                                                                    a_route,
                                                                    &committed_a_forced));
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    second_input,
                                                                    second_target,
                                                                    b_route,
                                                                    &committed_b_forced));
        committed_a_margin = selector_stage0.margin_after;

        for (size_t step = 0; step < stage1_step_limit; step++) {
                itty_feed_model_train_stats_t step_stats = { 0 };
                itty_feed_model_decoder_objective_t probe_a_shadow = { 0 };
                itty_feed_model_decoder_objective_t probe_a_forced = { 0 };
                itty_feed_model_decoder_objective_t probe_b_before = { 0 };
                itty_feed_model_decoder_objective_t probe_b_after = { 0 };
                itty_feed_model_segment_node_selection_summary_t probe_a_selector = { 0 };
                itty_feed_model_decoder_objective_t selector_routes[nodes];
                ptrdiff_t probe_a_margin = 0;

                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            second_input,
                                                                            second_target,
                                                                            b_route,
                                                                            &probe_b_before));
                if (!itty_feed_model_train_final_layer_with_suffix_oracle_for_node (decoder_model,
                                                                                    second_input,
                                                                                    second_target,
                                                                                    b_route,
                                                                                    &oracle_options,
                                                                                    &step_stats))
                        break;
                if (step_stats.flips == 0)
                        break;

                assert (replay_measure_shadow_selected_decode (decoder_model,
                                                               baseline_selector_model,
                                                               first_input,
                                                               first_target,
                                                               &probe_a_selector,
                                                               &probe_a_shadow));
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            first_input,
                                                                            first_target,
                                                                            a_route,
                                                                            &probe_a_forced));
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            second_input,
                                                                            second_target,
                                                                            b_route,
                                                                            &probe_b_after));
                for (size_t route = 0; route < nodes; route++)
                        assert (itty_feed_model_measure_decoder_objective_for_node (baseline_selector_model,
                                                                                    first_input,
                                                                                    first_target,
                                                                                    route,
                                                                                    &selector_routes[route]));
                probe_a_margin = replay_route_margin_for_objectives (selector_routes, nodes, a_route);

                if (!replay_route_is_a_safe (&probe_a_forced) ||
                    probe_a_shadow.selected_distance != 0 ||
                    probe_a_margin < owner_margin_reserve ||
                    !(probe_b_after.selected_distance < probe_b_before.selected_distance ||
                      (probe_b_after.selected_distance == probe_b_before.selected_distance &&
                       probe_b_after.false_negative_vote_deficit < probe_b_before.false_negative_vote_deficit)))
                        break;

                committed_a_shadow = probe_a_shadow;
                committed_a_forced = probe_a_forced;
                committed_b_forced = probe_b_after;
                committed_a_margin = probe_a_margin;
        }

        printf ("---frozen-shadow-selector-begin\n");
        printf ("---frozen_a_route=%zu\n", a_route);
        printf ("---frozen_b_route=%zu\n", b_route);
        printf ("---frozen_a_margin=%td\n", committed_a_margin);
        printf ("---frozen_a_shadow_dist=%zu\n", committed_a_shadow.selected_distance);
        printf ("---frozen_b_forced_dist=%zu\n", committed_b_forced.selected_distance);
        printf ("---frozen_b_forced_def=%zu\n", committed_b_forced.false_negative_vote_deficit);

        for (size_t word_index = 0; word_index < sizeof (selector_words_list) / sizeof (selector_words_list[0]); word_index++) {
                for (size_t density_index = 0; density_index < sizeof (selector_density_list) / sizeof (selector_density_list[0]); density_index++) {
                        size_t selector_words = selector_words_list[word_index];
                        size_t selector_numerator = selector_density_list[density_index][0];
                        size_t selector_denominator = selector_density_list[density_index][1];
                        size_t stage2_block_budget = selector_words > 1 ? 4 : 8;
                        size_t stage2_round_limit = selector_words > 1 ? 4 : 8;
                        if (single_config &&
                            (selector_words != selector_words_filter ||
                             selector_numerator != selector_numerator_filter ||
                             selector_denominator != selector_denominator_filter))
                                continue;

                        itty_feed_model_t *selector_model = itty_feed_model_new (2, nodes, 1, selector_words);
                        replay_selector_block_summary_t selector_stage0_summary = { 0 };
                        replay_shadow_selector_inventory_t inventory = { 0 };
                        itty_feed_model_segment_node_selection_summary_t committed_b_selector = { 0 };
                        ptrdiff_t committed_b_selector_margin = 0;
                        size_t stage2_scaffold_accepted = 0;
                        size_t stage2_b_selecting_combined_blocks[2] = { 0 };
                        ptrdiff_t stage2_best_margin_after_repair[2] = { -9999, -9999 };

                        assert (selector_model);
                        itty_feed_model_set_decoder (selector_model, ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
                        size_t selector_seed = single_seed
                                ? selector_seed_filter
                                : (0xa100 + case_index + selector_words * 17 + density_index);
                        assert (itty_feed_model_randomize_masks (selector_model,
                                                                 selector_seed,
                                                                 selector_numerator,
                                                                 selector_denominator));
                        assert (itty_feed_model_train_segment_condense_quota_repair_projection (selector_model,
                                                                                                first_input,
                                                                                                first_target,
                                                                                                &first_options,
                                                                                                &selector_first_summary));
                        assert (replay_apply_selector_protection_block (selector_model,
                                                                        first_input,
                                                                        first_target,
                                                                        a_route,
                                                                        NULL,
                                                                        NULL,
                                                                        0,
                                                                        nodes,
                                                                        owner_margin_reserve,
                                                                        8,
                                                                        &oracle_options,
                                                                        &selector_stage0_summary));
                        assert (replay_measure_shadow_selector_inventory (selector_model,
                                                                          first_input,
                                                                          first_target,
                                                                          a_route,
                                                                          second_input,
                                                                          second_target,
                                                                          b_route,
                                                                          nodes,
                                                                          &oracle_options,
                                                                          &inventory));
                        committed_b_selector_margin = inventory.b_margin;
                        assert (itty_feed_model_measure_segment_node_selection (selector_model,
                                                                                second_input,
                                                                                second_target,
                                                                                &committed_b_selector));

                        {
                                itty_feed_model_layer_state_snapshot_t *stage2_base_snapshot =
                                        itty_feed_model_snapshot_final_layer_state (selector_model);
                                replay_selector_joint_search_t *strict_search =
                                        calloc (1, sizeof *strict_search);

                                assert (stage2_base_snapshot);
                                assert (strict_search);
                                assert (replay_search_selector_joint_blocks (decoder_model,
                                                                            selector_model,
                                                                            first_input,
                                                                            first_target,
                                                                            a_route,
                                                                            second_input,
                                                                            second_target,
                                                                            b_route,
                                                                            nodes,
                                                                            owner_margin_reserve,
                                                                            committed_b_selector_margin,
                                                                            stage2_block_budget,
                                                                            true,
                                                                            &oracle_options,
                                                                            strict_search));
                                stage2_b_selecting_combined_blocks[0] = strict_search->b_selecting_blocks;
                                stage2_best_margin_after_repair[0] = strict_search->best_margin_after_repair;
                                itty_feed_model_restore_final_layer_state_snapshot (selector_model, stage2_base_snapshot);
                                free (strict_search);

                                for (size_t round = 0; round < stage2_round_limit; round++) {
                                        itty_feed_model_layer_state_snapshot_t *round_snapshot =
                                                itty_feed_model_snapshot_final_layer_state (selector_model);
                                        replay_selector_joint_search_t *scaffold_search =
                                                calloc (1, sizeof *scaffold_search);

                                        assert (round_snapshot);
                                        assert (scaffold_search);
                                        assert (replay_search_selector_joint_blocks (decoder_model,
                                                                                    selector_model,
                                                                                    first_input,
                                                                                    first_target,
                                                                                    a_route,
                                                                                    second_input,
                                                                                    second_target,
                                                                                    b_route,
                                                                                    nodes,
                                                                                    owner_margin_reserve,
                                                                                    committed_b_selector_margin,
                                                                                    stage2_block_budget,
                                                                                    false,
                                                                                    &oracle_options,
                                                                                    scaffold_search));
                                        stage2_b_selecting_combined_blocks[1] += scaffold_search->b_selecting_blocks;
                                        if (scaffold_search->best_margin_after_repair > stage2_best_margin_after_repair[1])
                                                stage2_best_margin_after_repair[1] = scaffold_search->best_margin_after_repair;
                                        if (!scaffold_search->found_solution) {
                                                itty_feed_model_restore_final_layer_state_snapshot (selector_model, round_snapshot);
                                                free (scaffold_search);
                                                break;
                                        }

                                        committed_b_selector = scaffold_search->best_state.b_selector;
                                        committed_b_selector_margin = scaffold_search->best_state.b_margin;
                                        stage2_scaffold_accepted++;
                                        assert (replay_measure_shadow_selector_inventory (selector_model,
                                                                                          first_input,
                                                                                          first_target,
                                                                                          a_route,
                                                                                          second_input,
                                                                                          second_target,
                                                                                          b_route,
                                                                                          nodes,
                                                                                          &oracle_options,
                                                                                          &inventory));
                                        itty_feed_model_free_final_layer_state_snapshot (selector_model, round_snapshot);
                                        free (scaffold_search);
                                        if (committed_b_selector.selected_by_popcount == b_route &&
                                            committed_b_selector.popcount_gap >= 2)
                                                break;
                                }
                        }

                        printf ("---frozen_selector_words=%zu density=%zu/%zu seed=%zu a_sm=%td b_forced=%zu/%zu b_sm=%td b_sel=%zu strict_b_select=%zu scaffold_acc=%zu scaffold_best=%td scaffold_b_select=%zu\n",
                                selector_words,
                                selector_numerator,
                                selector_denominator,
                                selector_seed,
                                selector_stage0_summary.margin_after,
                                committed_b_forced.selected_distance,
                                committed_b_forced.false_negative_vote_deficit,
                                committed_b_selector_margin,
                                committed_b_selector.selected_by_popcount,
                                stage2_b_selecting_combined_blocks[0],
                                stage2_scaffold_accepted,
                                stage2_best_margin_after_repair[1],
                                stage2_b_selecting_combined_blocks[1]);

                        itty_feed_model_free (selector_model);
                }
        }
        printf ("---frozen-shadow-selector-end\n");

        free (a_routes);
        free (b_routes);
        itty_feed_model_free (baseline_selector_model);
        itty_feed_model_free (decoder_model);
        itty_bit_string_free (first_target);
        itty_bit_string_free (second_target);
        itty_bit_string_list_free (first_input);
        itty_bit_string_list_free (second_input);
}

static void
test_itty_feed_model_segment_replay_shadow_selector_frozen_decoder_sweep (void)
{
        test_itty_feed_model_segment_replay_shadow_selector_frozen_decoder_sweep_with_config (0, 0, 0, 0, false, false);
}

static void
test_itty_feed_model_segment_replay_shadow_selector_frozen_decoder_config (size_t selector_words,
                                                                           size_t selector_numerator,
                                                                           size_t selector_denominator,
                                                                           size_t selector_seed)
{
        test_itty_feed_model_segment_replay_shadow_selector_frozen_decoder_sweep_with_config (selector_words,
                                                                                               selector_numerator,
                                                                                               selector_denominator,
                                                                                               selector_seed,
                                                                                               true,
                                                                                               true);
}

static void
test_itty_feed_model_segment_replay_route_key_selector_diagnostic (void)
{
        size_t case_index = 2;
        size_t nodes = 8;
        size_t projection_batch_size = 64;
        size_t projection_max_rounds = 32;
        size_t projection_max_layer_flips = 128;
        size_t oracle_max_flips = 8;
        size_t stage1_step_limit = 8;
        ptrdiff_t owner_margin_reserve = 2;
        itty_feed_model_t *decoder_model = itty_feed_model_new (2, nodes, 1, 1);
        itty_feed_model_t *baseline_selector_model = itty_feed_model_new (2, nodes, 1, 1);
        itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
        itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
        itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
        itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
        itty_feed_model_refreshed_projected_repair_options_t first_options = {
                .batch_size = projection_batch_size,
                .max_rounds = projection_max_rounds,
                .max_layer_flips_per_batch = projection_max_layer_flips
        };
        itty_feed_model_train_options_t oracle_options = {
                .max_flips = oracle_max_flips,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_segment_training_summary_t first_summary;
        itty_feed_model_segment_training_summary_t selector_first_summary;
        itty_feed_model_segment_node_selection_summary_t a_selection = { 0 };
        itty_feed_model_segment_node_selection_summary_t b_selection = { 0 };
        itty_feed_model_decoder_objective_t *a_routes = calloc (nodes, sizeof *a_routes);
        itty_feed_model_decoder_objective_t *b_routes = calloc (nodes, sizeof *b_routes);
        replay_selector_block_summary_t selector_stage0 = { 0 };
        itty_feed_model_decoder_objective_t committed_b_forced = { 0 };
        itty_bit_string_t *route_keys[nodes];
        itty_bit_string_t *a_probe;
        itty_bit_string_t *b_probe;
        size_t a_route;
        size_t b_route;
        route_key_selection_summary_t before_replay = { 0 };
        route_key_selection_summary_t after_replay = { 0 };
        route_key_selection_summary_t after_replay_b = { 0 };
        itty_feed_model_train_stats_t replay_a_stats = { 0 };
        itty_feed_model_train_stats_t replay_b_stats = { 0 };
        itty_feed_model_decoder_objective_t after_replay_a_forced = { 0 };
        itty_feed_model_decoder_objective_t after_replay_b_forced = { 0 };
        itty_feed_model_decoder_objective_t after_replay2_a_forced = { 0 };
        itty_feed_model_decoder_objective_t after_replay2_b_forced = { 0 };

        memset (route_keys, 0, sizeof route_keys);
        assert (decoder_model && baseline_selector_model);
        assert (a_routes && b_routes);

        itty_feed_model_set_decoder (decoder_model, ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        itty_feed_model_set_decoder (baseline_selector_model, ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        assert (itty_feed_model_randomize_masks (decoder_model, 0xa100 + case_index, 1, 8));
        assert (itty_feed_model_randomize_masks (baseline_selector_model, 0xa100 + case_index, 1, 8));
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (decoder_model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &first_summary));
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (baseline_selector_model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &selector_first_summary));
        assert (itty_feed_model_measure_segment_node_selection (decoder_model,
                                                                first_input,
                                                                first_target,
                                                                &a_selection));
        assert (itty_feed_model_measure_segment_node_selection (decoder_model,
                                                                second_input,
                                                                second_target,
                                                                &b_selection));
        for (size_t route = 0; route < nodes; route++) {
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            first_input,
                                                                            first_target,
                                                                            route,
                                                                            &a_routes[route]));
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            second_input,
                                                                            second_target,
                                                                            route,
                                                                            &b_routes[route]));
        }
        a_route = replay_choose_a_route (a_routes, nodes, &a_selection);
        b_route = replay_choose_b_route (b_routes, nodes, a_route, &b_selection);

        assert (replay_apply_selector_protection_block (baseline_selector_model,
                                                        first_input,
                                                        first_target,
                                                        a_route,
                                                        NULL,
                                                        NULL,
                                                        0,
                                                        nodes,
                                                        owner_margin_reserve,
                                                        8,
                                                        &oracle_options,
                                                        &selector_stage0));

        for (size_t step = 0; step < stage1_step_limit; step++) {
            itty_feed_model_train_stats_t step_stats = { 0 };
            itty_feed_model_decoder_objective_t probe_b_before = { 0 };
            itty_feed_model_decoder_objective_t probe_b_after = { 0 };
            itty_feed_model_decoder_objective_t probe_a_forced = { 0 };
            itty_feed_model_decoder_objective_t probe_a_shadow = { 0 };
            itty_feed_model_segment_node_selection_summary_t probe_a_selector = { 0 };
            itty_feed_model_decoder_objective_t selector_routes[nodes];
            ptrdiff_t probe_a_margin = 0;

            assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                        second_input,
                                                                        second_target,
                                                                        b_route,
                                                                        &probe_b_before));
            if (!itty_feed_model_train_final_layer_with_suffix_oracle_for_node (decoder_model,
                                                                                second_input,
                                                                                second_target,
                                                                                b_route,
                                                                                &oracle_options,
                                                                                &step_stats))
                    break;
            if (step_stats.flips == 0)
                    break;

            assert (replay_measure_shadow_selected_decode (decoder_model,
                                                           baseline_selector_model,
                                                           first_input,
                                                           first_target,
                                                           &probe_a_selector,
                                                           &probe_a_shadow));
            assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                        first_input,
                                                                        first_target,
                                                                        a_route,
                                                                        &probe_a_forced));
            assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                        second_input,
                                                                        second_target,
                                                                        b_route,
                                                                        &probe_b_after));
            for (size_t route = 0; route < nodes; route++)
                    assert (itty_feed_model_measure_decoder_objective_for_node (baseline_selector_model,
                                                                                first_input,
                                                                                first_target,
                                                                                route,
                                                                                &selector_routes[route]));
            probe_a_margin = replay_route_margin_for_objectives (selector_routes, nodes, a_route);

            if (!replay_route_is_a_safe (&probe_a_forced) ||
                probe_a_shadow.selected_distance != 0 ||
                probe_a_margin < owner_margin_reserve ||
                !(probe_b_after.selected_distance < probe_b_before.selected_distance ||
                  (probe_b_after.selected_distance == probe_b_before.selected_distance &&
                   probe_b_after.false_negative_vote_deficit < probe_b_before.false_negative_vote_deficit)))
                    break;

            committed_b_forced = probe_b_after;
        }

        a_probe = itty_bit_string_clone (itty_bit_string_list_fetch (first_input, 0));
        b_probe = itty_bit_string_clone (itty_bit_string_list_fetch (second_input, 0));
        for (size_t route = 0; route < nodes; route++)
                route_keys[route] = create_bit_string (create_mixed_word ());
        itty_bit_string_free (route_keys[a_route]);
        route_keys[a_route] = itty_bit_string_clone (a_probe);
        itty_bit_string_free (route_keys[b_route]);
        route_keys[b_route] = itty_bit_string_clone (b_probe);

        before_replay = measure_route_key_selection (route_keys, nodes, a_probe, b_probe);

        assert (itty_feed_model_train_final_layer_with_suffix_oracle_for_node (decoder_model,
                                                                               first_input,
                                                                               first_target,
                                                                               a_route,
                                                                               &oracle_options,
                                                                               &replay_a_stats));
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    first_input,
                                                                    first_target,
                                                                    a_route,
                                                                    &after_replay_a_forced));
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    second_input,
                                                                    second_target,
                                                                    b_route,
                                                                    &after_replay_b_forced));
        after_replay = measure_route_key_selection (route_keys, nodes, a_probe, b_probe);

        assert (itty_feed_model_train_final_layer_with_suffix_oracle_for_node (decoder_model,
                                                                               second_input,
                                                                               second_target,
                                                                               b_route,
                                                                               &oracle_options,
                                                                               &replay_b_stats));
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    first_input,
                                                                    first_target,
                                                                    a_route,
                                                                    &after_replay2_a_forced));
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    second_input,
                                                                    second_target,
                                                                    b_route,
                                                                    &after_replay2_b_forced));
        after_replay_b = measure_route_key_selection (route_keys, nodes, a_probe, b_probe);

        assert (before_replay.a_selected_route == a_route);
        assert (before_replay.a_selected_gap >= 1);
        assert (before_replay.b_selected_route == b_route);
        assert (before_replay.b_selected_gap >= 1);
        assert (committed_b_forced.selected_distance <= 28);
        assert (after_replay_a_forced.selected_distance == 0);
        assert (after_replay.a_selected_route == a_route);
        assert (after_replay.a_selected_gap >= 1);
        assert (after_replay.b_selected_route == b_route);
        assert (after_replay.b_selected_gap >= 1);
        assert (after_replay2_a_forced.selected_distance == 0);
        assert (after_replay_b.a_selected_route == a_route);
        assert (after_replay_b.a_selected_gap >= 1);
        assert (after_replay_b.b_selected_route == b_route);
        assert (after_replay_b.b_selected_gap >= 1);

        printf ("---route-key-selector-begin\n");
        printf ("---a_route=%zu\n", a_route);
        printf ("---b_route=%zu\n", b_route);
        for (size_t route = 0; route < nodes; route++) {
                size_t a_score = itty_bit_string_evaluate_similarity (a_probe, route_keys[route]);
                size_t b_score = itty_bit_string_evaluate_similarity (b_probe, route_keys[route]);
                printf ("---route_score_%zu_a=%zu\n", route, a_score);
                printf ("---route_score_%zu_b=%zu\n", route, b_score);
        }
        printf ("---a_selected_route=%zu\n", before_replay.a_selected_route);
        printf ("---a_selected_gap=%zu\n", before_replay.a_selected_gap);
        printf ("---b_selected_route=%zu\n", before_replay.b_selected_route);
        printf ("---b_selected_gap=%zu\n", before_replay.b_selected_gap);
        printf ("---b_forced_dist=%zu\n", committed_b_forced.selected_distance);
        printf ("---b_forced_def=%zu\n", committed_b_forced.false_negative_vote_deficit);
        printf ("---replay_a_flips=%zu\n", replay_a_stats.flips);
        printf ("---after_replay_a_forced_dist=%zu\n", after_replay_a_forced.selected_distance);
        printf ("---after_replay_b_forced_dist=%zu\n", after_replay_b_forced.selected_distance);
        printf ("---after_replay_b_forced_def=%zu\n", after_replay_b_forced.false_negative_vote_deficit);
        printf ("---after_replay_a_selected_route=%zu\n", after_replay.a_selected_route);
        printf ("---after_replay_a_selected_gap=%zu\n", after_replay.a_selected_gap);
        printf ("---after_replay_b_selected_route=%zu\n", after_replay.b_selected_route);
        printf ("---after_replay_b_selected_gap=%zu\n", after_replay.b_selected_gap);
        printf ("---replay_b_flips=%zu\n", replay_b_stats.flips);
        printf ("---after_replay2_a_forced_dist=%zu\n", after_replay2_a_forced.selected_distance);
        printf ("---after_replay2_b_forced_dist=%zu\n", after_replay2_b_forced.selected_distance);
        printf ("---after_replay2_b_forced_def=%zu\n", after_replay2_b_forced.false_negative_vote_deficit);
        printf ("---after_replay2_a_selected_route=%zu\n", after_replay_b.a_selected_route);
        printf ("---after_replay2_a_selected_gap=%zu\n", after_replay_b.a_selected_gap);
        printf ("---after_replay2_b_selected_route=%zu\n", after_replay_b.b_selected_route);
        printf ("---after_replay2_b_selected_gap=%zu\n", after_replay_b.b_selected_gap);
        printf ("---route-key-selector-end\n");

        for (size_t route = 0; route < nodes; route++)
                itty_bit_string_free (route_keys[route]);
        itty_bit_string_free (a_probe);
        itty_bit_string_free (b_probe);
        free (a_routes);
        free (b_routes);
        itty_feed_model_free (baseline_selector_model);
        itty_feed_model_free (decoder_model);
        itty_bit_string_free (first_target);
        itty_bit_string_free (second_target);
        itty_bit_string_list_free (first_input);
        itty_bit_string_list_free (second_input);
}

static void
test_itty_feed_model_segment_replay_route_key_selector_add_c_diagnostic (void)
{
        size_t case_index = 2;
        size_t nodes = 8;
        size_t projection_batch_size = 64;
        size_t projection_max_rounds = 32;
        size_t projection_max_layer_flips = 128;
        size_t oracle_max_flips = 8;
        size_t stage1_step_limit = 8;
        ptrdiff_t owner_margin_reserve = 2;
        itty_feed_model_t *decoder_model = itty_feed_model_new (2, nodes, 1, 1);
        itty_feed_model_t *baseline_selector_model = itty_feed_model_new (2, nodes, 1, 1);
        itty_bit_string_list_t *first_input = create_input_with_count (1, 0);
        itty_bit_string_list_t *second_input = create_input_with_count (1, 1);
        itty_bit_string_list_t *third_input = create_input_with_count (1,
                                                                       create_mixed_word () ^ ((size_t) 1 << 1));
        itty_bit_string_t *first_target = create_bit_string ((size_t) 1 << 3);
        itty_bit_string_t *second_target = create_bit_string (create_half_populated_word ());
        itty_bit_string_t *third_target = create_bit_string (((size_t) 1 << 1) | ((size_t) 1 << 6));
        itty_feed_model_refreshed_projected_repair_options_t first_options = {
                .batch_size = projection_batch_size,
                .max_rounds = projection_max_rounds,
                .max_layer_flips_per_batch = projection_max_layer_flips
        };
        itty_feed_model_train_options_t oracle_options = {
                .max_flips = oracle_max_flips,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_segment_training_summary_t first_summary;
        itty_feed_model_segment_training_summary_t selector_first_summary;
        itty_feed_model_segment_node_selection_summary_t a_selection = { 0 };
        itty_feed_model_segment_node_selection_summary_t b_selection = { 0 };
        itty_feed_model_decoder_objective_t *a_routes = calloc (nodes, sizeof *a_routes);
        itty_feed_model_decoder_objective_t *b_routes = calloc (nodes, sizeof *b_routes);
        itty_feed_model_decoder_objective_t *c_routes = calloc (nodes, sizeof *c_routes);
        replay_selector_block_summary_t selector_stage0 = { 0 };
        itty_feed_model_decoder_objective_t committed_b_forced = { 0 };
        itty_bit_string_t *route_keys[nodes];
        itty_bit_string_t *a_probe;
        itty_bit_string_t *b_probe;
        itty_bit_string_t *c_probe;
        size_t a_route;
        size_t b_route;
        size_t c_route;
        route_key_selection_summary_abc_t before_c = { 0 };
        route_key_selection_summary_abc_t after_c = { 0 };
        itty_feed_model_train_stats_t replay_a_stats = { 0 };
        itty_feed_model_train_stats_t replay_b_stats = { 0 };
        itty_feed_model_train_stats_t replay_c_stats = { 0 };
        itty_feed_model_decoder_objective_t after_replay2_a_forced = { 0 };
        itty_feed_model_decoder_objective_t after_replay2_b_forced = { 0 };
        itty_feed_model_decoder_objective_t before_c_forced = { 0 };
        itty_feed_model_decoder_objective_t after_c_a_forced = { 0 };
        itty_feed_model_decoder_objective_t after_c_b_forced = { 0 };
        itty_feed_model_decoder_objective_t after_c_c_forced = { 0 };

        memset (route_keys, 0, sizeof route_keys);
        assert (decoder_model && baseline_selector_model);
        assert (a_routes && b_routes && c_routes);

        itty_feed_model_set_decoder (decoder_model, ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        itty_feed_model_set_decoder (baseline_selector_model, ITTY_FEED_MODEL_DECODER_SEGMENT_CONDENSE);
        assert (itty_feed_model_randomize_masks (decoder_model, 0xa100 + case_index, 1, 8));
        assert (itty_feed_model_randomize_masks (baseline_selector_model, 0xa100 + case_index, 1, 8));
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (decoder_model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &first_summary));
        assert (itty_feed_model_train_segment_condense_quota_repair_projection (baseline_selector_model,
                                                                                first_input,
                                                                                first_target,
                                                                                &first_options,
                                                                                &selector_first_summary));
        assert (itty_feed_model_measure_segment_node_selection (decoder_model,
                                                                first_input,
                                                                first_target,
                                                                &a_selection));
        assert (itty_feed_model_measure_segment_node_selection (decoder_model,
                                                                second_input,
                                                                second_target,
                                                                &b_selection));
        for (size_t route = 0; route < nodes; route++) {
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            first_input,
                                                                            first_target,
                                                                            route,
                                                                            &a_routes[route]));
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            second_input,
                                                                            second_target,
                                                                            route,
                                                                            &b_routes[route]));
        }
        a_route = replay_choose_a_route (a_routes, nodes, &a_selection);
        b_route = replay_choose_b_route (b_routes, nodes, a_route, &b_selection);

        assert (replay_apply_selector_protection_block (baseline_selector_model,
                                                        first_input,
                                                        first_target,
                                                        a_route,
                                                        NULL,
                                                        NULL,
                                                        0,
                                                        nodes,
                                                        owner_margin_reserve,
                                                        8,
                                                        &oracle_options,
                                                        &selector_stage0));

        for (size_t step = 0; step < stage1_step_limit; step++) {
                itty_feed_model_train_stats_t step_stats = { 0 };
                itty_feed_model_decoder_objective_t probe_b_before = { 0 };
                itty_feed_model_decoder_objective_t probe_b_after = { 0 };
                itty_feed_model_decoder_objective_t probe_a_forced = { 0 };
                itty_feed_model_decoder_objective_t probe_a_shadow = { 0 };
                itty_feed_model_segment_node_selection_summary_t probe_a_selector = { 0 };
                itty_feed_model_decoder_objective_t selector_routes[nodes];
                ptrdiff_t probe_a_margin = 0;

                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            second_input,
                                                                            second_target,
                                                                            b_route,
                                                                            &probe_b_before));
                if (!itty_feed_model_train_final_layer_with_suffix_oracle_for_node (decoder_model,
                                                                                    second_input,
                                                                                    second_target,
                                                                                    b_route,
                                                                                    &oracle_options,
                                                                                    &step_stats))
                        break;
                if (step_stats.flips == 0)
                        break;

                assert (replay_measure_shadow_selected_decode (decoder_model,
                                                               baseline_selector_model,
                                                               first_input,
                                                               first_target,
                                                               &probe_a_selector,
                                                               &probe_a_shadow));
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            first_input,
                                                                            first_target,
                                                                            a_route,
                                                                            &probe_a_forced));
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            second_input,
                                                                            second_target,
                                                                            b_route,
                                                                            &probe_b_after));
                for (size_t route = 0; route < nodes; route++)
                        assert (itty_feed_model_measure_decoder_objective_for_node (baseline_selector_model,
                                                                                    first_input,
                                                                                    first_target,
                                                                                    route,
                                                                                    &selector_routes[route]));
                probe_a_margin = replay_route_margin_for_objectives (selector_routes, nodes, a_route);

                if (!replay_route_is_a_safe (&probe_a_forced) ||
                    probe_a_shadow.selected_distance != 0 ||
                    probe_a_margin < owner_margin_reserve ||
                    !(probe_b_after.selected_distance < probe_b_before.selected_distance ||
                      (probe_b_after.selected_distance == probe_b_before.selected_distance &&
                       probe_b_after.false_negative_vote_deficit < probe_b_before.false_negative_vote_deficit)))
                        break;

                committed_b_forced = probe_b_after;
        }

        assert (itty_feed_model_train_final_layer_with_suffix_oracle_for_node (decoder_model,
                                                                               first_input,
                                                                               first_target,
                                                                               a_route,
                                                                               &oracle_options,
                                                                               &replay_a_stats));
        assert (itty_feed_model_train_final_layer_with_suffix_oracle_for_node (decoder_model,
                                                                               second_input,
                                                                               second_target,
                                                                               b_route,
                                                                               &oracle_options,
                                                                               &replay_b_stats));
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    first_input,
                                                                    first_target,
                                                                    a_route,
                                                                    &after_replay2_a_forced));
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    second_input,
                                                                    second_target,
                                                                    b_route,
                                                                    &after_replay2_b_forced));

        for (size_t route = 0; route < nodes; route++)
                assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                            third_input,
                                                                            third_target,
                                                                            route,
                                                                            &c_routes[route]));
        c_route = replay_choose_c_route (c_routes, nodes, a_route, b_route);

        a_probe = itty_bit_string_clone (itty_bit_string_list_fetch (first_input, 0));
        b_probe = itty_bit_string_clone (itty_bit_string_list_fetch (second_input, 0));
        c_probe = itty_bit_string_clone (itty_bit_string_list_fetch (third_input, 0));
        for (size_t route = 0; route < nodes; route++)
                route_keys[route] = create_bit_string (create_mixed_word ());
        itty_bit_string_free (route_keys[a_route]);
        route_keys[a_route] = itty_bit_string_clone (a_probe);
        itty_bit_string_free (route_keys[b_route]);
        route_keys[b_route] = itty_bit_string_clone (b_probe);
        itty_bit_string_free (route_keys[c_route]);
        route_keys[c_route] = itty_bit_string_clone (c_probe);

        before_c = measure_route_key_selection_abc (route_keys, nodes, a_probe, b_probe, c_probe);
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    third_input,
                                                                    third_target,
                                                                    c_route,
                                                                    &before_c_forced));

        assert (itty_feed_model_train_final_layer_with_suffix_oracle_for_node (decoder_model,
                                                                               third_input,
                                                                               third_target,
                                                                               c_route,
                                                                               &oracle_options,
                                                                               &replay_c_stats));
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    first_input,
                                                                    first_target,
                                                                    a_route,
                                                                    &after_c_a_forced));
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    second_input,
                                                                    second_target,
                                                                    b_route,
                                                                    &after_c_b_forced));
        assert (itty_feed_model_measure_decoder_objective_for_node (decoder_model,
                                                                    third_input,
                                                                    third_target,
                                                                    c_route,
                                                                    &after_c_c_forced));
        after_c = measure_route_key_selection_abc (route_keys, nodes, a_probe, b_probe, c_probe);

        assert (c_route != a_route);
        assert (c_route != b_route);
        assert (before_c.a_selected_route == a_route);
        assert (before_c.a_selected_gap >= 1);
        assert (before_c.b_selected_route == b_route);
        assert (before_c.b_selected_gap >= 1);
        assert (before_c.c_selected_route == c_route);
        assert (before_c.c_selected_gap >= 1);
        assert (committed_b_forced.selected_distance <= 28);
        assert (after_replay2_a_forced.selected_distance == 0);
        assert (after_replay2_b_forced.selected_distance <= committed_b_forced.selected_distance);
        assert (after_c.a_selected_route == a_route);
        assert (after_c.a_selected_gap >= 1);
        assert (after_c.b_selected_route == b_route);
        assert (after_c.b_selected_gap >= 1);
        assert (after_c.c_selected_route == c_route);
        assert (after_c.c_selected_gap >= 1);
        assert (after_c_a_forced.selected_distance == 0);
        assert (after_c_b_forced.selected_distance <= after_replay2_b_forced.selected_distance + 1);
        assert (after_c_c_forced.selected_distance <= before_c_forced.selected_distance);

        printf ("---route-key-selector-c-begin\n");
        printf ("---a_route=%zu\n", a_route);
        printf ("---b_route=%zu\n", b_route);
        printf ("---c_route=%zu\n", c_route);
        printf ("---before_c_a_selected_route=%zu\n", before_c.a_selected_route);
        printf ("---before_c_a_selected_gap=%zu\n", before_c.a_selected_gap);
        printf ("---before_c_b_selected_route=%zu\n", before_c.b_selected_route);
        printf ("---before_c_b_selected_gap=%zu\n", before_c.b_selected_gap);
        printf ("---before_c_c_selected_route=%zu\n", before_c.c_selected_route);
        printf ("---before_c_c_selected_gap=%zu\n", before_c.c_selected_gap);
        printf ("---replay_a_flips=%zu\n", replay_a_stats.flips);
        printf ("---replay_b_flips=%zu\n", replay_b_stats.flips);
        printf ("---replay_c_flips=%zu\n", replay_c_stats.flips);
        printf ("---b_forced_after_ab_dist=%zu\n", after_replay2_b_forced.selected_distance);
        printf ("---b_forced_after_ab_def=%zu\n", after_replay2_b_forced.false_negative_vote_deficit);
        printf ("---c_forced_before_dist=%zu\n", before_c_forced.selected_distance);
        printf ("---c_forced_before_def=%zu\n", before_c_forced.false_negative_vote_deficit);
        printf ("---after_c_a_forced_dist=%zu\n", after_c_a_forced.selected_distance);
        printf ("---after_c_b_forced_dist=%zu\n", after_c_b_forced.selected_distance);
        printf ("---after_c_b_forced_def=%zu\n", after_c_b_forced.false_negative_vote_deficit);
        printf ("---after_c_c_forced_dist=%zu\n", after_c_c_forced.selected_distance);
        printf ("---after_c_c_forced_def=%zu\n", after_c_c_forced.false_negative_vote_deficit);
        printf ("---after_c_a_selected_route=%zu\n", after_c.a_selected_route);
        printf ("---after_c_a_selected_gap=%zu\n", after_c.a_selected_gap);
        printf ("---after_c_b_selected_route=%zu\n", after_c.b_selected_route);
        printf ("---after_c_b_selected_gap=%zu\n", after_c.b_selected_gap);
        printf ("---after_c_c_selected_route=%zu\n", after_c.c_selected_route);
        printf ("---after_c_c_selected_gap=%zu\n", after_c.c_selected_gap);
        printf ("---route-key-selector-c-end\n");

        for (size_t route = 0; route < nodes; route++)
                itty_bit_string_free (route_keys[route]);
        itty_bit_string_free (a_probe);
        itty_bit_string_free (b_probe);
        itty_bit_string_free (c_probe);
        free (a_routes);
        free (b_routes);
        free (c_routes);
        itty_feed_model_free (baseline_selector_model);
        itty_feed_model_free (decoder_model);
        itty_bit_string_free (first_target);
        itty_bit_string_free (second_target);
        itty_bit_string_free (third_target);
        itty_bit_string_list_free (first_input);
        itty_bit_string_list_free (second_input);
        itty_bit_string_list_free (third_input);
}

static void
test_itty_feed_model_four_node_plateau_tracks_layer5_blocker_policy (void)
{
        itty_feed_model_t *layer6_model = itty_feed_model_new (8, 4, 1, 1);
        itty_feed_model_t *blocker_model = itty_feed_model_new (8, 4, 1, 1);
        itty_bit_string_list_t *input = create_input_with_count (1, 0);
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t train_options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_refreshed_projected_repair_options_t layer6_options = {
                .batch_size = 64,
                .max_rounds = 128,
                .max_layer_flips_per_batch = 256
        };
        itty_feed_model_projected_repair_options_t blocker_options = {
                .max_projected_blocks = 8,
                .max_layer_flips = 16,
                .max_strict_distance_blocks = 0,
                .limit_strict_distance_blocks = true,
                .max_blocker_blocks = 1,
                .limit_blocker_blocks = true
        };
        itty_feed_model_refreshed_projected_repair_stats_t layer6_stats;
        itty_feed_model_projected_repair_stats_t blocker_stats;
        itty_feed_model_refreshed_projected_repair_stats_t blocker_cleanup_stats;

        train_hard_feed_model_to_stall_with_shape (layer6_model,
                                                   input,
                                                   target,
                                                   &train_options,
                                                   7);
        assert (itty_feed_model_train_penultimate_layer_with_refreshed_final_repairs (layer6_model,
                                                                                     input,
                                                                                     target,
                                                                                     &layer6_options,
                                                                                     &layer6_stats));
        assert (layer6_stats.after_distance == 23);
        assert (layer6_stats.after_blockers == 676);
        assert (layer6_stats.projected.layer_flips == 2898);

        train_hard_feed_model_to_stall_with_shape (blocker_model,
                                                   input,
                                                   target,
                                                   &train_options,
                                                   7);
        assert (itty_feed_model_train_antepenultimate_layer_with_projected_repairs (blocker_model,
                                                                                   input,
                                                                                   target,
                                                                                   &blocker_options,
                                                                                   &blocker_stats));
        assert (blocker_stats.accepted_blocks == 1);
        assert (blocker_stats.accepted_strict_distance_blocks == 0);
        assert (blocker_stats.accepted_blocker_blocks == 1);
        assert (blocker_stats.after_distance == 32);
        assert (blocker_stats.after_blockers == 1444);
        assert (itty_feed_model_train_penultimate_layer_with_refreshed_final_repairs (blocker_model,
                                                                                     input,
                                                                                     target,
                                                                                     &layer6_options,
                                                                                     &blocker_cleanup_stats));
        assert (blocker_cleanup_stats.after_distance == 23);
        assert (blocker_cleanup_stats.projected.layer_flips > 0);

        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (layer6_model);
        itty_feed_model_free (blocker_model);
}

static int
run_replay_focus_suite (void)
{
        run_segment_replay_final_layer_oracle_diagnostic ();
        run_segment_replay_final_layer_transaction_diagnostic ();
        run_segment_replay_restore_failure_diagnostic ();
        run_segment_replay_contender_restore_diagnostic ();
        run_segment_replay_transaction_scaffold_diagnostic ();
        test_itty_feed_model_segment_replay_transaction_capacity_matrix ();
        test_itty_feed_model_segment_replay_forced_route_diagnostic ();
        test_itty_feed_model_segment_replay_route_assignment_diagnostic ();
        test_itty_feed_model_segment_replay_route_aware_training_diagnostic ();
        test_itty_feed_model_segment_replay_route_owned_training_diagnostic ();
        test_itty_feed_model_segment_replay_route_owned_lane_split_diagnostic ();
        test_itty_feed_model_segment_replay_shadow_selector_diagnostic ();
        test_itty_feed_model_segment_replay_guard_blocks_solved_example_clobber ();
        printf ("Focused replay diagnostics passed.\n");
        return 0;
}

int
main (int   argc,
      char *argv[])
{
        if (argc > 1 && strcmp (argv[1], "--focus-replay") == 0)
                return run_replay_focus_suite ();
        if (argc > 1 && strcmp (argv[1], "--focus-shadow-selector") == 0) {
                test_itty_feed_model_segment_replay_shadow_selector_diagnostic ();
                printf ("Focused shadow selector diagnostics passed.\n");
                return 0;
        }
        if (argc > 1 && strcmp (argv[1], "--focus-shadow-selector-density") == 0) {
                test_itty_feed_model_segment_replay_shadow_selector_density_sweep ();
                printf ("Focused shadow selector density diagnostics passed.\n");
                return 0;
        }
        if (argc > 3 && strcmp (argv[1], "--focus-shadow-selector-config") == 0) {
                size_t numerator = (size_t) strtoull (argv[2], NULL, 10);
                size_t denominator = (size_t) strtoull (argv[3], NULL, 10);
                test_itty_feed_model_segment_replay_shadow_selector_diagnostic_with_selector_config (1,
                                                                                                     numerator,
                                                                                                     denominator);
                printf ("Focused shadow selector config diagnostics passed.\n");
                return 0;
        }
        if (argc > 1 && strcmp (argv[1], "--focus-shadow-selector-wide") == 0) {
                test_itty_feed_model_segment_replay_shadow_selector_diagnostic_with_selector_config (2, 1, 8);
                printf ("Focused shadow selector wide diagnostics passed.\n");
                return 0;
        }
        if (argc > 1 && strcmp (argv[1], "--focus-shadow-selector-frozen-sweep") == 0) {
                test_itty_feed_model_segment_replay_shadow_selector_frozen_decoder_sweep ();
                printf ("Focused shadow selector frozen sweep diagnostics passed.\n");
                return 0;
        }
        if (argc > 5 && strcmp (argv[1], "--focus-shadow-selector-frozen-config") == 0) {
                size_t selector_words = (size_t) strtoul (argv[2], NULL, 10);
                size_t selector_numerator = (size_t) strtoul (argv[3], NULL, 10);
                size_t selector_denominator = (size_t) strtoul (argv[4], NULL, 10);
                size_t selector_seed = (size_t) strtoul (argv[5], NULL, 10);
                test_itty_feed_model_segment_replay_shadow_selector_frozen_decoder_config (selector_words,
                                                                                            selector_numerator,
                                                                                            selector_denominator,
                                                                                            selector_seed);
                printf ("Focused shadow selector frozen config diagnostics passed.\n");
                return 0;
        }
        if (argc > 1 && strcmp (argv[1], "--focus-route-key-selector") == 0) {
                test_itty_feed_model_segment_replay_route_key_selector_diagnostic ();
                printf ("Focused route-key selector diagnostics passed.\n");
                return 0;
        }
        if (argc > 1 && strcmp (argv[1], "--focus-route-key-selector-c") == 0) {
                test_itty_feed_model_segment_replay_route_key_selector_add_c_diagnostic ();
                printf ("Focused route-key selector C diagnostics passed.\n");
                return 0;
        }

        test_itty_feed_model_train_one_learns_single_mask ();
        test_itty_feed_model_train_one_with_budget_moves_toward_target ();
        test_itty_feed_model_train_one_with_budget_prioritizes_largest_error ();
        test_itty_feed_model_train_one_trains_final_layer ();
        test_itty_feed_model_train_one_supports_multi_node_final_layer ();
        test_itty_feed_model_train_backwards_one_trains_all_layers ();
        test_itty_feed_model_rotation_affects_forward_output ();
        test_itty_feed_model_randomizes_masks_with_fixed_seed ();
        test_itty_feed_model_measures_final_layer_node_diagnostics ();
        test_itty_feed_model_randomized_masks_diverge_final_layer_node_diagnostics ();
        test_itty_feed_model_train_backwards_one_supports_hidden_rotation ();
        test_itty_feed_model_measures_segment_node_selection ();
        test_itty_feed_model_train_backwards_one_rejects_multi_node_chained_reduce ();
        test_itty_feed_model_measures_backward_layer_diagnostics ();
        test_itty_feed_model_measures_suffix_oracle ();
        test_itty_feed_model_final_layer_suffix_oracle_trainer_improves_hard_case ();
        test_itty_feed_model_penultimate_projected_repairs_reduce_final_cleanup ();
        test_itty_feed_model_segment_condense_decoder_reduces_target_one_fragility ();
        test_itty_feed_model_segment_shape_matrix_smoke ();
        test_itty_feed_model_segment_training_limits_multi_example_clobber ();
        test_itty_feed_model_segment_training_tracks_distinct_target_clobber ();
        run_replay_focus_suite ();
        test_itty_feed_model_segment_replay_scaffold_matrix ();
        test_itty_feed_model_segment_replay_transaction_capacity_matrix ();
        test_itty_feed_model_segment_replay_forced_route_diagnostic ();
        test_itty_feed_model_segment_replay_route_assignment_diagnostic ();
        test_itty_feed_model_segment_replay_route_aware_training_diagnostic ();
        test_itty_feed_model_segment_replay_route_owned_training_diagnostic ();
        test_itty_feed_model_segment_replay_route_owned_lane_split_diagnostic ();
        test_itty_feed_model_segment_replay_shadow_selector_diagnostic ();
        test_itty_feed_model_segment_training_matrix_tracks_bounded_runs ();
        test_itty_feed_model_four_node_plateau_tracks_layer5_blocker_policy ();
        printf ("All itty-feed-model tests passed.\n");
        return 0;
}
