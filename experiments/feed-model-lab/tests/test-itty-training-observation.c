#include "itty-feed-model.h"
#include "itty-model-metrics.h"
#include "itty-training-observation.h"
#include "itty-vocabulary.h"
#include <assert.h>
#include <math.h>
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

static itty_bit_string_list_t *
create_input (void)
{
        itty_bit_string_list_t *input = itty_bit_string_list_new ();
        itty_bit_string_list_append (input, create_bit_string (0b0011));
        return input;
}

static itty_bit_string_list_t *
create_zero_input (void)
{
        itty_bit_string_list_t *input = itty_bit_string_list_new ();
        itty_bit_string_list_append (input, create_bit_string (0));
        return input;
}

static size_t
create_half_populated_word (void)
{
        return ((size_t) 1 << (ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2)) - 1;
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

static void
assert_near (double actual,
             double expected)
{
        assert (fabs (actual - expected) < 0.000001);
}

static void
test_itty_training_observation_feed_model_train_one (void)
{
        char *text_file = NULL;
        char *bit_string_file = NULL;
        itty_vocabulary_t *vocabulary = create_vocabulary (&text_file, &bit_string_file);
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = itty_vocabulary_translate_to_bit_string (vocabulary, " b");
        itty_model_metrics_bit_summary_t summary;
        itty_model_metrics_activation_trace_t *trace = NULL;
        itty_feed_model_train_stats_t stats;
        size_t distance = 0;

        itty_training_observation_t *observation = itty_training_observation_feed_model_train_one (model,
                                                                                                    input,
                                                                                                    target,
                                                                                                    NULL,
                                                                                                    NULL);
        assert (observation != NULL);
        assert (itty_training_observation_did_train (observation));

        assert (itty_training_observation_get_before_mask_summary (observation, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert (summary.set_bits == 0);
        assert_near (summary.entropy, 0.0);

        assert (itty_training_observation_get_after_mask_summary (observation, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert (summary.set_bits == 4);
        assert (summary.entropy > 0.0);

        assert (itty_training_observation_get_train_stats (observation, &stats));
        assert (stats.flips == 4);
        assert (stats.candidate_bits == 4);
        assert (stats.largest_error == 1);

        assert (itty_training_observation_get_before_distance (observation, &distance));
        assert (distance == 4);
        assert (itty_training_observation_get_after_distance (observation, &distance));
        assert (distance == 0);

        trace = itty_training_observation_get_before_activations (observation);
        assert (trace != NULL);
        assert (itty_model_metrics_activation_trace_get_layer_count (trace) == 1);
        assert (itty_model_metrics_activation_trace_get_layer_summary (trace, 0, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS * 2);
        assert (summary.set_bits == 4);

        trace = itty_training_observation_get_after_activations (observation);
        assert (trace != NULL);
        assert (itty_model_metrics_activation_trace_get_layer_count (trace) == 1);
        assert (itty_model_metrics_activation_trace_get_layer_summary (trace, 0, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS * 2);
        assert (summary.set_bits == 4);

        itty_training_observation_free (observation);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

static void
test_itty_training_observation_feed_model_train_one_dense_target (void)
{
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_model_metrics_bit_summary_t summary;
        itty_model_metrics_activation_trace_t *trace = NULL;
        itty_feed_model_train_stats_t stats;
        size_t distance = 0;

        itty_training_observation_t *observation = itty_training_observation_feed_model_train_one (model,
                                                                                                    input,
                                                                                                    target,
                                                                                                    NULL,
                                                                                                    NULL);
        assert (observation != NULL);
        assert (itty_training_observation_did_train (observation));

        assert (itty_training_observation_get_before_mask_summary (observation, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert (summary.set_bits == 0);
        assert_near (summary.entropy, 0.0);

        assert (itty_training_observation_get_after_mask_summary (observation, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert (summary.set_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2);
        assert_near (summary.set_density, 0.5);
        assert_near (summary.entropy, 1.0);

        assert (itty_training_observation_get_train_stats (observation, &stats));
        assert (stats.flips == ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2);
        assert (stats.candidate_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2);
        assert (stats.largest_error == 1);

        assert (itty_training_observation_get_before_distance (observation, &distance));
        assert (distance == ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2);
        assert (itty_training_observation_get_after_distance (observation, &distance));
        assert (distance == 0);

        trace = itty_training_observation_get_before_activations (observation);
        assert (trace != NULL);
        assert (itty_model_metrics_activation_trace_get_layer_count (trace) == 1);
        assert (itty_model_metrics_activation_trace_get_layer_summary (trace, 0, &summary));
        assert (summary.bit_count == 0);
        assert (summary.set_bits == 0);
        assert_near (summary.entropy, 0.0);

        trace = itty_training_observation_get_after_activations (observation);
        assert (trace != NULL);
        assert (itty_model_metrics_activation_trace_get_layer_count (trace) == 1);
        assert (itty_model_metrics_activation_trace_get_layer_summary (trace, 0, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS * 2);
        assert (summary.set_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert_near (summary.set_density, 0.5);
        assert_near (summary.entropy, 1.0);

        itty_training_observation_free (observation);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_history_feed_model_train_one_budgeted_dense_target (void)
{
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST
        };
        size_t expected_before_distances[] = { 32, 24, 16, 8 };
        size_t expected_after_distances[] = { 24, 16, 8, 0 };
        itty_training_step_summary_t summary;

        itty_training_history_t *history = itty_training_history_feed_model_train_one (model,
                                                                                       input,
                                                                                       target,
                                                                                       &options,
                                                                                       NULL,
                                                                                       8);
        assert (history != NULL);
        assert (itty_training_history_get_step_count (history) == 4);

        for (size_t i = 0; i < 4; i++) {
                assert (itty_training_history_get_step_summary (history, i, &summary));
                assert (summary.step == i);
                assert (summary.before_distance == expected_before_distances[i]);
                assert (summary.after_distance == expected_after_distances[i]);
                assert (summary.flips == 8);
                assert (summary.candidate_bits == expected_before_distances[i]);
                assert (summary.largest_error == 1);
        }

        assert (itty_training_history_get_step_summary (history, 0, &summary));
        assert_near (summary.before_mask_entropy, 0.0);
        assert (summary.after_mask_entropy > 0.0);
        assert_near (summary.before_activation_entropy, 0.0);
        assert (summary.after_activation_entropy > 0.0);

        assert (itty_training_history_get_step_summary (history, 3, &summary));
        assert_near (summary.after_mask_entropy, 1.0);
        assert_near (summary.after_activation_entropy, 1.0);

        itty_training_history_free (history);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_optimizer_fixed_budget_matches_options (void)
{
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_training_optimizer_t *optimizer = itty_training_optimizer_new_fixed_budget (8);
        itty_training_step_summary_t summary;

        itty_training_history_t *history = itty_training_history_feed_model_train_one_with_optimizer (model,
                                                                                                      input,
                                                                                                      target,
                                                                                                      optimizer,
                                                                                                      NULL,
                                                                                                      8);
        assert (history != NULL);
        assert (itty_training_optimizer_get_kind (optimizer) == ITTY_TRAINING_OPTIMIZER_FIXED_BUDGET);
        assert (itty_training_history_get_step_count (history) == 4);
        for (size_t i = 0; i < 4; i++) {
                assert (itty_training_history_get_step_summary (history, i, &summary));
                assert (summary.flips == 8);
        }

        itty_training_history_free (history);
        itty_training_optimizer_free (optimizer);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_optimizer_distance_capped_budget_limits_sparse_target (void)
{
        char *text_file = NULL;
        char *bit_string_file = NULL;
        itty_vocabulary_t *vocabulary = create_vocabulary (&text_file, &bit_string_file);
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = itty_vocabulary_translate_to_bit_string (vocabulary, " b");
        itty_training_optimizer_t *optimizer = itty_training_optimizer_new_distance_capped_budget (8);
        itty_training_step_summary_t summary;

        itty_training_history_t *history = itty_training_history_feed_model_train_one_with_optimizer (model,
                                                                                                      input,
                                                                                                      target,
                                                                                                      optimizer,
                                                                                                      NULL,
                                                                                                      8);
        assert (history != NULL);
        assert (itty_training_optimizer_get_kind (optimizer) == ITTY_TRAINING_OPTIMIZER_DISTANCE_CAPPED_BUDGET);
        assert (itty_training_history_get_step_count (history) == 1);
        assert (itty_training_history_get_step_summary (history, 0, &summary));
        assert (summary.before_distance == 4);
        assert (summary.after_distance == 0);
        assert (summary.flips == 4);
        assert (summary.candidate_bits == 4);

        itty_training_history_free (history);
        itty_training_optimizer_free (optimizer);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

static void
test_itty_training_optimizer_distance_fraction_budget_halves_dense_target (void)
{
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_training_optimizer_t *optimizer = itty_training_optimizer_new_distance_fraction_budget (1, 2, 32);
        size_t expected_before_distances[] = { 32, 16, 8, 4, 2, 1 };
        size_t expected_after_distances[] = { 16, 8, 4, 2, 1, 0 };
        size_t expected_flips[] = { 16, 8, 4, 2, 1, 1 };
        itty_training_step_summary_t summary;

        itty_training_history_t *history = itty_training_history_feed_model_train_one_with_optimizer (model,
                                                                                                      input,
                                                                                                      target,
                                                                                                      optimizer,
                                                                                                      NULL,
                                                                                                      8);
        assert (history != NULL);
        assert (itty_training_optimizer_get_kind (optimizer) == ITTY_TRAINING_OPTIMIZER_DISTANCE_FRACTION_BUDGET);
        assert (itty_training_history_get_step_count (history) == 6);

        for (size_t i = 0; i < 6; i++) {
                assert (itty_training_history_get_step_summary (history, i, &summary));
                assert (summary.before_distance == expected_before_distances[i]);
                assert (summary.after_distance == expected_after_distances[i]);
                assert (summary.flips == expected_flips[i]);
                assert (summary.candidate_bits == expected_before_distances[i]);
        }

        assert (itty_training_history_get_step_summary (history, 5, &summary));
        assert_near (summary.after_mask_entropy, 1.0);
        assert_near (summary.after_activation_entropy, 1.0);

        itty_training_history_free (history);
        itty_training_optimizer_free (optimizer);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_optimizer_distance_fraction_budget_halves_sparse_target (void)
{
        char *text_file = NULL;
        char *bit_string_file = NULL;
        itty_vocabulary_t *vocabulary = create_vocabulary (&text_file, &bit_string_file);
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *input = create_input ();
        itty_bit_string_t *target = itty_vocabulary_translate_to_bit_string (vocabulary, " b");
        itty_training_optimizer_t *optimizer = itty_training_optimizer_new_distance_fraction_budget (1, 2, 32);
        size_t expected_before_distances[] = { 4, 2, 1 };
        size_t expected_after_distances[] = { 2, 1, 0 };
        size_t expected_flips[] = { 2, 1, 1 };
        itty_training_step_summary_t summary;

        itty_training_history_t *history = itty_training_history_feed_model_train_one_with_optimizer (model,
                                                                                                      input,
                                                                                                      target,
                                                                                                      optimizer,
                                                                                                      NULL,
                                                                                                      8);
        assert (history != NULL);
        assert (itty_training_history_get_step_count (history) == 3);

        for (size_t i = 0; i < 3; i++) {
                assert (itty_training_history_get_step_summary (history, i, &summary));
                assert (summary.before_distance == expected_before_distances[i]);
                assert (summary.after_distance == expected_after_distances[i]);
                assert (summary.flips == expected_flips[i]);
                assert (summary.candidate_bits == expected_before_distances[i]);
        }

        itty_training_history_free (history);
        itty_training_optimizer_free (optimizer);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

static void
test_itty_training_history_summarizes_dense_optimizer_tradeoff (void)
{
        itty_feed_model_t *fixed_model = itty_feed_model_new (1, 1, 1, 1);
        itty_feed_model_t *fraction_model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *fixed_input = create_zero_input ();
        itty_bit_string_list_t *fraction_input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_training_optimizer_t *fixed_optimizer = itty_training_optimizer_new_fixed_budget (8);
        itty_training_optimizer_t *fraction_optimizer = itty_training_optimizer_new_distance_fraction_budget (1, 2, 32);
        itty_training_history_summary_t fixed_summary;
        itty_training_history_summary_t fraction_summary;

        itty_training_history_t *fixed_history = itty_training_history_feed_model_train_one_with_optimizer (fixed_model,
                                                                                                            fixed_input,
                                                                                                            target,
                                                                                                            fixed_optimizer,
                                                                                                            NULL,
                                                                                                            8);
        itty_training_history_t *fraction_history = itty_training_history_feed_model_train_one_with_optimizer (fraction_model,
                                                                                                               fraction_input,
                                                                                                               target,
                                                                                                               fraction_optimizer,
                                                                                                               NULL,
                                                                                                               8);

        assert (itty_training_history_summarize (fixed_history, &fixed_summary));
        assert (itty_training_history_summarize (fraction_history, &fraction_summary));

        assert (fixed_summary.reached_target);
        assert (fraction_summary.reached_target);
        assert (fixed_summary.steps == 4);
        assert (fraction_summary.steps == 6);
        assert (fixed_summary.total_flips == 32);
        assert (fraction_summary.total_flips == 32);
        assert_near (fixed_summary.final_mask_entropy, 1.0);
        assert_near (fraction_summary.final_mask_entropy, 1.0);
        assert_near (fixed_summary.final_activation_entropy, 1.0);
        assert_near (fraction_summary.final_activation_entropy, 1.0);

        itty_training_history_free (fixed_history);
        itty_training_history_free (fraction_history);
        itty_training_optimizer_free (fixed_optimizer);
        itty_training_optimizer_free (fraction_optimizer);
        itty_bit_string_free (target);
        itty_bit_string_list_free (fixed_input);
        itty_bit_string_list_free (fraction_input);
        itty_feed_model_free (fixed_model);
        itty_feed_model_free (fraction_model);
}

static void
test_itty_training_history_summarizes_sparse_optimizer_tradeoff (void)
{
        char *text_file = NULL;
        char *bit_string_file = NULL;
        itty_vocabulary_t *vocabulary = create_vocabulary (&text_file, &bit_string_file);
        itty_feed_model_t *fixed_model = itty_feed_model_new (1, 1, 1, 1);
        itty_feed_model_t *fraction_model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *fixed_input = create_input ();
        itty_bit_string_list_t *fraction_input = create_input ();
        itty_bit_string_t *target = itty_vocabulary_translate_to_bit_string (vocabulary, " b");
        itty_training_optimizer_t *fixed_optimizer = itty_training_optimizer_new_fixed_budget (8);
        itty_training_optimizer_t *fraction_optimizer = itty_training_optimizer_new_distance_fraction_budget (1, 2, 32);
        itty_training_history_summary_t fixed_summary;
        itty_training_history_summary_t fraction_summary;

        itty_training_history_t *fixed_history = itty_training_history_feed_model_train_one_with_optimizer (fixed_model,
                                                                                                            fixed_input,
                                                                                                            target,
                                                                                                            fixed_optimizer,
                                                                                                            NULL,
                                                                                                            8);
        itty_training_history_t *fraction_history = itty_training_history_feed_model_train_one_with_optimizer (fraction_model,
                                                                                                               fraction_input,
                                                                                                               target,
                                                                                                               fraction_optimizer,
                                                                                                               NULL,
                                                                                                               8);

        assert (itty_training_history_summarize (fixed_history, &fixed_summary));
        assert (itty_training_history_summarize (fraction_history, &fraction_summary));

        assert (fixed_summary.reached_target);
        assert (fraction_summary.reached_target);
        assert (fixed_summary.steps == 1);
        assert (fraction_summary.steps == 3);
        assert (fixed_summary.total_flips == 4);
        assert (fraction_summary.total_flips == 4);
        assert (fixed_summary.final_mask_entropy == fraction_summary.final_mask_entropy);
        assert (fixed_summary.final_activation_entropy == fraction_summary.final_activation_entropy);

        itty_training_history_free (fixed_history);
        itty_training_history_free (fraction_history);
        itty_training_optimizer_free (fixed_optimizer);
        itty_training_optimizer_free (fraction_optimizer);
        itty_bit_string_list_free (fixed_input);
        itty_bit_string_list_free (fraction_input);
        itty_feed_model_free (fixed_model);
        itty_feed_model_free (fraction_model);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

static void
test_itty_training_history_backwards_two_layer_optimizer_tradeoff (void)
{
        itty_feed_model_t *fixed_model = itty_feed_model_new (2, 1, 1, 1);
        itty_feed_model_t *fraction_model = itty_feed_model_new (2, 1, 1, 1);
        itty_bit_string_list_t *fixed_input = create_zero_input ();
        itty_bit_string_list_t *fraction_input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_training_optimizer_t *fixed_optimizer = itty_training_optimizer_new_fixed_budget (8);
        itty_training_optimizer_t *fraction_optimizer = itty_training_optimizer_new_distance_fraction_budget (1, 2, 32);
        itty_training_history_summary_t fixed_summary;
        itty_training_history_summary_t fraction_summary;

        itty_training_history_t *fixed_history = itty_training_history_feed_model_train_backwards_one_with_optimizer (fixed_model,
                                                                                                                      fixed_input,
                                                                                                                      target,
                                                                                                                      fixed_optimizer,
                                                                                                                      NULL,
                                                                                                                      64);
        itty_training_history_t *fraction_history = itty_training_history_feed_model_train_backwards_one_with_optimizer (fraction_model,
                                                                                                                         fraction_input,
                                                                                                                         target,
                                                                                                                         fraction_optimizer,
                                                                                                                         NULL,
                                                                                                                         64);

        assert (itty_training_history_summarize (fixed_history, &fixed_summary));
        assert (itty_training_history_summarize (fraction_history, &fraction_summary));
        assert (fixed_summary.reached_target);
        assert (fraction_summary.reached_target);
        assert (fixed_summary.steps == 4);
        assert (fraction_summary.steps == 6);
        assert (fixed_summary.total_flips == 48);
        assert (fraction_summary.total_flips == 48);
        assert (fixed_summary.final_distance == 0);
        assert (fraction_summary.final_distance == 0);
        assert_near (fixed_summary.final_mask_entropy, fraction_summary.final_mask_entropy);
        assert_near (fixed_summary.final_activation_entropy, 1.0);
        assert_near (fraction_summary.final_activation_entropy, 1.0);

        itty_training_history_free (fixed_history);
        itty_training_history_free (fraction_history);
        itty_training_optimizer_free (fixed_optimizer);
        itty_training_optimizer_free (fraction_optimizer);
        itty_bit_string_free (target);
        itty_bit_string_list_free (fixed_input);
        itty_bit_string_list_free (fraction_input);
        itty_feed_model_free (fixed_model);
        itty_feed_model_free (fraction_model);
}

static void
test_itty_training_history_backwards_two_layer_hidden_rotation_optimizer_tradeoff (void)
{
        itty_feed_model_t *fixed_model = itty_feed_model_new (2, 1, 1, 1);
        itty_feed_model_t *fraction_model = itty_feed_model_new (2, 1, 1, 1);
        itty_bit_string_list_t *fixed_input = create_zero_input ();
        itty_bit_string_list_t *fraction_input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_training_optimizer_t *fixed_optimizer = itty_training_optimizer_new_fixed_budget (8);
        itty_training_optimizer_t *fraction_optimizer = itty_training_optimizer_new_distance_fraction_budget (1, 2, 32);
        itty_training_history_summary_t fixed_summary;
        itty_training_history_summary_t fraction_summary;

        itty_feed_model_set_layer_rotation (fixed_model, 0, 1);
        itty_feed_model_set_layer_rotation (fraction_model, 0, 1);

        itty_training_history_t *fixed_history = itty_training_history_feed_model_train_backwards_one_with_optimizer (fixed_model,
                                                                                                                      fixed_input,
                                                                                                                      target,
                                                                                                                      fixed_optimizer,
                                                                                                                      NULL,
                                                                                                                      64);
        itty_training_history_t *fraction_history = itty_training_history_feed_model_train_backwards_one_with_optimizer (fraction_model,
                                                                                                                         fraction_input,
                                                                                                                         target,
                                                                                                                         fraction_optimizer,
                                                                                                                         NULL,
                                                                                                                         64);

        assert (itty_training_history_summarize (fixed_history, &fixed_summary));
        assert (itty_training_history_summarize (fraction_history, &fraction_summary));
        assert (fixed_summary.reached_target);
        assert (fraction_summary.reached_target);
        assert (fixed_summary.steps == 5);
        assert (fraction_summary.steps == 6);
        assert (fixed_summary.total_flips == 49);
        assert (fraction_summary.total_flips == 49);
        assert (fixed_summary.final_distance == 0);
        assert (fraction_summary.final_distance == 0);
        assert_near (fixed_summary.final_mask_entropy, fraction_summary.final_mask_entropy);
        assert_near (fixed_summary.final_activation_entropy, 1.0);
        assert_near (fraction_summary.final_activation_entropy, 1.0);

        itty_training_history_free (fixed_history);
        itty_training_history_free (fraction_history);
        itty_training_optimizer_free (fixed_optimizer);
        itty_training_optimizer_free (fraction_optimizer);
        itty_bit_string_free (target);
        itty_bit_string_list_free (fixed_input);
        itty_bit_string_list_free (fraction_input);
        itty_feed_model_free (fixed_model);
        itty_feed_model_free (fraction_model);
}

static void
test_itty_training_history_backwards_three_layer_optimizer_tradeoff (void)
{
        itty_feed_model_t *fixed_model = itty_feed_model_new (3, 1, 1, 1);
        itty_feed_model_t *fraction_model = itty_feed_model_new (3, 1, 1, 1);
        itty_bit_string_list_t *fixed_input = create_zero_input ();
        itty_bit_string_list_t *fraction_input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_training_optimizer_t *fixed_optimizer = itty_training_optimizer_new_fixed_budget (8);
        itty_training_optimizer_t *fraction_optimizer = itty_training_optimizer_new_distance_fraction_budget (1, 2, 32);
        itty_training_history_summary_t fixed_summary;
        itty_training_history_summary_t fraction_summary;

        itty_training_history_t *fixed_history = itty_training_history_feed_model_train_backwards_one_with_optimizer (fixed_model,
                                                                                                                      fixed_input,
                                                                                                                      target,
                                                                                                                      fixed_optimizer,
                                                                                                                      NULL,
                                                                                                                      128);
        itty_training_history_t *fraction_history = itty_training_history_feed_model_train_backwards_one_with_optimizer (fraction_model,
                                                                                                                         fraction_input,
                                                                                                                         target,
                                                                                                                         fraction_optimizer,
                                                                                                                         NULL,
                                                                                                                         128);

        assert (itty_training_history_summarize (fixed_history, &fixed_summary));
        assert (itty_training_history_summarize (fraction_history, &fraction_summary));
        assert (fixed_summary.reached_target);
        assert (fraction_summary.reached_target);
        assert (fixed_summary.steps == 8);
        assert (fraction_summary.steps == 9);
        assert (fixed_summary.total_flips == 104);
        assert (fraction_summary.total_flips == 96);
        assert (fixed_summary.final_distance == 0);
        assert (fraction_summary.final_distance == 0);
        assert (fraction_summary.total_flips < fixed_summary.total_flips);
        assert (fraction_summary.final_mask_entropy > fixed_summary.final_mask_entropy);
        assert_near (fixed_summary.final_activation_entropy, 1.0);
        assert_near (fraction_summary.final_activation_entropy, 1.0);

        itty_training_history_free (fixed_history);
        itty_training_history_free (fraction_history);
        itty_training_optimizer_free (fixed_optimizer);
        itty_training_optimizer_free (fraction_optimizer);
        itty_bit_string_free (target);
        itty_bit_string_list_free (fixed_input);
        itty_bit_string_list_free (fraction_input);
        itty_feed_model_free (fixed_model);
        itty_feed_model_free (fraction_model);
}

static void
test_itty_training_history_backwards_three_layer_hidden_rotation_optimizer_tradeoff (void)
{
        itty_feed_model_t *fixed_model = itty_feed_model_new (3, 1, 1, 1);
        itty_feed_model_t *fraction_model = itty_feed_model_new (3, 1, 1, 1);
        itty_bit_string_list_t *fixed_input = create_zero_input ();
        itty_bit_string_list_t *fraction_input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_training_optimizer_t *fixed_optimizer = itty_training_optimizer_new_fixed_budget (8);
        itty_training_optimizer_t *fraction_optimizer = itty_training_optimizer_new_distance_fraction_budget (1, 2, 32);
        itty_training_history_summary_t fixed_summary;
        itty_training_history_summary_t fraction_summary;

        itty_feed_model_set_layer_rotation (fixed_model, 0, 1);
        itty_feed_model_set_layer_rotation (fraction_model, 0, 1);

        itty_training_history_t *fixed_history = itty_training_history_feed_model_train_backwards_one_with_optimizer (fixed_model,
                                                                                                                      fixed_input,
                                                                                                                      target,
                                                                                                                      fixed_optimizer,
                                                                                                                      NULL,
                                                                                                                      128);
        itty_training_history_t *fraction_history = itty_training_history_feed_model_train_backwards_one_with_optimizer (fraction_model,
                                                                                                                         fraction_input,
                                                                                                                         target,
                                                                                                                         fraction_optimizer,
                                                                                                                         NULL,
                                                                                                                         128);

        assert (itty_training_history_summarize (fixed_history, &fixed_summary));
        assert (itty_training_history_summarize (fraction_history, &fraction_summary));
        assert (fixed_summary.reached_target);
        assert (fraction_summary.reached_target);
        assert (fixed_summary.steps == 8);
        assert (fraction_summary.steps == 9);
        assert (fixed_summary.total_flips == 104);
        assert (fraction_summary.total_flips == 96);
        assert (fixed_summary.final_distance == 0);
        assert (fraction_summary.final_distance == 0);
        assert (fraction_summary.total_flips < fixed_summary.total_flips);
        assert (fraction_summary.final_mask_entropy > fixed_summary.final_mask_entropy);
        assert_near (fixed_summary.final_activation_entropy, 1.0);
        assert_near (fraction_summary.final_activation_entropy, 1.0);

        itty_training_history_free (fixed_history);
        itty_training_history_free (fraction_history);
        itty_training_optimizer_free (fixed_optimizer);
        itty_training_optimizer_free (fraction_optimizer);
        itty_bit_string_free (target);
        itty_bit_string_list_free (fixed_input);
        itty_bit_string_list_free (fraction_input);
        itty_feed_model_free (fixed_model);
        itty_feed_model_free (fraction_model);
}

static void
test_itty_training_history_backwards_eight_layer_bounded_depth_stress (void)
{
        itty_feed_model_t *fixed_model = itty_feed_model_new (8, 1, 1, 1);
        itty_feed_model_t *fraction_model = itty_feed_model_new (8, 1, 1, 1);
        itty_bit_string_list_t *fixed_input = create_zero_input ();
        itty_bit_string_list_t *fraction_input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_training_optimizer_t *fixed_optimizer = itty_training_optimizer_new_fixed_budget (8);
        itty_training_optimizer_t *fraction_optimizer = itty_training_optimizer_new_distance_fraction_budget (1, 2, 32);
        itty_training_history_summary_t fixed_summary;
        itty_training_history_summary_t fraction_summary;

        itty_training_history_t *fixed_history = itty_training_history_feed_model_train_backwards_one_with_optimizer (fixed_model,
                                                                                                                      fixed_input,
                                                                                                                      target,
                                                                                                                      fixed_optimizer,
                                                                                                                      NULL,
                                                                                                                      8);
        itty_training_history_t *fraction_history = itty_training_history_feed_model_train_backwards_one_with_optimizer (fraction_model,
                                                                                                                         fraction_input,
                                                                                                                         target,
                                                                                                                         fraction_optimizer,
                                                                                                                         NULL,
                                                                                                                         8);

        assert (itty_training_history_summarize (fixed_history, &fixed_summary));
        assert (itty_training_history_summarize (fraction_history, &fraction_summary));
        assert (!fixed_summary.reached_target);
        assert (!fraction_summary.reached_target);
        assert (fixed_summary.steps == 8);
        assert (fraction_summary.steps == 8);
        assert (fixed_summary.total_flips == 368);
        assert (fraction_summary.total_flips == 640);
        assert (fixed_summary.final_distance == 32);
        assert (fraction_summary.final_distance == 32);
        assert (fraction_summary.total_flips > fixed_summary.total_flips);
        assert (fraction_summary.final_mask_entropy > fixed_summary.final_mask_entropy);

        itty_training_history_free (fixed_history);
        itty_training_history_free (fraction_history);
        itty_training_optimizer_free (fixed_optimizer);
        itty_training_optimizer_free (fraction_optimizer);
        itty_bit_string_free (target);
        itty_bit_string_list_free (fixed_input);
        itty_bit_string_list_free (fraction_input);
        itty_feed_model_free (fixed_model);
        itty_feed_model_free (fraction_model);
}

static void
test_itty_training_observation_backwards_eight_layer_reports_per_layer_work (void)
{
        itty_feed_model_t *model = itty_feed_model_new (8, 1, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST
        };
        size_t expected_candidates[] = { 0, 16, 88, 224, 488, 1008, 2040, 4096 };
        size_t expected_flips[] = { 0, 8, 8, 8, 8, 8, 8, 8 };
        itty_feed_model_train_stats_t stats;

        itty_training_observation_t *observation = itty_training_observation_feed_model_train_backwards_one (model,
                                                                                                              input,
                                                                                                              target,
                                                                                                              &options,
                                                                                                              NULL);
        assert (observation != NULL);
        assert (itty_training_observation_did_train (observation));
        assert (itty_training_observation_get_layer_summary_count (observation) == 8);
        assert (itty_training_observation_get_train_stats (observation, &stats));
        assert (stats.flips == 56);

        for (size_t layer_index = 0; layer_index < 8; layer_index++) {
                itty_training_layer_summary_t layer_summary;

                assert (itty_training_observation_get_layer_summary (observation,
                                                                     layer_index,
                                                                     &layer_summary));
                assert (layer_summary.layer_index == layer_index);
                assert (layer_summary.flips == expected_flips[layer_index]);
                assert (layer_summary.candidate_bits == expected_candidates[layer_index]);
                assert (layer_summary.largest_error == (layer_index == 0 ? 0 : 1));
                assert (layer_summary.before_mask_entropy == 0.0);
        }

        itty_training_observation_free (observation);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_observation_backwards_eight_layer_segment_fold_reports_input_work (void)
{
        itty_feed_model_t *model = itty_feed_model_new (8, 1, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        size_t expected_candidates[] = { 32, 64, 128, 256, 512, 1024, 2048, 4096 };
        itty_feed_model_train_stats_t stats;

        itty_training_observation_t *observation = itty_training_observation_feed_model_train_backwards_one (model,
                                                                                                              input,
                                                                                                              target,
                                                                                                              &options,
                                                                                                              NULL);
        assert (observation != NULL);
        assert (itty_training_observation_did_train (observation));
        assert (itty_training_observation_get_layer_summary_count (observation) == 8);
        assert (itty_training_observation_get_train_stats (observation, &stats));
        assert (stats.flips == 64);
        assert (stats.candidate_bits == 8160);
        assert (stats.largest_error == 1);

        for (size_t layer_index = 0; layer_index < 8; layer_index++) {
                itty_training_layer_summary_t layer_summary;

                assert (itty_training_observation_get_layer_summary (observation,
                                                                     layer_index,
                                                                     &layer_summary));
                assert (layer_summary.layer_index == layer_index);
                assert (layer_summary.flips == 8);
                assert (layer_summary.candidate_bits == expected_candidates[layer_index]);
                assert (layer_summary.largest_error == 1);
                assert (layer_summary.before_mask_entropy == 0.0);
        }

        itty_training_observation_free (observation);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_history_backwards_eight_layer_segment_fold_reaches_target (void)
{
        itty_feed_model_t *model = itty_feed_model_new (8, 1, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_training_history_summary_t summary;

        itty_training_history_t *history = itty_training_history_feed_model_train_backwards_one (model,
                                                                                                  input,
                                                                                                  target,
                                                                                                  &options,
                                                                                                  NULL,
                                                                                                  8);

        assert (itty_training_history_summarize (history, &summary));
        assert (summary.reached_target);
        assert (summary.steps == 2);
        assert (summary.total_flips == 128);
        assert (summary.final_distance == 0);
        assert (summary.final_mask_entropy > 0.0);
        assert (summary.final_activation_entropy > 0.0);

        itty_training_history_free (history);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_history_backwards_eight_layer_segment_fold_supports_uniform_rotation (void)
{
        itty_feed_model_t *model = itty_feed_model_new (8, 1, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_training_history_summary_t summary;

        for (size_t layer_index = 0; layer_index < 8; layer_index++)
                itty_feed_model_set_layer_rotation (model, layer_index, 1);

        itty_training_history_t *history = itty_training_history_feed_model_train_backwards_one (model,
                                                                                                  input,
                                                                                                  target,
                                                                                                  &options,
                                                                                                  NULL,
                                                                                                  8);

        assert (itty_training_history_summarize (history, &summary));
        assert (summary.reached_target);
        assert (summary.steps == 2);
        assert (summary.total_flips == 128);
        assert (summary.final_distance == 0);
        assert (summary.final_mask_entropy > 0.0);
        assert (summary.final_activation_entropy > 0.0);

        itty_training_history_free (history);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_history_backwards_eight_layer_segment_fold_supports_varied_hidden_rotations (void)
{
        itty_feed_model_t *model = itty_feed_model_new (8, 1, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 64,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_training_history_summary_t summary;

        for (size_t layer_index = 0; layer_index < 7; layer_index++)
                itty_feed_model_set_layer_rotation (model, layer_index, layer_index + 1);

        itty_training_history_t *history = itty_training_history_feed_model_train_backwards_one (model,
                                                                                                  input,
                                                                                                  target,
                                                                                                  &options,
                                                                                                  NULL,
                                                                                                  32);

        assert (itty_training_history_summarize (history, &summary));
        assert (summary.reached_target);
        assert (summary.steps == 23);
        assert (summary.total_flips == 4018);
        assert (summary.final_distance == 0);

        itty_training_history_free (history);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_history_backwards_segment_fold_trains_multi_node_model (void)
{
        itty_feed_model_t *model = itty_feed_model_new (3, 2, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_training_history_summary_t summary;

        itty_training_history_t *history = itty_training_history_feed_model_train_backwards_one (model,
                                                                                                  input,
                                                                                                  target,
                                                                                                  &options,
                                                                                                  NULL,
                                                                                                  8);

        assert (itty_training_history_summarize (history, &summary));
        assert (summary.reached_target);
        assert (summary.steps == 4);
        assert (summary.total_flips == 192);
        assert (summary.final_distance == 0);
        assert (summary.final_activation_entropy > 0.0);

        itty_training_history_free (history);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_history_backwards_segment_fold_can_split_disagreement_target (void)
{
        itty_feed_model_t *model = itty_feed_model_new (3, 2, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE,
                .backward_node_target = ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DISAGREEMENT
        };
        itty_training_history_summary_t summary;

        itty_training_history_t *history = itty_training_history_feed_model_train_backwards_one (model,
                                                                                                  input,
                                                                                                  target,
                                                                                                  &options,
                                                                                                  NULL,
                                                                                                  32);

        assert (itty_training_history_summarize (history, &summary));
        assert (summary.reached_target);
        assert (summary.steps == 28);
        assert (summary.total_flips == 352);
        assert (summary.final_distance == 0);
        assert (summary.final_mask_entropy > 0.0);
        assert (summary.final_activation_entropy > 0.0);

        itty_training_history_free (history);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_history_backwards_segment_fold_can_split_pairwise_and_segment_targets (void)
{
        itty_feed_model_t *model = itty_feed_model_new (3, 2, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE,
                .backward_node_target = ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_PAIRWISE_AND_SEGMENT
        };
        itty_training_history_summary_t summary;

        itty_training_history_t *history = itty_training_history_feed_model_train_backwards_one (model,
                                                                                                  input,
                                                                                                  target,
                                                                                                  &options,
                                                                                                  NULL,
                                                                                                  16);

        assert (itty_training_history_summarize (history, &summary));
        assert (summary.reached_target);
        assert (summary.steps == 14);
        assert (summary.total_flips == 400);
        assert (summary.final_distance == 0);
        assert (summary.final_mask_entropy > 0.0);
        assert (summary.final_activation_entropy > 0.0);

        itty_training_history_free (history);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_history_backwards_segment_fold_can_partition_segment_targets (void)
{
        itty_feed_model_t *model = itty_feed_model_new (3, 2, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE,
                .backward_node_target = ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_SEGMENT_PARTITION
        };
        itty_training_history_summary_t summary;

        itty_training_history_t *history = itty_training_history_feed_model_train_backwards_one (model,
                                                                                                  input,
                                                                                                  target,
                                                                                                  &options,
                                                                                                  NULL,
                                                                                                  8);

        assert (itty_training_history_summarize (history, &summary));
        assert (summary.reached_target);
        assert (summary.steps == 4);
        assert (summary.total_flips == 192);
        assert (summary.final_distance == 0);
        assert (summary.final_mask_entropy > 0.0);
        assert (summary.final_activation_entropy > 0.0);

        itty_training_history_free (history);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_history_backwards_segment_fold_downstream_mask_target_spends_less_on_randomized_small_shape (void)
{
        itty_feed_model_t *same_model = itty_feed_model_new (3, 2, 1, 1);
        itty_feed_model_t *downstream_model = itty_feed_model_new (3, 2, 1, 1);
        itty_bit_string_list_t *same_input = create_zero_input ();
        itty_bit_string_list_t *downstream_input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t same_options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE
        };
        itty_feed_model_train_options_t downstream_options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE,
                .backward_node_target = ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE
        };
        itty_training_history_summary_t same_summary;
        itty_training_history_summary_t downstream_summary;

        assert (itty_feed_model_randomize_masks (same_model, 7, 1, 8));
        assert (itty_feed_model_randomize_masks (downstream_model, 7, 1, 8));
        itty_feed_model_set_layer_rotation (same_model, 0, 1);
        itty_feed_model_set_layer_rotation (same_model, 1, 2);
        itty_feed_model_set_layer_rotation (downstream_model, 0, 1);
        itty_feed_model_set_layer_rotation (downstream_model, 1, 2);

        itty_training_history_t *same_history = itty_training_history_feed_model_train_backwards_one (same_model,
                                                                                                      same_input,
                                                                                                      target,
                                                                                                      &same_options,
                                                                                                      NULL,
                                                                                                      32);
        itty_training_history_t *downstream_history = itty_training_history_feed_model_train_backwards_one (downstream_model,
                                                                                                            downstream_input,
                                                                                                            target,
                                                                                                            &downstream_options,
                                                                                                            NULL,
                                                                                                            32);

        assert (itty_training_history_summarize (same_history, &same_summary));
        assert (itty_training_history_summarize (downstream_history, &downstream_summary));
        assert (same_summary.reached_target);
        assert (downstream_summary.reached_target);
        assert (same_summary.steps == 13);
        assert (downstream_summary.steps == 13);
        assert (same_summary.total_flips == 375);
        assert (downstream_summary.total_flips == 290);
        assert (downstream_summary.total_flips < same_summary.total_flips);
        assert (downstream_summary.final_mask_entropy > same_summary.final_mask_entropy);

        itty_training_history_free (same_history);
        itty_training_history_free (downstream_history);
        itty_bit_string_free (target);
        itty_bit_string_list_free (same_input);
        itty_bit_string_list_free (downstream_input);
        itty_feed_model_free (same_model);
        itty_feed_model_free (downstream_model);
}

static void
test_itty_training_history_backwards_segment_fold_gated_downstream_mask_target_reaches_simple_shape (void)
{
        itty_feed_model_t *model = itty_feed_model_new (3, 2, 1, 1);
        itty_bit_string_list_t *input = create_zero_input ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST,
                .backward_fold = ITTY_FEED_MODEL_BACKWARD_FOLD_SEGMENT_CONDENSE,
                .backward_node_target = ITTY_FEED_MODEL_BACKWARD_NODE_TARGET_DOWNSTREAM_MASK_AWARE_GATED
        };
        itty_training_history_summary_t summary;

        itty_training_history_t *history = itty_training_history_feed_model_train_backwards_one (model,
                                                                                                  input,
                                                                                                  target,
                                                                                                  &options,
                                                                                                  NULL,
                                                                                                  8);

        assert (itty_training_history_summarize (history, &summary));
        assert (summary.reached_target);
        assert (summary.steps == 6);
        assert (summary.total_flips == 216);
        assert (summary.final_distance == 0);
        assert (summary.final_mask_entropy > 0.0);

        itty_training_history_free (history);
        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

static void
test_itty_training_history_feed_model_train_one_randomized_wide_shape_spends_less (void)
{
        itty_feed_model_t *zero_model = itty_feed_model_new (4, 8, 1, 1);
        itty_feed_model_t *random_model = itty_feed_model_new (4, 8, 1, 1);
        itty_bit_string_list_t *zero_input = create_input ();
        itty_bit_string_list_t *random_input = create_input ();
        itty_bit_string_t *target = create_bit_string (0b1100);
        itty_feed_model_train_options_t options = {
                .max_flips = 8,
                .budget_policy = ITTY_FEED_MODEL_TRAIN_BUDGET_POLICY_LARGEST_ERROR_FIRST
        };
        itty_training_history_summary_t zero_summary;
        itty_training_history_summary_t random_summary;

        assert (itty_feed_model_randomize_masks (random_model, 7, 1, 8));

        itty_training_history_t *zero_history = itty_training_history_feed_model_train_one (zero_model,
                                                                                            zero_input,
                                                                                            target,
                                                                                            &options,
                                                                                            NULL,
                                                                                            32);
        itty_training_history_t *random_history = itty_training_history_feed_model_train_one (random_model,
                                                                                              random_input,
                                                                                              target,
                                                                                              &options,
                                                                                              NULL,
                                                                                              32);

        assert (itty_training_history_summarize (zero_history, &zero_summary));
        assert (itty_training_history_summarize (random_history, &random_summary));
        assert (zero_summary.reached_target);
        assert (random_summary.reached_target);
        assert (zero_summary.steps == 18);
        assert (random_summary.steps == 15);
        assert (zero_summary.total_flips == 1152);
        assert (random_summary.total_flips == 886);
        assert (random_summary.steps < zero_summary.steps);
        assert (random_summary.total_flips < zero_summary.total_flips);
        assert (random_summary.final_mask_entropy > zero_summary.final_mask_entropy);

        itty_training_history_free (zero_history);
        itty_training_history_free (random_history);
        itty_bit_string_free (target);
        itty_bit_string_list_free (zero_input);
        itty_bit_string_list_free (random_input);
        itty_feed_model_free (zero_model);
        itty_feed_model_free (random_model);
}

static void
test_itty_training_history_backwards_eight_layer_budget_sweep (void)
{
        size_t budgets[] = { 8, 16, 32 };
        size_t expected_total_flips[] = { 368, 640, 1088 };
        double previous_mask_entropy = 0.0;

        for (size_t i = 0; i < 3; i++) {
                itty_feed_model_t *model = itty_feed_model_new (8, 1, 1, 1);
                itty_bit_string_list_t *input = create_zero_input ();
                itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
                itty_training_optimizer_t *optimizer = itty_training_optimizer_new_fixed_budget (budgets[i]);
                itty_training_history_summary_t summary;

                itty_training_history_t *history = itty_training_history_feed_model_train_backwards_one_with_optimizer (model,
                                                                                                                         input,
                                                                                                                         target,
                                                                                                                         optimizer,
                                                                                                                         NULL,
                                                                                                                         8);

                assert (itty_training_history_summarize (history, &summary));
                assert (!summary.reached_target);
                assert (summary.steps == 8);
                assert (summary.total_flips == expected_total_flips[i]);
                assert (summary.final_distance == 32);
                assert (summary.final_mask_entropy > previous_mask_entropy);
                previous_mask_entropy = summary.final_mask_entropy;

                itty_training_history_free (history);
                itty_training_optimizer_free (optimizer);
                itty_bit_string_free (target);
                itty_bit_string_list_free (input);
                itty_feed_model_free (model);
        }
}

int
main (void)
{
        test_itty_training_observation_feed_model_train_one ();
        test_itty_training_observation_feed_model_train_one_dense_target ();
        test_itty_training_history_feed_model_train_one_budgeted_dense_target ();
        test_itty_training_optimizer_fixed_budget_matches_options ();
        test_itty_training_optimizer_distance_capped_budget_limits_sparse_target ();
        test_itty_training_optimizer_distance_fraction_budget_halves_dense_target ();
        test_itty_training_optimizer_distance_fraction_budget_halves_sparse_target ();
        test_itty_training_history_summarizes_dense_optimizer_tradeoff ();
        test_itty_training_history_summarizes_sparse_optimizer_tradeoff ();
        test_itty_training_history_backwards_two_layer_optimizer_tradeoff ();
        test_itty_training_history_backwards_two_layer_hidden_rotation_optimizer_tradeoff ();
        test_itty_training_history_backwards_three_layer_optimizer_tradeoff ();
        test_itty_training_history_backwards_three_layer_hidden_rotation_optimizer_tradeoff ();
        test_itty_training_history_backwards_eight_layer_bounded_depth_stress ();
        test_itty_training_observation_backwards_eight_layer_reports_per_layer_work ();
        test_itty_training_observation_backwards_eight_layer_segment_fold_reports_input_work ();
        test_itty_training_history_backwards_eight_layer_segment_fold_reaches_target ();
        test_itty_training_history_backwards_eight_layer_segment_fold_supports_uniform_rotation ();
        test_itty_training_history_backwards_eight_layer_segment_fold_supports_varied_hidden_rotations ();
        test_itty_training_history_backwards_segment_fold_trains_multi_node_model ();
        test_itty_training_history_backwards_segment_fold_can_split_disagreement_target ();
        test_itty_training_history_backwards_segment_fold_can_split_pairwise_and_segment_targets ();
        test_itty_training_history_backwards_segment_fold_can_partition_segment_targets ();
        test_itty_training_history_backwards_segment_fold_downstream_mask_target_spends_less_on_randomized_small_shape ();
        test_itty_training_history_backwards_segment_fold_gated_downstream_mask_target_reaches_simple_shape ();
        test_itty_training_history_feed_model_train_one_randomized_wide_shape_spends_less ();
        test_itty_training_history_backwards_eight_layer_budget_sweep ();
        printf ("All itty-training-observation tests passed.\n");
        return 0;
}
