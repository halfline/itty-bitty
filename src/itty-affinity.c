#include "itty-affinity.h"
#include "itty-exec-buffer.h"
#include "itty-position.h"
#include <stdlib.h>
#include <string.h>

#define ITTY_AFFINITY_STAGE_MATCH "affinity match"
#define ITTY_AFFINITY_STAGE_SCORE "affinity score"
#define ITTY_AFFINITY_STAGE_CAUSAL_MASK "affinity causal mask"
#define ITTY_AFFINITY_STAGE_OUTPUT "affinity output"

#define ITTY_AFFINITY_BUFFER_PROBE "affinity probe"
#define ITTY_AFFINITY_BUFFER_SCORES "affinity scores"
#define ITTY_AFFINITY_BUFFER_TRAIT "affinity trait"
#define ITTY_AFFINITY_BUFFER_MATCH "affinity match"
#define ITTY_AFFINITY_BUFFER_IMPRINT "affinity imprint"
#define ITTY_AFFINITY_BUFFER_VOTES "affinity votes"
#define ITTY_AFFINITY_BUFFER_CAUSAL_CLEAR_START "affinity causal clear start"
#define ITTY_AFFINITY_BUFFER_CAUSAL_CLEAR_COUNT "affinity causal clear count"
#define ITTY_AFFINITY_BUFFER_OUTPUT "affinity output"

struct itty_affinity_t {
        itty_bit_string_list_t *traits;
        itty_bit_string_list_t *imprints;
};

struct itty_affinity_plan_t {
        itty_affinity_t          *affinity;
        itty_exec_buffer_t       *exec_buffer;
        size_t                    count;
        size_t                    probe_word_count;
        size_t                    score_bit_length;
        size_t                    output_word_count;
        itty_bit_string_t        *probe_placeholder;
        itty_bit_string_t        *output_placeholder;
        size_t                   *scores;
        size_t                   *votes;
        size_t                    causal_clear_start;
        size_t                    causal_clear_count;
        itty_exec_buffer_id_t     probe_buffer;
        itty_exec_buffer_id_t     output_buffer;
        size_t                    match_stage;
        size_t                    score_stage;
        size_t                    causal_mask_stage;
        size_t                    output_stage;
        itty_exec_buffer_slice_t *match_slices;
};

typedef struct {
        itty_affinity_t              *affinity;
        itty_bit_string_t            *probe;
        itty_affinity_probe_options_t options;
        itty_bit_string_t           **output;
} itty_affinity_probe_task_data_t;

static void *
itty_affinity_probe_task (void *data)
{
        itty_affinity_probe_task_data_t *task_data = data;
        *task_data->output = itty_affinity_probe_with_options (task_data->affinity,
                                                               task_data->probe,
                                                               &task_data->options);
        return *task_data->output;
}

itty_affinity_t *
itty_affinity_new (itty_bit_string_list_t *traits,
                   itty_bit_string_list_t *imprints)
{
        if (!traits || !imprints ||
            itty_bit_string_list_get_length (traits) != itty_bit_string_list_get_length (imprints))
                return NULL;

        itty_affinity_t *affinity = malloc (sizeof (itty_affinity_t));
        if (!affinity)
                return NULL;

        affinity->traits = traits;
        affinity->imprints = imprints;

        return affinity;
}

void
itty_affinity_free (itty_affinity_t *affinity)
{
        free (affinity);
}

static size_t
max_size (size_t a,
          size_t b)
{
        return a > b ? a : b;
}

static size_t
word_count_for_bit_length (size_t bit_length)
{
        return (bit_length + ITTY_BIT_STRING_WORD_SIZE_IN_BITS - 1) / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
}

static size_t
bit_string_score_bit_length (itty_bit_string_t *bit_string)
{
        size_t bit_length = itty_bit_string_get_length (bit_string);
        if (bit_length != 0)
                return bit_length;

        return itty_bit_string_get_bit_capacity (bit_string);
}

static bool
build_affinity_scoring_stages (itty_exec_buffer_t       *exec_buffer,
                               itty_affinity_t          *affinity,
                               itty_exec_buffer_id_t     probe_buffer,
                               size_t                    probe_word_count,
                               size_t                    score_bit_length,
                               size_t                    count,
                               itty_exec_buffer_id_t     scores_buffer,
                               itty_exec_buffer_slice_t *match_slices)
{
        size_t empty_word = 0;

        itty_exec_buffer_begin_named_stage (exec_buffer, ITTY_AFFINITY_STAGE_MATCH);
        for (size_t i = 0; i < count; i++) {
                itty_bit_string_t *trait = itty_bit_string_list_fetch (affinity->traits, i);
                size_t trait_word_count = itty_bit_string_get_number_of_words (trait);
                size_t *trait_words = itty_bit_string_get_words (trait);
                if (!trait_words)
                        trait_words = &empty_word;

                size_t match_bit_length = score_bit_length;
                size_t match_word_count = max_size (probe_word_count, trait_word_count);
                if (match_word_count == 0)
                        continue;

                itty_exec_buffer_id_t trait_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                      trait_words,
                                                                                      trait_word_count,
                                                                                      ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                      ITTY_AFFINITY_BUFFER_TRAIT);
                itty_exec_buffer_id_t match_buffer = itty_exec_buffer_allocate_words (exec_buffer,
                                                                                      match_word_count,
                                                                                      ITTY_AFFINITY_BUFFER_MATCH);
                if (trait_buffer == ITTY_EXEC_BUFFER_INVALID_ID ||
                    match_buffer == ITTY_EXEC_BUFFER_INVALID_ID)
                        return false;

                match_slices[i] = itty_exec_buffer_get_bit_slice (match_buffer, 0, match_word_count, match_bit_length);
                if (!itty_exec_buffer_add_xnor (exec_buffer,
                                                match_slices[i],
                                                itty_exec_buffer_get_word_slice (probe_buffer, 0, probe_word_count),
                                                itty_exec_buffer_get_word_slice (trait_buffer, 0, trait_word_count)))
                        return false;
        }

        itty_exec_buffer_begin_named_stage (exec_buffer, ITTY_AFFINITY_STAGE_SCORE);
        for (size_t i = 0; i < count; i++) {
                if (match_slices[i].buffer_id == ITTY_EXEC_BUFFER_INVALID_ID)
                        continue;

                if (!itty_exec_buffer_add_popcount (exec_buffer,
                                                    itty_exec_buffer_get_word_slice (scores_buffer, i, 1),
                                                    match_slices[i]))
                        return false;
        }

        return true;
}

static bool
build_affinity_output_stage (itty_exec_buffer_t *exec_buffer,
                             itty_affinity_t    *affinity,
                             itty_exec_buffer_id_t output_buffer,
                             size_t             *votes,
                             size_t              count)
{
        size_t bit_length = itty_bit_string_list_get_bit_length (affinity->imprints);
        size_t number_of_words = (bit_length + ITTY_BIT_STRING_WORD_SIZE_IN_BITS - 1) / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        if (number_of_words == 0)
                return true;

        itty_exec_buffer_slice_t *imprint_slices = calloc (count, sizeof (itty_exec_buffer_slice_t));
        if (!imprint_slices)
                return false;

        size_t empty_word = 0;
        for (size_t i = 0; i < count; i++) {
                itty_bit_string_t *imprint = itty_bit_string_list_fetch (affinity->imprints, i);
                size_t number_of_imprint_words = itty_bit_string_get_number_of_words (imprint);
                size_t *imprint_words = itty_bit_string_get_words (imprint);
                if (!imprint_words)
                        imprint_words = &empty_word;

                itty_exec_buffer_id_t imprint_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                        imprint_words,
                                                                                        number_of_imprint_words,
                                                                                        ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                        ITTY_AFFINITY_BUFFER_IMPRINT);
                if (imprint_buffer == ITTY_EXEC_BUFFER_INVALID_ID) {
                        free (imprint_slices);
                        return false;
                }

                imprint_slices[i] = itty_exec_buffer_get_word_slice (imprint_buffer, 0, number_of_imprint_words);
        }

        itty_exec_buffer_id_t votes_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              votes,
                                                                              count,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                              ITTY_AFFINITY_BUFFER_VOTES);
        itty_exec_buffer_begin_named_stage (exec_buffer, ITTY_AFFINITY_STAGE_OUTPUT);
        if (votes_buffer == ITTY_EXEC_BUFFER_INVALID_ID ||
            output_buffer == ITTY_EXEC_BUFFER_INVALID_ID ||
            !itty_exec_buffer_add_weighted_condense (exec_buffer,
                                                     itty_exec_buffer_get_bit_slice (output_buffer, 0, number_of_words, bit_length),
                                                     imprint_slices,
                                                     itty_exec_buffer_get_word_slice (votes_buffer, 0, count),
                                                     count)) {
                free (imprint_slices);
                return false;
        }

        free (imprint_slices);

        return true;
}

static bool
build_affinity_causal_clear_stage (itty_exec_buffer_t *exec_buffer,
                                   itty_exec_buffer_id_t scores_buffer,
                                   size_t count,
                                   size_t *causal_clear_start,
                                   size_t *causal_clear_count)
{
        itty_exec_buffer_id_t start_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              causal_clear_start,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              ITTY_AFFINITY_BUFFER_CAUSAL_CLEAR_START);
        itty_exec_buffer_id_t count_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              causal_clear_count,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              ITTY_AFFINITY_BUFFER_CAUSAL_CLEAR_COUNT);
        itty_exec_buffer_begin_named_stage (exec_buffer, ITTY_AFFINITY_STAGE_CAUSAL_MASK);
        return start_buffer != ITTY_EXEC_BUFFER_INVALID_ID &&
               count_buffer != ITTY_EXEC_BUFFER_INVALID_ID &&
               itty_exec_buffer_add_clear_array_range (exec_buffer,
                                                       itty_exec_buffer_get_array (scores_buffer, 0, count),
                                                       itty_exec_buffer_get_value (start_buffer, 0),
                                                       itty_exec_buffer_get_value (count_buffer, 0));
}

static bool
cache_affinity_plan_stages (itty_affinity_plan_t *plan)
{
        if (!itty_exec_buffer_find_stage (plan->exec_buffer,
                                          ITTY_AFFINITY_STAGE_MATCH,
                                          &plan->match_stage) ||
            !itty_exec_buffer_find_stage (plan->exec_buffer,
                                          ITTY_AFFINITY_STAGE_SCORE,
                                          &plan->score_stage) ||
            !itty_exec_buffer_find_stage (plan->exec_buffer,
                                          ITTY_AFFINITY_STAGE_CAUSAL_MASK,
                                          &plan->causal_mask_stage))
                return false;

        plan->output_stage = (size_t) -1;
        if (plan->output_word_count > 0 &&
            !itty_exec_buffer_find_stage (plan->exec_buffer,
                                          ITTY_AFFINITY_STAGE_OUTPUT,
                                          &plan->output_stage))
                return false;

        return true;
}

itty_affinity_plan_t *
itty_affinity_plan_new (itty_affinity_t *affinity,
                        size_t           score_bit_length)
{
        if (!affinity || score_bit_length == 0)
                return NULL;

        size_t count = itty_bit_string_list_get_length (affinity->traits);
        if (count == 0)
                return NULL;

        itty_affinity_plan_t *plan = calloc (1, sizeof (itty_affinity_plan_t));
        if (!plan)
                return NULL;

        plan->affinity = affinity;
        plan->count = count;
        plan->score_bit_length = score_bit_length;
        size_t max_trait_word_count = 0;
        for (size_t i = 0; i < count; i++) {
                size_t trait_word_count = itty_bit_string_get_number_of_words (itty_bit_string_list_fetch (affinity->traits, i));
                if (trait_word_count > max_trait_word_count)
                        max_trait_word_count = trait_word_count;
        }
        plan->probe_word_count = max_size (word_count_for_bit_length (score_bit_length), max_trait_word_count);
        size_t output_bit_length = itty_bit_string_list_get_bit_length (affinity->imprints);
        plan->output_word_count = word_count_for_bit_length (output_bit_length);

        plan->exec_buffer = itty_exec_buffer_new ();
        plan->probe_placeholder = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        plan->output_placeholder = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        plan->scores = calloc (count, sizeof (size_t));
        plan->votes = calloc (count, sizeof (size_t));
        plan->match_slices = calloc (count, sizeof (itty_exec_buffer_slice_t));

        if (!plan->exec_buffer || !plan->probe_placeholder || !plan->output_placeholder ||
            !plan->scores || !plan->votes || !plan->match_slices) {
                itty_affinity_plan_free (plan);
                return NULL;
        }

        itty_bit_string_append_zeros (plan->probe_placeholder, plan->probe_word_count);
        itty_bit_string_append_zeros (plan->output_placeholder, plan->output_word_count);
        for (size_t i = 0; i < count; i++)
                plan->match_slices[i] = itty_exec_buffer_get_word_slice (ITTY_EXEC_BUFFER_INVALID_ID, 0, 0);

        plan->probe_buffer = itty_exec_buffer_register_bit_string (plan->exec_buffer,
                                                                   plan->probe_placeholder,
                                                                   ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                   ITTY_AFFINITY_BUFFER_PROBE);
        itty_exec_buffer_id_t scores_buffer = itty_exec_buffer_register_words (plan->exec_buffer,
                                                                               plan->scores,
                                                                               count,
                                                                               ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                               ITTY_AFFINITY_BUFFER_SCORES);
        if (plan->probe_buffer == ITTY_EXEC_BUFFER_INVALID_ID ||
            scores_buffer == ITTY_EXEC_BUFFER_INVALID_ID ||
            !build_affinity_scoring_stages (plan->exec_buffer,
                                            affinity,
                                            plan->probe_buffer,
                                            plan->probe_word_count,
                                            score_bit_length,
                                            count,
                                            scores_buffer,
                                            plan->match_slices) ||
            !build_affinity_causal_clear_stage (plan->exec_buffer,
                                                scores_buffer,
                                                count,
                                                &plan->causal_clear_start,
                                                &plan->causal_clear_count)) {
                itty_affinity_plan_free (plan);
                return NULL;
        }

        plan->output_buffer = itty_exec_buffer_register_bit_string (plan->exec_buffer,
                                                                    plan->output_placeholder,
                                                                    ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                    ITTY_AFFINITY_BUFFER_OUTPUT);
        if (plan->output_buffer == ITTY_EXEC_BUFFER_INVALID_ID ||
            !build_affinity_output_stage (plan->exec_buffer,
                                          affinity,
                                          plan->output_buffer,
                                          plan->votes,
                                          count) ||
            !cache_affinity_plan_stages (plan)) {
                itty_affinity_plan_free (plan);
                return NULL;
        }

        return plan;
}

void
itty_affinity_plan_free (itty_affinity_plan_t *plan)
{
        if (!plan)
                return;

        itty_exec_buffer_free (plan->exec_buffer);
        itty_bit_string_free (plan->output_placeholder);
        itty_bit_string_free (plan->probe_placeholder);
        free (plan->match_slices);
        free (plan->votes);
        free (plan->scores);
        free (plan);
}

char *
itty_affinity_plan_present (itty_affinity_plan_t *plan)
{
        if (!plan)
                return NULL;

        return itty_exec_buffer_present (plan->exec_buffer);
}

itty_bit_string_t *
itty_affinity_plan_probe (itty_affinity_plan_t                 *plan,
                          itty_bit_string_t                   *probe,
                          itty_affinity_probe_options_t const *options)
{
        if (!plan || !probe ||
            bit_string_score_bit_length (probe) < plan->score_bit_length ||
            itty_bit_string_get_number_of_words (probe) > plan->probe_word_count)
                return NULL;

        itty_bit_string_t *output = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        if (!output)
                return NULL;

        itty_bit_string_append_zeros (output, plan->output_word_count);
        memset (plan->scores, 0, plan->count * sizeof (size_t));
        memset (plan->votes, 0, plan->count * sizeof (size_t));
        plan->causal_clear_start = 0;
        plan->causal_clear_count = 0;
        if (options && options->causal) {
                plan->causal_clear_start = options->probe_index + 1;
                plan->causal_clear_count = plan->count;
        }

        size_t *probe_placeholder_words = itty_bit_string_get_words (plan->probe_placeholder);
        size_t *probe_words = itty_bit_string_get_words (probe);
        memset (probe_placeholder_words, 0, plan->probe_word_count * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
        if (probe_words)
                memcpy (probe_placeholder_words,
                        probe_words,
                        itty_bit_string_get_number_of_words (probe) * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);

        if (!itty_exec_buffer_rebind_bit_string (plan->exec_buffer, plan->output_buffer, output) ||
            !itty_exec_buffer_run_stage (plan->exec_buffer, plan->match_stage) ||
            !itty_exec_buffer_run_stage (plan->exec_buffer, plan->score_stage) ||
            !itty_exec_buffer_run_stage (plan->exec_buffer, plan->causal_mask_stage)) {
                itty_bit_string_free (output);
                return NULL;
        }

        for (size_t i = 0; i < plan->count; i++) {
                if (options && options->causal && i > options->probe_index)
                        continue;

                if (options) {
                        plan->scores[i] += itty_position_locality_bonus (options->probe_index, i, options->locality_window);
                        plan->scores[i] += options->gray_position_weight *
                                itty_position_gray_similarity (options->probe_index, i, options->gray_position_bits);
                }
        }

        size_t total_votes = options ? options->total_votes : 0;
        if (!itty_popcount_allocate_votes (plan->scores, plan->count, total_votes, plan->votes)) {
                itty_bit_string_free (output);
                return NULL;
        }

        if (plan->output_word_count > 0 &&
            !itty_exec_buffer_run_stage (plan->exec_buffer, plan->output_stage)) {
                itty_bit_string_free (output);
                return NULL;
        }

        return output;
}

itty_bit_string_t *
itty_affinity_probe (itty_affinity_t *affinity,
                     itty_bit_string_t  *probe,
                     size_t              total_votes)
{
        itty_affinity_probe_options_t options = {
                .total_votes = total_votes
        };

        return itty_affinity_probe_with_options (affinity, probe, &options);
}

itty_bit_string_t *
itty_affinity_probe_at (itty_affinity_t *affinity,
                        itty_bit_string_t  *probe,
                        size_t              probe_index,
                        size_t              total_votes,
                        size_t              locality_window)
{
        itty_affinity_probe_options_t options = {
                .total_votes = total_votes,
                .probe_index = probe_index,
                .locality_window = locality_window
        };

        return itty_affinity_probe_with_options (affinity, probe, &options);
}

itty_bit_string_t *
itty_affinity_probe_with_options (itty_affinity_t                     *affinity,
                                  itty_bit_string_t                  *probe,
                                  itty_affinity_probe_options_t const *options)
{
        if (!affinity || !probe)
                return NULL;

        size_t count = itty_bit_string_list_get_length (affinity->traits);
        if (count == 0)
                return NULL;

        size_t score_bit_length = options && options->score_bit_length != 0 ? options->score_bit_length : bit_string_score_bit_length (probe);
        itty_affinity_plan_t *plan = itty_affinity_plan_new (affinity, score_bit_length);
        if (!plan)
                return NULL;

        itty_bit_string_t *output = itty_affinity_plan_probe (plan, probe, options);
        itty_affinity_plan_free (plan);

        return output;
}

itty_bit_string_list_t *
itty_affinity_probe_list (itty_affinity_t                     *affinity,
                          itty_bit_string_list_t              *probes,
                          itty_affinity_probe_options_t const *base_options)
{
        if (!affinity || !probes)
                return NULL;

        itty_bit_string_list_t *outputs = itty_bit_string_list_new ();
        if (!outputs)
                return NULL;

        size_t probe_count = itty_bit_string_list_get_length (probes);
        itty_affinity_plan_t *plan = NULL;
        size_t plan_score_bit_length = 0;

        for (size_t i = 0; i < probe_count; i++) {
                itty_affinity_probe_options_t options = { 0 };
                if (base_options)
                        options = *base_options;
                options.probe_index = i;

                itty_bit_string_t *probe = itty_bit_string_list_fetch (probes, i);
                size_t score_bit_length = options.score_bit_length != 0 ? options.score_bit_length : bit_string_score_bit_length (probe);
                if (!plan || score_bit_length != plan_score_bit_length) {
                        itty_affinity_plan_free (plan);
                        plan = itty_affinity_plan_new (affinity, score_bit_length);
                        plan_score_bit_length = score_bit_length;
                }

                itty_bit_string_t *output = itty_affinity_plan_probe (plan, probe, &options);
                if (!output) {
                        itty_affinity_plan_free (plan);
                        itty_bit_string_list_free (outputs);
                        return NULL;
                }

                itty_bit_string_list_append (outputs, output);
        }
        itty_affinity_plan_free (plan);

        return outputs;
}

itty_bit_string_list_t *
itty_affinity_probe_list_with_manager (itty_affinity_t                     *affinity,
                                       itty_bit_string_list_t              *probes,
                                       itty_affinity_probe_options_t const *base_options,
                                       itty_manager_t                      *manager)
{
        if (!manager)
                return itty_affinity_probe_list (affinity, probes, base_options);

        if (!affinity || !probes)
                return NULL;

        size_t probe_count = itty_bit_string_list_get_length (probes);
        itty_bit_string_t **outputs = calloc (probe_count, sizeof (itty_bit_string_t *));
        itty_affinity_probe_task_data_t *task_data = calloc (probe_count, sizeof (itty_affinity_probe_task_data_t));
        itty_task_group_t *task_group = itty_manager_create_task_group (manager);
        if (!outputs || !task_data || !task_group) {
                free (outputs);
                free (task_data);
                itty_manager_free_task_group (task_group);
                return NULL;
        }

        bool submitted_all = true;
        for (size_t i = 0; i < probe_count; i++) {
                task_data[i].affinity = affinity;
                task_data[i].probe = itty_bit_string_list_fetch (probes, i);
                if (base_options)
                        task_data[i].options = *base_options;
                task_data[i].options.probe_index = i;
                task_data[i].output = &outputs[i];

                if (!itty_manager_task_group_submit (task_group, itty_affinity_probe_task, &task_data[i])) {
                        submitted_all = false;
                        break;
                }
        }

        itty_manager_wait_for_task_group (task_group);
        itty_manager_free_task_group (task_group);

        itty_bit_string_list_t *output_list = NULL;
        if (submitted_all)
                output_list = itty_bit_string_list_new ();

        for (size_t i = 0; i < probe_count; i++) {
                if (!submitted_all || !outputs[i]) {
                        for (size_t j = 0; j < probe_count; j++)
                                itty_bit_string_free (outputs[j]);
                        itty_bit_string_list_free (output_list);
                        free (outputs);
                        free (task_data);
                        return NULL;
                }

                itty_bit_string_list_append (output_list, outputs[i]);
                outputs[i] = NULL;
        }

        free (outputs);
        free (task_data);

        return output_list;
}
