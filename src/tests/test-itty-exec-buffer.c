#include "itty-bit-string.h"
#include "itty-exec-buffer.h"
#include "itty-manager.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static itty_bit_string_t *
create_bit_string (size_t word)
{
        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        itty_bit_string_append_word (bit_string, word);
        return bit_string;
}

static void
test_itty_exec_buffer_feed_commands (void)
{
        itty_bit_string_t *input_a = create_bit_string (0b1111);
        itty_bit_string_t *input_b = create_bit_string (0b1111);
        itty_bit_string_t *mask_a = create_bit_string (0);
        itty_bit_string_t *mask_b = create_bit_string (0);
        itty_bit_string_t *output = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        itty_bit_string_append_zeros (output, 2);

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();
        itty_manager_t *manager = itty_manager_new ();

        itty_exec_buffer_id_t input_a_buffer = itty_exec_buffer_register_words (exec_buffer, itty_bit_string_get_words (input_a), 1, ITTY_EXEC_BUFFER_ACCESS_READ_ONLY, "input_a");
        itty_exec_buffer_id_t input_b_buffer = itty_exec_buffer_register_words (exec_buffer, itty_bit_string_get_words (input_b), 1, ITTY_EXEC_BUFFER_ACCESS_READ_ONLY, "input_b");
        itty_exec_buffer_id_t mask_a_buffer = itty_exec_buffer_register_words (exec_buffer, itty_bit_string_get_words (mask_a), 1, ITTY_EXEC_BUFFER_ACCESS_READ_ONLY, "mask_a");
        itty_exec_buffer_id_t mask_b_buffer = itty_exec_buffer_register_words (exec_buffer, itty_bit_string_get_words (mask_b), 1, ITTY_EXEC_BUFFER_ACCESS_READ_ONLY, "mask_b");
        itty_exec_buffer_id_t xor_a_buffer = itty_exec_buffer_allocate_words (exec_buffer, 1, "xor_a");
        itty_exec_buffer_id_t xor_b_buffer = itty_exec_buffer_allocate_words (exec_buffer, 1, "xor_b");
        itty_exec_buffer_id_t condensed_buffer = itty_exec_buffer_allocate_words (exec_buffer, 1, "condensed");
        itty_exec_buffer_id_t output_buffer = itty_exec_buffer_register_words (exec_buffer, itty_bit_string_get_words (output), 2, ITTY_EXEC_BUFFER_ACCESS_READ_WRITE, "output");

        itty_exec_buffer_slice_t xor_slices[2] = {
                itty_exec_buffer_get_word_slice (xor_a_buffer, 0, 1),
                itty_exec_buffer_get_word_slice (xor_b_buffer, 0, 1)
        };

        itty_exec_buffer_begin_stage (exec_buffer);
        assert (itty_exec_buffer_add_xor (exec_buffer,
                                          xor_slices[0],
                                          itty_exec_buffer_get_word_slice (input_a_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (mask_a_buffer, 0, 1)));
        assert (itty_exec_buffer_add_xor (exec_buffer,
                                          xor_slices[1],
                                          itty_exec_buffer_get_word_slice (input_b_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (mask_b_buffer, 0, 1)));

        itty_exec_buffer_begin_stage (exec_buffer);
        assert (itty_exec_buffer_add_condense (exec_buffer,
                                               itty_exec_buffer_get_word_slice (condensed_buffer, 0, 1),
                                               xor_slices,
                                               2));

        itty_exec_buffer_begin_stage (exec_buffer);
        assert (itty_exec_buffer_add_double (exec_buffer,
                                             itty_exec_buffer_get_word_slice (output_buffer, 0, 2),
                                             itty_exec_buffer_get_word_slice (condensed_buffer, 0, 1)));

        assert (itty_exec_buffer_run_with_manager (exec_buffer, manager));

        char *representation = itty_bit_string_present (output, ITTY_BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
        assert (representation != NULL);
        assert (strcmp (representation, "000000000000000f000000000000000f") == 0);
        free (representation);

        itty_exec_buffer_free (exec_buffer);
        itty_manager_free (manager);
        itty_bit_string_free (output);
        itty_bit_string_free (mask_b);
        itty_bit_string_free (mask_a);
        itty_bit_string_free (input_b);
        itty_bit_string_free (input_a);
}

static void
test_itty_exec_buffer_invalidates_registered_bit_string (void)
{
        itty_bit_string_t *input = create_bit_string (~0UL);
        itty_bit_string_t *mask = create_bit_string (0);
        itty_bit_string_t *output = create_bit_string (0);

        assert (itty_bit_string_get_pop_count (output) == 0);

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();
        itty_exec_buffer_id_t input_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              itty_bit_string_get_words (input),
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "input");
        itty_exec_buffer_id_t mask_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                             itty_bit_string_get_words (mask),
                                                                             1,
                                                                             ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                             "mask");
        itty_exec_buffer_id_t output_buffer = itty_exec_buffer_register_bit_string (exec_buffer,
                                                                                   output,
                                                                                   ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                                   "output");

        assert (itty_exec_buffer_add_xor (exec_buffer,
                                          itty_exec_buffer_get_word_slice (output_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (input_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (mask_buffer, 0, 1)));
        assert (itty_exec_buffer_run (exec_buffer));
        assert (itty_bit_string_get_pop_count (output) == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);

        itty_exec_buffer_free (exec_buffer);
        itty_bit_string_free (output);
        itty_bit_string_free (mask);
        itty_bit_string_free (input);
}

static void
test_itty_exec_buffer_attention_scores (void)
{
        itty_bit_string_t *query = create_bit_string (0b11110000);
        itty_bit_string_t *key_exact = create_bit_string (0b11110000);
        itty_bit_string_t *key_partial = create_bit_string (0b11111111);
        itty_bit_string_t *key_inverse = create_bit_string (0b00001111);
        size_t scores[3] = { 0, 0, 0 };

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();
        itty_manager_t *manager = itty_manager_new ();

        itty_exec_buffer_id_t query_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              itty_bit_string_get_words (query),
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "query");
        itty_exec_buffer_id_t key_exact_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                  itty_bit_string_get_words (key_exact),
                                                                                  1,
                                                                                  ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                  "key_exact");
        itty_exec_buffer_id_t key_partial_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                    itty_bit_string_get_words (key_partial),
                                                                                    1,
                                                                                    ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                    "key_partial");
        itty_exec_buffer_id_t key_inverse_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                    itty_bit_string_get_words (key_inverse),
                                                                                    1,
                                                                                    ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                    "key_inverse");
        itty_exec_buffer_id_t match_exact_buffer = itty_exec_buffer_allocate_words (exec_buffer, 1, "match_exact");
        itty_exec_buffer_id_t match_partial_buffer = itty_exec_buffer_allocate_words (exec_buffer, 1, "match_partial");
        itty_exec_buffer_id_t match_inverse_buffer = itty_exec_buffer_allocate_words (exec_buffer, 1, "match_inverse");
        itty_exec_buffer_id_t scores_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                               scores,
                                                                               3,
                                                                               ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                               "scores");

        itty_exec_buffer_begin_stage (exec_buffer);
        assert (itty_exec_buffer_add_xnor (exec_buffer,
                                           itty_exec_buffer_get_word_slice (match_exact_buffer, 0, 1),
                                           itty_exec_buffer_get_word_slice (query_buffer, 0, 1),
                                           itty_exec_buffer_get_word_slice (key_exact_buffer, 0, 1)));
        assert (itty_exec_buffer_add_xnor (exec_buffer,
                                           itty_exec_buffer_get_word_slice (match_partial_buffer, 0, 1),
                                           itty_exec_buffer_get_word_slice (query_buffer, 0, 1),
                                           itty_exec_buffer_get_word_slice (key_partial_buffer, 0, 1)));
        assert (itty_exec_buffer_add_xnor (exec_buffer,
                                           itty_exec_buffer_get_word_slice (match_inverse_buffer, 0, 1),
                                           itty_exec_buffer_get_word_slice (query_buffer, 0, 1),
                                           itty_exec_buffer_get_word_slice (key_inverse_buffer, 0, 1)));

        itty_exec_buffer_begin_stage (exec_buffer);
        assert (itty_exec_buffer_add_popcount (exec_buffer,
                                               itty_exec_buffer_get_word_slice (scores_buffer, 0, 1),
                                               itty_exec_buffer_get_word_slice (match_exact_buffer, 0, 1)));
        assert (itty_exec_buffer_add_popcount (exec_buffer,
                                               itty_exec_buffer_get_word_slice (scores_buffer, 1, 1),
                                               itty_exec_buffer_get_word_slice (match_partial_buffer, 0, 1)));
        assert (itty_exec_buffer_add_popcount (exec_buffer,
                                               itty_exec_buffer_get_word_slice (scores_buffer, 2, 1),
                                               itty_exec_buffer_get_word_slice (match_inverse_buffer, 0, 1)));

        assert (itty_exec_buffer_run_with_manager (exec_buffer, manager));
        assert (scores[0] == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert (scores[1] == ITTY_BIT_STRING_WORD_SIZE_IN_BITS - 4);
        assert (scores[2] == ITTY_BIT_STRING_WORD_SIZE_IN_BITS - 8);
        assert (scores[0] > scores[1]);
        assert (scores[1] > scores[2]);

        itty_exec_buffer_free (exec_buffer);
        itty_manager_free (manager);
        itty_bit_string_free (key_inverse);
        itty_bit_string_free (key_partial);
        itty_bit_string_free (key_exact);
        itty_bit_string_free (query);
}

static void
test_itty_exec_buffer_popcount_respects_bit_slice_length (void)
{
        size_t input = ~0UL;
        size_t score = 0;

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();

        itty_exec_buffer_id_t input_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              &input,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "input");
        itty_exec_buffer_id_t score_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              &score,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                              "score");

        assert (itty_exec_buffer_add_popcount (exec_buffer,
                                               itty_exec_buffer_get_word_slice (score_buffer, 0, 1),
                                               itty_exec_buffer_get_bit_slice (input_buffer, 0, 1, 4)));
        assert (itty_exec_buffer_run (exec_buffer));
        assert (score == 4);

        itty_exec_buffer_free (exec_buffer);
}

static void
test_itty_exec_buffer_weighted_condense (void)
{
        itty_bit_string_t *imprint_a = create_bit_string (0b1010 | (1UL << 8));
        itty_bit_string_t *imprint_b = create_bit_string (0b1100 | (1UL << 8));
        itty_bit_string_t *imprint_c = create_bit_string (0b0011 | (1UL << 8));
        itty_bit_string_t *output = create_bit_string (~0UL);
        size_t votes[3] = { 1, 5, 1 };

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();

        itty_exec_buffer_id_t imprint_a_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                  itty_bit_string_get_words (imprint_a),
                                                                                  1,
                                                                                  ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                  "imprint_a");
        itty_exec_buffer_id_t imprint_b_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                  itty_bit_string_get_words (imprint_b),
                                                                                  1,
                                                                                  ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                  "imprint_b");
        itty_exec_buffer_id_t imprint_c_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                  itty_bit_string_get_words (imprint_c),
                                                                                  1,
                                                                                  ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                  "imprint_c");
        itty_exec_buffer_id_t votes_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              votes,
                                                                              3,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "votes");
        itty_exec_buffer_id_t output_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                               itty_bit_string_get_words (output),
                                                                               1,
                                                                               ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                               "output");

        itty_exec_buffer_slice_t imprints[3] = {
                itty_exec_buffer_get_word_slice (imprint_a_buffer, 0, 1),
                itty_exec_buffer_get_word_slice (imprint_b_buffer, 0, 1),
                itty_exec_buffer_get_word_slice (imprint_c_buffer, 0, 1)
        };

        assert (itty_exec_buffer_add_weighted_condense (exec_buffer,
                                                        itty_exec_buffer_get_bit_slice (output_buffer, 0, 1, 4),
                                                        imprints,
                                                        itty_exec_buffer_get_word_slice (votes_buffer, 0, 3),
                                                        3));
        assert (itty_exec_buffer_run (exec_buffer));
        assert (((size_t *) itty_bit_string_get_words (output))[0] == 0b1100);

        itty_exec_buffer_free (exec_buffer);
        itty_bit_string_free (output);
        itty_bit_string_free (imprint_c);
        itty_bit_string_free (imprint_b);
        itty_bit_string_free (imprint_a);
}

static void
test_itty_exec_buffer_weighted_condense_zero_votes (void)
{
        itty_bit_string_t *imprint_a = create_bit_string (~0UL);
        itty_bit_string_t *imprint_b = create_bit_string (~0UL);
        itty_bit_string_t *output = create_bit_string (~0UL);
        size_t votes[2] = { 0, 0 };

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();

        itty_exec_buffer_id_t imprint_a_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                  itty_bit_string_get_words (imprint_a),
                                                                                  1,
                                                                                  ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                  "imprint_a");
        itty_exec_buffer_id_t imprint_b_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                  itty_bit_string_get_words (imprint_b),
                                                                                  1,
                                                                                  ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                  "imprint_b");
        itty_exec_buffer_id_t votes_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              votes,
                                                                              2,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "votes");
        itty_exec_buffer_id_t output_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                               itty_bit_string_get_words (output),
                                                                               1,
                                                                               ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                               "output");

        itty_exec_buffer_slice_t imprints[2] = {
                itty_exec_buffer_get_word_slice (imprint_a_buffer, 0, 1),
                itty_exec_buffer_get_word_slice (imprint_b_buffer, 0, 1)
        };

        assert (itty_exec_buffer_add_weighted_condense (exec_buffer,
                                                        itty_exec_buffer_get_bit_slice (output_buffer, 0, 1, 4),
                                                        imprints,
                                                        itty_exec_buffer_get_word_slice (votes_buffer, 0, 2),
                                                        2));
        assert (itty_exec_buffer_run (exec_buffer));
        assert (((size_t *) itty_bit_string_get_words (output))[0] == 0);

        itty_exec_buffer_free (exec_buffer);
        itty_bit_string_free (output);
        itty_bit_string_free (imprint_b);
        itty_bit_string_free (imprint_a);
}

static void
test_itty_exec_buffer_run_stage (void)
{
        size_t input = 0b1010;
        size_t mask = 0b0011;
        size_t scratch = 0;
        size_t output[2] = { 0, 0 };

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();

        itty_exec_buffer_id_t input_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              &input,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "input");
        itty_exec_buffer_id_t mask_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                             &mask,
                                                                             1,
                                                                             ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                             "mask");
        itty_exec_buffer_id_t scratch_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                &scratch,
                                                                                1,
                                                                                ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                                "scratch");
        itty_exec_buffer_id_t output_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                               output,
                                                                               2,
                                                                               ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                               "output");

        itty_exec_buffer_begin_stage (exec_buffer);
        assert (itty_exec_buffer_add_xor (exec_buffer,
                                          itty_exec_buffer_get_word_slice (scratch_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (input_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (mask_buffer, 0, 1)));

        itty_exec_buffer_begin_stage (exec_buffer);
        assert (itty_exec_buffer_add_double (exec_buffer,
                                             itty_exec_buffer_get_word_slice (output_buffer, 0, 2),
                                             itty_exec_buffer_get_word_slice (scratch_buffer, 0, 1)));

        assert (itty_exec_buffer_get_stage_count (exec_buffer) == 2);
        assert (itty_exec_buffer_run_stage (exec_buffer, 0));
        assert (scratch == 0b1001);
        assert (output[0] == 0);
        assert (output[1] == 0);

        assert (itty_exec_buffer_run_stage (exec_buffer, 1));
        assert (output[0] == 0b1001);
        assert (output[1] == 0b1001);
        assert (!itty_exec_buffer_run_stage (exec_buffer, 2));

        itty_exec_buffer_free (exec_buffer);
}

static void
test_itty_exec_buffer_rebind_words (void)
{
        size_t input = 0b1111;
        size_t replacement_input = 0b0101;
        size_t mask = 0b0011;
        size_t output = 0;

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();

        itty_exec_buffer_id_t input_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              &input,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "input");
        itty_exec_buffer_id_t mask_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                             &mask,
                                                                             1,
                                                                             ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                             "mask");
        itty_exec_buffer_id_t output_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                               &output,
                                                                               1,
                                                                               ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                               "output");
        itty_exec_buffer_id_t owned_buffer = itty_exec_buffer_allocate_words (exec_buffer,
                                                                              1,
                                                                              "owned");

        assert (itty_exec_buffer_add_xor (exec_buffer,
                                          itty_exec_buffer_get_word_slice (output_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (input_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (mask_buffer, 0, 1)));

        assert (itty_exec_buffer_run (exec_buffer));
        assert (output == 0b1100);

        assert (itty_exec_buffer_rebind_words (exec_buffer, input_buffer, &replacement_input, 1));
        assert (!itty_exec_buffer_rebind_words (exec_buffer, owned_buffer, &replacement_input, 1));
        assert (itty_exec_buffer_run (exec_buffer));
        assert (output == 0b0110);

        itty_exec_buffer_free (exec_buffer);
}

static void
test_itty_exec_buffer_clear_array_range (void)
{
        size_t words[5] = { 1, 2, 3, 4, 5 };
        size_t start = 1;
        size_t count = 2;

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();

        itty_exec_buffer_id_t words_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              words,
                                                                              5,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                              "words");
        itty_exec_buffer_id_t start_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              &start,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "start");
        itty_exec_buffer_id_t count_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              &count,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "count");

        assert (itty_exec_buffer_add_clear_array_range (exec_buffer,
                                                        itty_exec_buffer_get_array (words_buffer, 0, 5),
                                                        itty_exec_buffer_get_value (start_buffer, 0),
                                                        itty_exec_buffer_get_value (count_buffer, 0)));
        assert (itty_exec_buffer_run (exec_buffer));
        assert (words[0] == 1);
        assert (words[1] == 0);
        assert (words[2] == 0);
        assert (words[3] == 4);
        assert (words[4] == 5);

        start = 3;
        count = 100;
        assert (itty_exec_buffer_run (exec_buffer));
        assert (words[0] == 1);
        assert (words[1] == 0);
        assert (words[2] == 0);
        assert (words[3] == 0);
        assert (words[4] == 0);

        char *description = itty_exec_buffer_present (exec_buffer);
        assert (description != NULL);
        assert (strstr (description, "command 0: CLEAR_ARRAY_RANGE") != NULL);
        free (description);

        itty_exec_buffer_free (exec_buffer);
}

static void
test_itty_exec_buffer_rejects_invalid_slices_when_adding_commands (void)
{
        size_t input = 0b1111;
        size_t output = 0;
        size_t votes[1] = { 1 };

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();

        itty_exec_buffer_id_t input_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              &input,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "input");
        itty_exec_buffer_id_t output_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                               &output,
                                                                               1,
                                                                               ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                               "output");
        itty_exec_buffer_id_t votes_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              votes,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "votes");
        itty_exec_buffer_slice_t inputs[1] = {
                itty_exec_buffer_get_word_slice (input_buffer, 0, 1)
        };

        assert (!itty_exec_buffer_add_xor (exec_buffer,
                                           itty_exec_buffer_get_word_slice (output_buffer, 1, 1),
                                           itty_exec_buffer_get_word_slice (input_buffer, 0, 1),
                                           itty_exec_buffer_get_word_slice (input_buffer, 0, 1)));
        assert (!itty_exec_buffer_add_popcount (exec_buffer,
                                                itty_exec_buffer_get_word_slice (output_buffer, 0, 1),
                                                itty_exec_buffer_get_bit_slice (input_buffer, 0, 1, ITTY_BIT_STRING_WORD_SIZE_IN_BITS + 1)));
        assert (!itty_exec_buffer_add_xor (exec_buffer,
                                           itty_exec_buffer_get_word_slice (input_buffer, 0, 1),
                                           itty_exec_buffer_get_word_slice (input_buffer, 0, 1),
                                           itty_exec_buffer_get_word_slice (input_buffer, 0, 1)));
        assert (!itty_exec_buffer_add_weighted_condense (exec_buffer,
                                                         itty_exec_buffer_get_word_slice (output_buffer, 0, 1),
                                                         inputs,
                                                         itty_exec_buffer_get_word_slice (votes_buffer, 0, 0),
                                                         1));
        assert (!itty_exec_buffer_add_clear_array_range (exec_buffer,
                                                         itty_exec_buffer_get_bits (output_buffer, 0, 1, 4),
                                                         itty_exec_buffer_get_value (votes_buffer, 0),
                                                         itty_exec_buffer_get_value (votes_buffer, 0)));
        assert (!itty_exec_buffer_add_clear_array_range (exec_buffer,
                                                         itty_exec_buffer_get_array (output_buffer, 0, 1),
                                                         itty_exec_buffer_get_array (votes_buffer, 0, 0),
                                                         itty_exec_buffer_get_value (votes_buffer, 0)));
        assert (itty_exec_buffer_get_stage_count (exec_buffer) == 0);

        itty_exec_buffer_free (exec_buffer);
}

static void
test_itty_exec_buffer_present (void)
{
        size_t input = 0b1010;
        size_t mask = 0b0011;
        size_t output = 0;

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();

        itty_exec_buffer_id_t input_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              &input,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "input");
        itty_exec_buffer_id_t mask_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                             &mask,
                                                                             1,
                                                                             ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                             "mask");
        itty_exec_buffer_id_t output_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                               &output,
                                                                               1,
                                                                               ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                               "output");

        assert (itty_exec_buffer_add_xor (exec_buffer,
                                          itty_exec_buffer_get_word_slice (output_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (input_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (mask_buffer, 0, 1)));

        char *description = itty_exec_buffer_present (exec_buffer);
        assert (description != NULL);
        assert (strstr (description, "exec_buffer descriptors=3 stages=1 commands=1") != NULL);
        assert (strstr (description, "descriptor 0: name=input storage=external access=read-only words=1") != NULL);
        assert (strstr (description, "descriptor 2: name=output storage=external access=read-write words=1") != NULL);
        assert (strstr (description, "stage 0: name=(unnamed) first_command=0 command_count=1") != NULL);
        assert (strstr (description, "command 0: XOR destination=(buffer=2 word_offset=0 words=1 bit_length=0)") != NULL);
        free (description);

        itty_exec_buffer_free (exec_buffer);
}

static void
test_itty_exec_buffer_named_stage_present (void)
{
        size_t input = 0b1010;
        size_t output = 0;

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();

        itty_exec_buffer_id_t input_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              &input,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "input");
        itty_exec_buffer_id_t output_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                               &output,
                                                                               1,
                                                                               ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                               "output");

        itty_exec_buffer_begin_named_stage (exec_buffer, "named");
        assert (itty_exec_buffer_add_xor (exec_buffer,
                                          itty_exec_buffer_get_word_slice (output_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (input_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (input_buffer, 0, 1)));

        char *description = itty_exec_buffer_present (exec_buffer);
        assert (description != NULL);
        assert (strstr (description, "stage 0: name=named first_command=0 command_count=1") != NULL);
        free (description);

        size_t stage_index = 100;
        assert (itty_exec_buffer_find_stage (exec_buffer, "named", &stage_index));
        assert (stage_index == 0);
        assert (!itty_exec_buffer_find_stage (exec_buffer, "missing", &stage_index));

        itty_exec_buffer_free (exec_buffer);
}

static void
test_itty_exec_buffer_run_named_stage (void)
{
        size_t input = 0b1010;
        size_t mask = 0b0011;
        size_t scratch = 0;
        size_t output[2] = { 0, 0 };

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();
        itty_manager_t *manager = itty_manager_new ();

        itty_exec_buffer_id_t input_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              &input,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "input");
        itty_exec_buffer_id_t mask_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                             &mask,
                                                                             1,
                                                                             ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                             "mask");
        itty_exec_buffer_id_t scratch_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                &scratch,
                                                                                1,
                                                                                ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                                "scratch");
        itty_exec_buffer_id_t output_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                               output,
                                                                               2,
                                                                               ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                               "output");

        assert (manager != NULL);
        itty_exec_buffer_begin_named_stage (exec_buffer, "xor scratch");
        assert (itty_exec_buffer_add_xor (exec_buffer,
                                          itty_exec_buffer_get_word_slice (scratch_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (input_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (mask_buffer, 0, 1)));

        itty_exec_buffer_begin_named_stage (exec_buffer, "double output");
        assert (itty_exec_buffer_add_double (exec_buffer,
                                             itty_exec_buffer_get_word_slice (output_buffer, 0, 2),
                                             itty_exec_buffer_get_word_slice (scratch_buffer, 0, 1)));

        assert (itty_exec_buffer_run_named_stage (exec_buffer, "xor scratch"));
        assert (scratch == 0b1001);
        assert (output[0] == 0);
        assert (output[1] == 0);

        assert (itty_exec_buffer_run_named_stage_with_manager (exec_buffer, "double output", manager));
        assert (output[0] == 0b1001);
        assert (output[1] == 0b1001);
        assert (!itty_exec_buffer_run_named_stage (exec_buffer, "missing"));
        assert (!itty_exec_buffer_run_named_stage_with_manager (exec_buffer, "missing", manager));

        itty_manager_free (manager);
        itty_exec_buffer_free (exec_buffer);
}

static void
test_itty_exec_buffer_duplicate_stage_names_are_ambiguous (void)
{
        size_t input = 0b1010;
        size_t first_output = 0;
        size_t second_output = 0;

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();

        itty_exec_buffer_id_t input_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              &input,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "input");
        itty_exec_buffer_id_t first_output_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                     &first_output,
                                                                                     1,
                                                                                     ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                                     "first output");
        itty_exec_buffer_id_t second_output_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                      &second_output,
                                                                                      1,
                                                                                      ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                                      "second output");

        itty_exec_buffer_begin_named_stage (exec_buffer, "duplicate");
        assert (itty_exec_buffer_add_xor (exec_buffer,
                                          itty_exec_buffer_get_word_slice (first_output_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (input_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (input_buffer, 0, 1)));

        itty_exec_buffer_begin_named_stage (exec_buffer, "duplicate");
        assert (itty_exec_buffer_add_xor (exec_buffer,
                                          itty_exec_buffer_get_word_slice (second_output_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (input_buffer, 0, 1),
                                          itty_exec_buffer_get_word_slice (input_buffer, 0, 1)));

        size_t stage_index = 0;
        assert (!itty_exec_buffer_find_stage (exec_buffer, "duplicate", &stage_index));
        assert (!itty_exec_buffer_run_named_stage (exec_buffer, "duplicate"));

        assert (itty_exec_buffer_run_stage (exec_buffer, 0));
        assert (itty_exec_buffer_run_stage (exec_buffer, 1));
        assert (first_output == 0);
        assert (second_output == 0);

        itty_exec_buffer_free (exec_buffer);
}

static void
test_itty_exec_buffer_find_descriptor (void)
{
        size_t input = 0;
        size_t first_duplicate = 1;
        size_t second_duplicate = 2;

        itty_exec_buffer_t *exec_buffer = itty_exec_buffer_new ();

        itty_exec_buffer_id_t input_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                              &input,
                                                                              1,
                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                              "input");
        itty_exec_buffer_id_t first_duplicate_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                        &first_duplicate,
                                                                                        1,
                                                                                        ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                        "duplicate");
        itty_exec_buffer_id_t second_duplicate_buffer = itty_exec_buffer_register_words (exec_buffer,
                                                                                         &second_duplicate,
                                                                                         1,
                                                                                         ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                         "duplicate");
        assert (input_buffer != ITTY_EXEC_BUFFER_INVALID_ID);
        assert (first_duplicate_buffer != ITTY_EXEC_BUFFER_INVALID_ID);
        assert (second_duplicate_buffer != ITTY_EXEC_BUFFER_INVALID_ID);

        itty_exec_buffer_id_t found_buffer = ITTY_EXEC_BUFFER_INVALID_ID;
        assert (itty_exec_buffer_find_descriptor (exec_buffer, "input", &found_buffer));
        assert (found_buffer == input_buffer);
        assert (!itty_exec_buffer_find_descriptor (exec_buffer, "missing", &found_buffer));
        assert (!itty_exec_buffer_find_descriptor (exec_buffer, "duplicate", &found_buffer));

        itty_exec_buffer_free (exec_buffer);
}

static void
test_itty_exec_buffer_value_array_and_bits_aliases (void)
{
        itty_exec_buffer_value_t value = itty_exec_buffer_get_value (7, 2);
        assert (value.buffer_id == 7);
        assert (value.word_offset == 2);
        assert (value.number_of_words == 1);
        assert (value.bit_length == 0);

        itty_exec_buffer_array_t array = itty_exec_buffer_get_array (8, 3, 5);
        assert (array.buffer_id == 8);
        assert (array.word_offset == 3);
        assert (array.number_of_words == 5);
        assert (array.bit_length == 0);

        itty_exec_buffer_bits_t bits = itty_exec_buffer_get_bits (9, 4, 6, 31);
        assert (bits.buffer_id == 9);
        assert (bits.word_offset == 4);
        assert (bits.number_of_words == 6);
        assert (bits.bit_length == 31);
}

int
main (void)
{
        test_itty_exec_buffer_feed_commands ();
        test_itty_exec_buffer_invalidates_registered_bit_string ();
        test_itty_exec_buffer_attention_scores ();
        test_itty_exec_buffer_popcount_respects_bit_slice_length ();
        test_itty_exec_buffer_weighted_condense ();
        test_itty_exec_buffer_weighted_condense_zero_votes ();
        test_itty_exec_buffer_run_stage ();
        test_itty_exec_buffer_rebind_words ();
        test_itty_exec_buffer_clear_array_range ();
        test_itty_exec_buffer_rejects_invalid_slices_when_adding_commands ();
        test_itty_exec_buffer_present ();
        test_itty_exec_buffer_named_stage_present ();
        test_itty_exec_buffer_run_named_stage ();
        test_itty_exec_buffer_duplicate_stage_names_are_ambiguous ();
        test_itty_exec_buffer_find_descriptor ();
        test_itty_exec_buffer_value_array_and_bits_aliases ();
        printf ("All itty-exec-buffer tests passed.\n");
        return 0;
}
