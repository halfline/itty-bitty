#include "itty-trainer.h"

#include "itty-decoder.h"

#include <stdlib.h>

struct itty_trainer_result_t {
        itty_bit_string_list_t *input;
        itty_bit_string_list_t *outputs;
        size_t                  selected_index;
        bool                    header_valid;
        size_t                  score_pop_count;
        size_t                  payload_start;
        itty_bit_string_t      *payload;
        size_t                  payload_distance;
};

itty_trainer_result_t *
itty_trainer_run (itty_network_t         *network,
                  itty_bit_string_list_t *input,
                  itty_bit_string_t      *expected_payload,
                  size_t                  payload_bit_count,
                  itty_manager_t         *manager)
{
        size_t selected_index = 0;
        itty_bit_string_list_t *outputs = itty_network_feed_with_manager (network,
                                                                          input,
                                                                          manager);
        itty_bit_string_t *selected_activation;
        itty_decoder_result_t *decoder_result;
        itty_trainer_result_t *result;

        if (!outputs || !expected_payload || payload_bit_count == 0 ||
            !itty_decoder_select_output (outputs, payload_bit_count, &selected_index)) {
                if (outputs && outputs != input)
                        itty_bit_string_list_free (outputs);
                return NULL;
        }

        selected_activation = itty_bit_string_list_fetch (outputs, selected_index);
        decoder_result = itty_decoder_decode (selected_activation, payload_bit_count);
        if (!decoder_result) {
                if (outputs != input)
                        itty_bit_string_list_free (outputs);
                return NULL;
        }

        result = malloc (sizeof (itty_trainer_result_t));
        if (!result) {
                itty_decoder_result_free (decoder_result);
                if (outputs != input)
                        itty_bit_string_list_free (outputs);
                return NULL;
        }

        result->input = input;
        result->outputs = outputs;
        result->selected_index = selected_index;
        result->header_valid = itty_decoder_result_header_valid (decoder_result);
        result->score_pop_count = itty_decoder_result_get_score_pop_count (decoder_result);
        result->payload_start = itty_decoder_result_get_payload_start (decoder_result);
        result->payload = itty_bit_string_clone (itty_decoder_result_get_payload (decoder_result));
        result->payload_distance = itty_decoder_measure_distance (decoder_result, expected_payload);
        itty_decoder_result_free (decoder_result);

        if (!result->payload) {
                free (result);
                if (outputs != input)
                        itty_bit_string_list_free (outputs);
                return NULL;
        }
        return result;
}

void
itty_trainer_result_free (itty_trainer_result_t *result)
{
        if (!result)
                return;

        if (result->outputs != result->input)
                itty_bit_string_list_free (result->outputs);
        itty_bit_string_free (result->payload);
        free (result);
}

itty_bit_string_list_t *
itty_trainer_result_get_outputs (itty_trainer_result_t *result)
{
        return result->outputs;
}

itty_bit_string_t *
itty_trainer_result_get_selected_activation (itty_trainer_result_t *result)
{
        return itty_bit_string_list_fetch (result->outputs, result->selected_index);
}

size_t
itty_trainer_result_get_selected_index (itty_trainer_result_t *result)
{
        return result->selected_index;
}

bool
itty_trainer_result_header_valid (itty_trainer_result_t *result)
{
        return result->header_valid;
}

size_t
itty_trainer_result_get_score_pop_count (itty_trainer_result_t *result)
{
        return result->score_pop_count;
}

size_t
itty_trainer_result_get_payload_start (itty_trainer_result_t *result)
{
        return result->payload_start;
}

itty_bit_string_t *
itty_trainer_result_get_payload (itty_trainer_result_t *result)
{
        return result->payload;
}

size_t
itty_trainer_result_get_payload_distance (itty_trainer_result_t *result)
{
        return result->payload_distance;
}
