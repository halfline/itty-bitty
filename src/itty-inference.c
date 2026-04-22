#include "itty-inference.h"

#include <stdlib.h>

struct itty_inference_result_t {
        itty_bit_string_list_t *input;
        itty_bit_string_list_t *outputs;
        char                   *text;
        size_t                  selected_index;
        size_t                  distance;
};

itty_inference_result_t *
itty_inference_run (itty_network_t         *network,
                    itty_bit_string_list_t *input,
                    itty_vocabulary_t      *vocabulary,
                    itty_manager_t         *manager)
{
        size_t selected_index = 0;
        itty_bit_string_list_t *outputs = itty_network_feed_with_manager (network,
                                                                          input,
                                                                          manager);
        if (!outputs || !itty_network_select_output (outputs, &selected_index)) {
                if (outputs && outputs != input)
                        itty_bit_string_list_free (outputs);
                return NULL;
        }

        itty_bit_string_t *selected_activation = itty_bit_string_list_fetch (outputs,
                                                                             selected_index);
        char *text = NULL;
        size_t distance = 0;
        if (!itty_vocabulary_decode_nearest (vocabulary, selected_activation, &text, &distance)) {
                if (outputs != input)
                        itty_bit_string_list_free (outputs);
                return NULL;
        }

        itty_inference_result_t *result = malloc (sizeof (itty_inference_result_t));
        result->input = input;
        result->outputs = outputs;
        result->text = text;
        result->selected_index = selected_index;
        result->distance = distance;

        return result;
}

void
itty_inference_result_free (itty_inference_result_t *result)
{
        if (!result)
                return;

        if (result->outputs != result->input)
                itty_bit_string_list_free (result->outputs);
        free (result->text);
        free (result);
}

itty_bit_string_list_t *
itty_inference_result_get_outputs (itty_inference_result_t *result)
{
        return result->outputs;
}

itty_bit_string_t *
itty_inference_result_get_selected_activation (itty_inference_result_t *result)
{
        return itty_bit_string_list_fetch (result->outputs, result->selected_index);
}

size_t
itty_inference_result_get_selected_index (itty_inference_result_t *result)
{
        return result->selected_index;
}

const char *
itty_inference_result_get_text (itty_inference_result_t *result)
{
        return result->text;
}

size_t
itty_inference_result_get_distance (itty_inference_result_t *result)
{
        return result->distance;
}
