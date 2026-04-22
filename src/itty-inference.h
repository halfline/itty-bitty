#pragma once

#include "itty-bit-string.h"
#include "itty-bit-string-list.h"
#include "itty-manager.h"
#include "itty-network.h"
#include "itty-vocabulary.h"

typedef struct itty_inference_result_t itty_inference_result_t;

itty_inference_result_t *itty_inference_run (itty_network_t         *network,
                                             itty_bit_string_list_t *input,
                                             itty_vocabulary_t      *vocabulary,
                                             itty_manager_t         *manager);
void itty_inference_result_free (itty_inference_result_t *result);

itty_bit_string_list_t *itty_inference_result_get_outputs (itty_inference_result_t *result);
itty_bit_string_t *itty_inference_result_get_selected_activation (itty_inference_result_t *result);
size_t itty_inference_result_get_selected_index (itty_inference_result_t *result);
const char *itty_inference_result_get_text (itty_inference_result_t *result);
size_t itty_inference_result_get_distance (itty_inference_result_t *result);
