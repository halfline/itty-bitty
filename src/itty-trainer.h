#pragma once

#include "itty-bit-string-list.h"
#include "itty-bit-string.h"
#include "itty-manager.h"
#include "itty-network.h"

typedef struct itty_trainer_result_t itty_trainer_result_t;

itty_trainer_result_t *itty_trainer_run (itty_network_t         *network,
                                         itty_bit_string_list_t *input,
                                         itty_bit_string_t      *expected_payload,
                                         size_t                  payload_bit_count,
                                         itty_manager_t         *manager);
void itty_trainer_result_free (itty_trainer_result_t *result);

itty_bit_string_list_t *itty_trainer_result_get_outputs (itty_trainer_result_t *result);
itty_bit_string_t *itty_trainer_result_get_selected_activation (itty_trainer_result_t *result);
size_t itty_trainer_result_get_selected_index (itty_trainer_result_t *result);
bool itty_trainer_result_header_valid (itty_trainer_result_t *result);
size_t itty_trainer_result_get_score_pop_count (itty_trainer_result_t *result);
size_t itty_trainer_result_get_payload_start (itty_trainer_result_t *result);
itty_bit_string_t *itty_trainer_result_get_payload (itty_trainer_result_t *result);
size_t itty_trainer_result_get_payload_distance (itty_trainer_result_t *result);
