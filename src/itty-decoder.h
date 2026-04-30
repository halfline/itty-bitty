#pragma once

#include "itty-bit-string.h"
#include <stdbool.h>
#include <stddef.h>

#define ITTY_DECODER_HEADER_BITS 8
#define ITTY_DECODER_SCORE_BITS 8

typedef struct itty_decoder_result_t itty_decoder_result_t;

size_t itty_decoder_gray_code (size_t value);
size_t itty_decoder_gray_decode (size_t gray_code);
size_t itty_decoder_get_score_vote_count (itty_bit_string_t *activation);
bool itty_decoder_select_output (itty_bit_string_list_t *outputs,
                                 size_t                  payload_bit_count,
                                 size_t                 *index);

itty_decoder_result_t *itty_decoder_decode (itty_bit_string_t *activation,
                                            size_t             payload_bit_count);
void itty_decoder_result_free (itty_decoder_result_t *result);

bool itty_decoder_result_header_valid (itty_decoder_result_t *result);
size_t itty_decoder_result_get_payload_start (itty_decoder_result_t *result);
itty_bit_string_t *itty_decoder_result_get_score (itty_decoder_result_t *result);
size_t itty_decoder_result_get_score_pop_count (itty_decoder_result_t *result);
itty_bit_string_t *itty_decoder_result_get_payload (itty_decoder_result_t *result);
size_t itty_decoder_measure_distance (itty_decoder_result_t *result,
                                      itty_bit_string_t     *expected_payload);
