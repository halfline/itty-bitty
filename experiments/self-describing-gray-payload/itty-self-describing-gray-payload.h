#pragma once

#include "itty-bit-string.h"
#include <stdbool.h>
#include <stddef.h>

#define ITTY_SELF_DESCRIBING_GRAY_PAYLOAD_HEADER_BITS 8

typedef struct {
        size_t decoded_payload_start;
        size_t measured_distance;
        bool   header_valid;
} itty_self_describing_gray_payload_result_t;

size_t itty_self_describing_gray_payload_gray_code (size_t value);
size_t itty_self_describing_gray_payload_gray_decode (size_t gray_code);
itty_bit_string_t *itty_self_describing_gray_payload_build_activation (itty_bit_string_t *payload,
                                                                       size_t             payload_bit_count,
                                                                       size_t             payload_start,
                                                                       size_t             activation_bit_count);
itty_bit_string_t *itty_self_describing_gray_payload_build_target (itty_bit_string_t *payload,
                                                                   size_t             payload_bit_count);
bool itty_self_describing_gray_payload_measure (itty_bit_string_t                           *activation,
                                                itty_bit_string_t                           *target,
                                                size_t                                       payload_bit_count,
                                                itty_self_describing_gray_payload_result_t  *result);
