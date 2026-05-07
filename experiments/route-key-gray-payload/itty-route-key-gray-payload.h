#pragma once

#include "itty-bit-string.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct itty_route_key_gray_payload_table_t itty_route_key_gray_payload_table_t;

typedef struct {
        bool   has_selection;
        size_t selected_route;
        size_t selected_gap;
        size_t decoded_payload_start;
        bool   header_valid;
        size_t measured_distance;
        size_t committed_distance;
} itty_route_key_gray_payload_result_t;

itty_route_key_gray_payload_table_t *itty_route_key_gray_payload_table_new (void);
void itty_route_key_gray_payload_table_free (itty_route_key_gray_payload_table_t *table);
bool itty_route_key_gray_payload_table_append_route (itty_route_key_gray_payload_table_t *table,
                                                     itty_bit_string_t                   *route_key,
                                                     itty_bit_string_t                   *target,
                                                     size_t                               payload_bit_count);
size_t itty_route_key_gray_payload_table_get_route_count (itty_route_key_gray_payload_table_t *table);
bool itty_route_key_gray_payload_table_measure (itty_route_key_gray_payload_table_t   *table,
                                                itty_bit_string_t                     *probe,
                                                itty_bit_string_t                     *activation,
                                                itty_route_key_gray_payload_result_t  *result);
bool itty_route_key_gray_payload_table_commit (itty_route_key_gray_payload_table_t   *table,
                                               itty_bit_string_t                     *probe,
                                               itty_bit_string_t                     *activation,
                                               itty_route_key_gray_payload_result_t  *result);
