#pragma once

#include "itty-bit-string-list.h"
#include "itty-bit-string.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct itty_route_key_selector_t itty_route_key_selector_t;

typedef struct {
        bool   has_selection;
        size_t selected_route;
        size_t selected_score;
        size_t runner_up_score;
        size_t selected_gap;
} itty_route_key_selection_t;

itty_route_key_selector_t *itty_route_key_selector_new (void);
void itty_route_key_selector_free (itty_route_key_selector_t *selector);
bool itty_route_key_selector_append_route (itty_route_key_selector_t *selector,
                                           itty_bit_string_t         *route_key);
size_t itty_route_key_selector_get_route_count (itty_route_key_selector_t *selector);
bool itty_route_key_selector_select (itty_route_key_selector_t    *selector,
                                     itty_bit_string_t            *probe,
                                     itty_route_key_selection_t   *selection);
