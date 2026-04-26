#include "itty-route-key-selector.h"

#include <stdlib.h>

struct itty_route_key_selector_t {
        itty_bit_string_list_t *route_keys;
};

itty_route_key_selector_t *
itty_route_key_selector_new (void)
{
        itty_route_key_selector_t *selector = malloc (sizeof (itty_route_key_selector_t));
        if (!selector)
                return NULL;

        selector->route_keys = itty_bit_string_list_new ();
        if (!selector->route_keys) {
                free (selector);
                return NULL;
        }

        return selector;
}

void
itty_route_key_selector_free (itty_route_key_selector_t *selector)
{
        if (!selector)
                return;

        itty_bit_string_list_free (selector->route_keys);
        free (selector);
}

bool
itty_route_key_selector_append_route (itty_route_key_selector_t *selector,
                                      itty_bit_string_t         *route_key)
{
        itty_bit_string_t *route_key_clone;

        if (!selector || !route_key)
                return false;

        route_key_clone = itty_bit_string_clone (route_key);
        if (!route_key_clone)
                return false;

        itty_bit_string_list_append (selector->route_keys, route_key_clone);
        return true;
}

size_t
itty_route_key_selector_get_route_count (itty_route_key_selector_t *selector)
{
        if (!selector)
                return 0;

        return itty_bit_string_list_get_length (selector->route_keys);
}

bool
itty_route_key_selector_select (itty_route_key_selector_t    *selector,
                                itty_bit_string_t            *probe,
                                itty_route_key_selection_t   *selection)
{
        size_t route_count;
        size_t best_route = 0;
        size_t best_score = 0;
        size_t runner_up_score = 0;
        bool found_one = false;

        if (!selector || !probe || !selection)
                return false;

        *selection = (itty_route_key_selection_t) { 0 };
        route_count = itty_bit_string_list_get_length (selector->route_keys);

        for (size_t route = 0; route < route_count; route++) {
                itty_bit_string_t *route_key = itty_bit_string_list_fetch (selector->route_keys, route);
                size_t score;

                if (!route_key)
                        continue;

                score = itty_bit_string_evaluate_similarity (probe, route_key);
                if (!found_one || score > best_score) {
                        runner_up_score = found_one ? best_score : 0;
                        best_score = score;
                        best_route = route;
                        found_one = true;
                } else if (score > runner_up_score) {
                        runner_up_score = score;
                }
        }

        if (!found_one)
                return true;

        *selection = (itty_route_key_selection_t) {
                .has_selection = true,
                .selected_route = best_route,
                .selected_score = best_score,
                .runner_up_score = runner_up_score,
                .selected_gap = best_score >= runner_up_score ? best_score - runner_up_score : 0
        };
        return true;
}
