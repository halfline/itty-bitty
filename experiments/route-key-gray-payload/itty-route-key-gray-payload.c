#include "itty-route-key-gray-payload.h"

#include "itty-route-key-selector.h"
#include "itty-self-describing-gray-payload.h"

#include <stdint.h>
#include <stdlib.h>

typedef struct {
        itty_bit_string_t *target;
        size_t             payload_bit_count;
        size_t             committed_distance;
} itty_route_key_gray_payload_route_t;

struct itty_route_key_gray_payload_table_t {
        itty_route_key_selector_t            *selector;
        itty_route_key_gray_payload_route_t  *routes;
        size_t                                route_count;
};

static bool
measure_selected_route (itty_route_key_gray_payload_table_t   *table,
                        size_t                                 route_index,
                        size_t                                 selected_gap,
                        itty_bit_string_t                     *activation,
                        itty_route_key_gray_payload_result_t  *result)
{
        itty_route_key_gray_payload_route_t *route;
        itty_self_describing_gray_payload_result_t payload_result;

        if (!table || !activation || !result || route_index >= table->route_count)
                return false;

        route = &table->routes[route_index];
        if (!itty_self_describing_gray_payload_measure (activation,
                                                        route->target,
                                                        route->payload_bit_count,
                                                        &payload_result))
                return false;

        *result = (itty_route_key_gray_payload_result_t) {
                .has_selection = true,
                .selected_route = route_index,
                .selected_gap = selected_gap,
                .decoded_payload_start = payload_result.decoded_payload_start,
                .header_valid = payload_result.header_valid,
                .measured_distance = payload_result.measured_distance,
                .committed_distance = route->committed_distance,
        };
        return true;
}

itty_route_key_gray_payload_table_t *
itty_route_key_gray_payload_table_new (void)
{
        itty_route_key_gray_payload_table_t *table = malloc (sizeof (itty_route_key_gray_payload_table_t));

        if (!table)
                return NULL;

        table->selector = itty_route_key_selector_new ();
        table->routes = NULL;
        table->route_count = 0;

        if (!table->selector) {
                free (table);
                return NULL;
        }

        return table;
}

void
itty_route_key_gray_payload_table_free (itty_route_key_gray_payload_table_t *table)
{
        if (!table)
                return;

        for (size_t route_index = 0; route_index < table->route_count; route_index++)
                itty_bit_string_free (table->routes[route_index].target);

        free (table->routes);
        itty_route_key_selector_free (table->selector);
        free (table);
}

bool
itty_route_key_gray_payload_table_append_route (itty_route_key_gray_payload_table_t *table,
                                                itty_bit_string_t                   *route_key,
                                                itty_bit_string_t                   *target,
                                                size_t                               payload_bit_count)
{
        itty_route_key_gray_payload_route_t *routes;
        itty_bit_string_t *target_clone;

        if (!table || !route_key || !target || payload_bit_count == 0)
                return false;

        target_clone = itty_bit_string_clone (target);
        if (!target_clone)
                return false;

        routes = realloc (table->routes,
                          (table->route_count + 1) * sizeof (itty_route_key_gray_payload_route_t));
        if (!routes) {
                itty_bit_string_free (target_clone);
                return false;
        }

        table->routes = routes;
        table->routes[table->route_count] = (itty_route_key_gray_payload_route_t) {
                .target = target_clone,
                .payload_bit_count = payload_bit_count,
                .committed_distance = SIZE_MAX,
        };

        if (!itty_route_key_selector_append_route (table->selector, route_key)) {
                itty_bit_string_free (table->routes[table->route_count].target);
                return false;
        }

        table->route_count++;
        return true;
}

size_t
itty_route_key_gray_payload_table_get_route_count (itty_route_key_gray_payload_table_t *table)
{
        return table ? table->route_count : 0;
}

bool
itty_route_key_gray_payload_table_measure (itty_route_key_gray_payload_table_t   *table,
                                           itty_bit_string_t                     *probe,
                                           itty_bit_string_t                     *activation,
                                           itty_route_key_gray_payload_result_t  *result)
{
        itty_route_key_selection_t selection;

        if (!table || !probe || !activation || !result)
                return false;

        if (!itty_route_key_selector_select (table->selector, probe, &selection))
                return false;

        if (!selection.has_selection) {
                *result = (itty_route_key_gray_payload_result_t) { 0 };
                return true;
        }

        return measure_selected_route (table,
                                       selection.selected_route,
                                       selection.selected_gap,
                                       activation,
                                       result);
}

bool
itty_route_key_gray_payload_table_commit (itty_route_key_gray_payload_table_t   *table,
                                          itty_bit_string_t                     *probe,
                                          itty_bit_string_t                     *activation,
                                          itty_route_key_gray_payload_result_t  *result)
{
        itty_route_key_selection_t selection;
        itty_route_key_gray_payload_route_t *route;

        if (!table || !probe || !activation || !result)
                return false;

        if (!itty_route_key_selector_select (table->selector, probe, &selection))
                return false;

        if (!selection.has_selection) {
                *result = (itty_route_key_gray_payload_result_t) { 0 };
                return true;
        }

        route = &table->routes[selection.selected_route];
        if (!measure_selected_route (table,
                                     selection.selected_route,
                                     selection.selected_gap,
                                     activation,
                                     result))
                return false;

        if (result->header_valid && result->measured_distance < route->committed_distance)
                route->committed_distance = result->measured_distance;

        result->committed_distance = route->committed_distance;
        return true;
}
