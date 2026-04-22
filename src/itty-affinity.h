#pragma once

#include "itty-bit-string-list.h"
#include "itty-bit-string.h"
#include "itty-manager.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct itty_affinity_t itty_affinity_t;
typedef struct itty_affinity_plan_t itty_affinity_plan_t;

typedef struct {
        size_t total_votes;
        size_t probe_index;
        size_t locality_window;
        size_t gray_position_bits;
        size_t gray_position_weight;
        size_t score_bit_length;
        bool causal;
} itty_affinity_probe_options_t;

itty_affinity_t *itty_affinity_new (itty_bit_string_list_t *traits,
                                    itty_bit_string_list_t *imprints);
void itty_affinity_free (itty_affinity_t *affinity);

itty_affinity_plan_t *itty_affinity_plan_new (itty_affinity_t *affinity,
                                              size_t           score_bit_length);
void itty_affinity_plan_free (itty_affinity_plan_t *plan);
char *itty_affinity_plan_present (itty_affinity_plan_t *plan);
itty_bit_string_t *itty_affinity_plan_probe (itty_affinity_plan_t                 *plan,
                                             itty_bit_string_t                   *probe,
                                             itty_affinity_probe_options_t const *options);

itty_bit_string_t *itty_affinity_probe (itty_affinity_t *affinity,
                                        itty_bit_string_t  *probe,
                                        size_t              total_votes);
itty_bit_string_t *itty_affinity_probe_at (itty_affinity_t *affinity,
                                           itty_bit_string_t  *probe,
                                           size_t              probe_index,
                                           size_t              total_votes,
                                           size_t              locality_window);
itty_bit_string_t *itty_affinity_probe_with_options (itty_affinity_t                     *affinity,
                                                     itty_bit_string_t                  *probe,
                                                     itty_affinity_probe_options_t const *options);
itty_bit_string_list_t *itty_affinity_probe_list (itty_affinity_t                     *affinity,
                                                  itty_bit_string_list_t              *probes,
                                                  itty_affinity_probe_options_t const *base_options);
itty_bit_string_list_t *itty_affinity_probe_list_with_manager (itty_affinity_t                     *affinity,
                                                               itty_bit_string_list_t              *probes,
                                                               itty_affinity_probe_options_t const *base_options,
                                                               itty_manager_t                      *manager);
