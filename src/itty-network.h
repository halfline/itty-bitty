#pragma once

#include "itty-affinity.h"
#include "itty-bit-string-list.h"
#include "itty-manager.h"
#include <stdbool.h>

typedef struct itty_network_t itty_network_t;
typedef struct itty_network_layer_t itty_network_layer_t;
typedef struct itty_network_node_t itty_network_node_t;

typedef struct {
        itty_network_layer_t *layer;
        size_t current_index;
} itty_network_layer_iterator_t;

typedef struct {
        itty_network_t *network;
        size_t current_index;
} itty_network_iterator_t;

/* Takes ownership of modulation_masks on success. */
itty_network_node_t *itty_network_feed_node_new (itty_bit_string_list_t *modulation_masks);
itty_network_node_t *itty_network_feed_node_new_with_rotation (itty_bit_string_list_t *modulation_masks,
                                                               size_t                  rotation);
/*
 * Takes ownership of traits and imprints on success. On validation failure,
 * returns NULL and leaves ownership with the caller.
 */
itty_network_node_t *itty_network_affinity_node_new (itty_bit_string_list_t                   *traits,
                                                     itty_bit_string_list_t                   *imprints,
                                                     itty_affinity_probe_options_t const      *options);
void itty_network_node_free (itty_network_node_t *node);

itty_network_layer_t *itty_network_layer_new (void);

void itty_network_layer_append (itty_network_layer_t *layer,
                                itty_network_node_t  *node);
void itty_network_layer_free (itty_network_layer_t *layer);
itty_network_t *itty_network_new (void);
void itty_network_free (itty_network_t *network);
void itty_network_append (itty_network_t       *network,
                          itty_network_layer_t *layer);

itty_bit_string_list_t *itty_network_feed (itty_network_t         *network,
                                           itty_bit_string_list_t *input);
itty_bit_string_list_t *itty_network_feed_with_manager (itty_network_t         *network,
                                                        itty_bit_string_list_t *input,
                                                        itty_manager_t         *manager);
itty_bit_string_list_t *itty_network_layer_feed (itty_network_layer_t   *layer,
                                                 itty_bit_string_list_t *input);
itty_bit_string_list_t *itty_network_layer_feed_with_manager (itty_network_layer_t   *layer,
                                                              itty_bit_string_list_t *input,
                                                              itty_manager_t         *manager);
bool itty_network_select_output (itty_bit_string_list_t *outputs,
                                 size_t                 *index);
char *itty_network_layer_present (itty_network_layer_t   *layer,
                                  itty_bit_string_list_t *input);
char *itty_network_present_plan (itty_network_t         *network,
                                 itty_bit_string_list_t *input);

void itty_network_iterator_init (itty_network_t          *network,
                                 itty_network_iterator_t *iterator);

bool itty_network_iterator_next (itty_network_iterator_t *iterator,
                                 itty_network_layer_t **layer);
