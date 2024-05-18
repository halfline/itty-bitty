#pragma once

#include "itty-bit-string-list.h"

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

itty_network_node_t *itty_network_node_new (itty_bit_string_list_t *modulation_masks);
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

void itty_network_iterator_init (itty_network_t          *network,
                                 itty_network_iterator_t *iterator);

bool itty_network_iterator_next (itty_network_iterator_t *iterator,
                                 itty_network_layer_t **layer);
