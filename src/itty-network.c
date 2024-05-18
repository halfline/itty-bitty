#include "itty-network.h"
#include "itty-bit-string.h"

#include <stdio.h>
#include <stdlib.h>

struct itty_network_node_t {
        itty_bit_string_list_t *modulation_masks;
};

struct itty_network_layer_t {
        itty_network_node_t **nodes;
        size_t number_of_nodes;
};

struct itty_network_t {
        itty_network_layer_t **layers;
        size_t number_of_layers;
};

itty_network_node_t *
itty_network_node_new (itty_bit_string_list_t *modulation_masks)
{
        itty_network_node_t *node = malloc (sizeof (itty_network_node_t));
        node->modulation_masks = modulation_masks;
        return node;
}

void
itty_network_node_free (itty_network_node_t *node)
{
        if (!node)
                return;
        free (node);
}

itty_network_layer_t *
itty_network_layer_new (void)
{
    itty_network_layer_t *layer = malloc (sizeof (itty_network_layer_t));
    layer->number_of_nodes = 0;
    layer->nodes = NULL;

    return layer;
}

void
itty_network_layer_append (itty_network_layer_t *layer,
                           itty_network_node_t  *node)
{
        layer->nodes = realloc (layer->nodes,
                                (layer->number_of_nodes + 1) * sizeof (itty_network_node_t *));
        layer->nodes[layer->number_of_nodes] = node;
        layer->number_of_nodes++;
}

void
itty_network_layer_free (itty_network_layer_t *layer)
{
        if (!layer)
                return;

        for (size_t i = 0; i < layer->number_of_nodes; i++) {
                itty_network_node_free (layer->nodes[i]);
        }
        free (layer->nodes);
        free (layer);
}

void
itty_network_iterator_init (itty_network_t          *network,
                            itty_network_iterator_t *iterator)
{
        iterator->network = network;
        iterator->current_index = 0;
}

bool
itty_network_iterator_next (itty_network_iterator_t  *iterator,
                            itty_network_layer_t    **layer)
{
        if (iterator->current_index < iterator->network->number_of_layers) {
                *layer = iterator->network->layers[iterator->current_index++];
                return true;
        } else {
                *layer = NULL;
                return false;
        }
}

void
itty_network_layer_iterator_init (itty_network_layer_t          *layer,
                                  itty_network_layer_iterator_t *iterator)
{
        iterator->layer = layer;
        iterator->current_index = 0;
}

bool
itty_network_layer_iterator_next (itty_network_layer_iterator_t  *iterator,
                                  itty_network_node_t           **node)
{
    if (iterator->current_index < iterator->layer->number_of_nodes) {
            *node = iterator->layer->nodes[iterator->current_index++];
            return true;
    } else {
            *node = NULL;
            return false;
    }
}

itty_bit_string_list_t *
itty_network_feed (itty_network_t         *network,
                   itty_bit_string_list_t *input)
{
        itty_bit_string_list_t *current_input = input;
        itty_network_iterator_t net_iterator;
        itty_network_iterator_init (network, &net_iterator);
        itty_network_layer_t *layer;

        size_t layer_index = 0;
        while (itty_network_iterator_next (&net_iterator, &layer)) {
                itty_bit_string_list_t *layer_outputs = itty_bit_string_list_new ();
                itty_network_layer_iterator_t layer_iterator;
                itty_network_layer_iterator_init (layer, &layer_iterator);
                itty_network_node_t *node;

                printf ("Layer %zu\n", layer_index);
                while (itty_network_layer_iterator_next (&layer_iterator, &node)) {
                        printf ("\tlayer inputs:\n%s\n", itty_bit_string_list_present (current_input, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY));
                        itty_bit_string_list_t *modulated_inputs = itty_bit_string_list_exclusive_or (current_input, node->modulation_masks);
                        printf ("\tmodulated inputs:\n%s\n", itty_bit_string_list_present (modulated_inputs, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY));
                        itty_bit_string_t *condensed_output = itty_bit_string_list_condense (modulated_inputs);
                        printf ("\tcondensed output: %s\n", itty_bit_string_present (condensed_output, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY));
                        itty_bit_string_t *doubled_output = itty_bit_string_double (condensed_output);
                        printf ("\tdoubled output: %s\n", itty_bit_string_present (doubled_output, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY));
                        itty_bit_string_free (condensed_output);
                        itty_bit_string_list_append (layer_outputs, doubled_output);
                        printf ("\tlayout outputs:\n%s\n", itty_bit_string_list_present (layer_outputs, ITTY_BIT_STRING_PRESENTATION_FORMAT_BINARY_FOR_DISPLAY));
                }

                if (current_input != input)
                        itty_bit_string_list_free (current_input);

                current_input = layer_outputs;
                layer_index++;
        }

        return current_input;
}

itty_network_t *
itty_network_new (void)
{
        itty_network_t *network = malloc (sizeof (itty_network_t));
        network->number_of_layers = 0;
        network->layers = NULL;

        return network;
}

void
itty_network_free (itty_network_t *network)
{
        if (!network)
                return;

        for (size_t i = 0; i < network->number_of_layers; i++) {
                itty_network_layer_free (network->layers[i]);
        }
        free (network);
}

void
itty_network_append (itty_network_t      *network,
                     itty_network_layer_t *layer)
{
        network->layers = realloc (network->layers,
                                   (network->number_of_layers + 1) * sizeof (itty_network_layer_t *));
        network->layers[network->number_of_layers] = layer;
        network->number_of_layers++;
}
