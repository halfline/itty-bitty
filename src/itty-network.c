#include "itty-network.h"
#include "itty-affinity.h"
#include "itty-bit-string.h"
#include "itty-bit-string-list-private.h"
#include "itty-bit-string-private.h"
#include "itty-exec-buffer.h"
#include "itty-manager.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ITTY_NETWORK_STAGE_MODULATE_INPUTS "network modulate inputs"
#define ITTY_NETWORK_STAGE_CONDENSE_NODES "network condense nodes"
#define ITTY_NETWORK_STAGE_DOUBLE_OUTPUTS "network double outputs"

#define ITTY_NETWORK_BUFFER_INPUT "network input"
#define ITTY_NETWORK_BUFFER_MASK "network mask"
#define ITTY_NETWORK_BUFFER_MODULATED_INPUT "network modulated input"
#define ITTY_NETWORK_BUFFER_CONDENSED_OUTPUT "network condensed output"
#define ITTY_NETWORK_BUFFER_LAYER_OUTPUT "network layer output"

typedef enum {
        ITTY_NETWORK_NODE_FEED,
        ITTY_NETWORK_NODE_AFFINITY
} itty_network_node_kind_t;

struct itty_network_node_t {
        itty_network_node_kind_t kind;
        union {
                struct {
                        itty_bit_string_list_t *modulation_masks;
                        size_t                  rotation;
                } feed;
                struct {
                        itty_bit_string_list_t              *traits;
                        itty_bit_string_list_t              *imprints;
                        itty_affinity_t                     *affinity;
                        itty_affinity_probe_options_t        options;
                } affinity;
        };
};

struct itty_network_layer_t {
        itty_network_node_t **nodes;
        size_t number_of_nodes;
};

struct itty_network_t {
        itty_network_layer_t **layers;
        size_t number_of_layers;
};

typedef struct {
        itty_exec_buffer_t       *exec_buffer;
        itty_bit_string_list_t   *layer_outputs;
        itty_exec_buffer_slice_t **modulated_slices_by_node;
        size_t                   *modulated_counts;
        size_t                   *max_modulated_words_by_node;
        itty_exec_buffer_slice_t *condensed_slices;
        size_t                    number_of_nodes;
} itty_network_feed_layer_plan_t;

typedef enum {
        ITTY_NETWORK_LAYER_RUNNER_FEED_PLAN,
        ITTY_NETWORK_LAYER_RUNNER_MIXED
} itty_network_layer_runner_kind_t;

itty_network_node_t *
itty_network_feed_node_new (itty_bit_string_list_t *modulation_masks)
{
        return itty_network_feed_node_new_with_rotation (modulation_masks, 0);
}

itty_network_node_t *
itty_network_feed_node_new_with_rotation (itty_bit_string_list_t *modulation_masks,
                                          size_t                  rotation)
{
        itty_network_node_t *node = malloc (sizeof (itty_network_node_t));
        node->kind = ITTY_NETWORK_NODE_FEED;
        node->feed.modulation_masks = modulation_masks;
        node->feed.rotation = rotation;
        return node;
}

itty_network_node_t *
itty_network_affinity_node_new (itty_bit_string_list_t              *traits,
                                itty_bit_string_list_t              *imprints,
                                itty_affinity_probe_options_t const *options)
{
        itty_affinity_t *affinity = itty_affinity_new (traits, imprints);
        if (!affinity)
                return NULL;

        itty_network_node_t *node = malloc (sizeof (itty_network_node_t));
        node->kind = ITTY_NETWORK_NODE_AFFINITY;
        node->affinity.traits = traits;
        node->affinity.imprints = imprints;
        node->affinity.affinity = affinity;
        node->affinity.options = options ? *options : (itty_affinity_probe_options_t) { 0 };
        return node;
}

void
itty_network_node_free (itty_network_node_t *node)
{
        if (!node)
                return;
        switch (node->kind) {
        case ITTY_NETWORK_NODE_FEED:
                itty_bit_string_list_free (node->feed.modulation_masks);
                break;
        case ITTY_NETWORK_NODE_AFFINITY:
                itty_affinity_free (node->affinity.affinity);
                itty_bit_string_list_free (node->affinity.imprints);
                itty_bit_string_list_free (node->affinity.traits);
                break;
        }
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

static void
itty_bit_string_list_append_steal_all (itty_bit_string_list_t *destination,
                                       itty_bit_string_list_t *source)
{
        for (size_t i = 0; i < source->count; i++) {
                itty_bit_string_list_append (destination, source->bit_strings[i]);
                source->bit_strings[i] = NULL;
        }
}

static itty_bit_string_t *
itty_network_feed_node_run (itty_network_node_t      *node,
                            itty_bit_string_list_t  *current_input)
{
        itty_bit_string_list_t *modulated_inputs = itty_bit_string_list_exclusive_or (current_input,
                                                                                      node->feed.modulation_masks);
        itty_bit_string_t *condensed_output = itty_bit_string_list_condense (modulated_inputs);
        itty_bit_string_list_free (modulated_inputs);

        if (!condensed_output)
                condensed_output = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);

        itty_bit_string_t *doubled_output = node->feed.rotation == 0 ?
                                            itty_bit_string_double (condensed_output) :
                                            itty_bit_string_double_with_rotated_half (condensed_output,
                                                                                     node->feed.rotation);
        itty_bit_string_free (condensed_output);

        return doubled_output;
}

static itty_bit_string_list_t *
itty_network_layer_run (itty_network_layer_t   *layer,
                        itty_bit_string_list_t *current_input)
{
        itty_bit_string_list_t *layer_outputs = itty_bit_string_list_new ();
        itty_network_layer_iterator_t layer_iterator;
        itty_network_layer_iterator_init (layer, &layer_iterator);
        itty_network_node_t *node;

        while (itty_network_layer_iterator_next (&layer_iterator, &node)) {
                if (node->kind == ITTY_NETWORK_NODE_FEED) {
                        itty_bit_string_t *doubled_output = itty_network_feed_node_run (node, current_input);
                        if (!doubled_output) {
                                itty_bit_string_list_free (layer_outputs);
                                return NULL;
                        }
                        itty_bit_string_list_append (layer_outputs, doubled_output);
                } else {
                        itty_bit_string_list_t *affinity_outputs = itty_affinity_probe_list (node->affinity.affinity,
                                                                                              current_input,
                                                                                              &node->affinity.options);
                        if (!affinity_outputs) {
                                itty_bit_string_list_free (layer_outputs);
                                return NULL;
                        }

                        itty_bit_string_list_append_steal_all (layer_outputs, affinity_outputs);
                        itty_bit_string_list_free (affinity_outputs);
                }
        }

        return layer_outputs;
}

itty_bit_string_list_t *
itty_network_feed (itty_network_t         *network,
                   itty_bit_string_list_t *input)
{
        itty_bit_string_list_t *current_input = input;
        itty_network_iterator_t net_iterator;
        itty_network_iterator_init (network, &net_iterator);
        itty_network_layer_t *layer;

        while (itty_network_iterator_next (&net_iterator, &layer)) {
                itty_bit_string_list_t *layer_outputs = itty_network_layer_run (layer, current_input);
                if (!layer_outputs) {
                        if (current_input != input)
                                itty_bit_string_list_free (current_input);
                        return NULL;
                }

                if (current_input != input)
                        itty_bit_string_list_free (current_input);

                current_input = layer_outputs;
        }

        return current_input;
}

static void
itty_network_feed_layer_plan_free (itty_network_feed_layer_plan_t *plan,
                                   bool                            free_outputs)
{
        if (!plan)
                return;

        if (free_outputs)
                itty_bit_string_list_free (plan->layer_outputs);

        itty_exec_buffer_free (plan->exec_buffer);
        if (plan->modulated_slices_by_node) {
                for (size_t node_index = 0; node_index < plan->number_of_nodes; node_index++)
                        free (plan->modulated_slices_by_node[node_index]);
        }
        free (plan->modulated_slices_by_node);
        free (plan->modulated_counts);
        free (plan->max_modulated_words_by_node);
        free (plan->condensed_slices);
        free (plan);
}

static bool
build_feed_layer_modulation_stage (itty_network_feed_layer_plan_t *plan,
                                   itty_network_layer_t           *layer,
                                   itty_bit_string_list_t         *current_input)
{
        itty_exec_buffer_begin_named_stage (plan->exec_buffer, ITTY_NETWORK_STAGE_MODULATE_INPUTS);
        for (size_t node_index = 0; node_index < layer->number_of_nodes; node_index++) {
                itty_network_node_t *node = layer->nodes[node_index];
                if (node->kind != ITTY_NETWORK_NODE_FEED)
                        return false;

                size_t input_count = itty_bit_string_list_get_length (current_input);
                size_t mask_count = itty_bit_string_list_get_length (node->feed.modulation_masks);
                size_t modulated_count = input_count < mask_count ? input_count : mask_count;
                itty_exec_buffer_slice_t *modulated_slices = NULL;
                size_t max_modulated_words = 0;

                if (modulated_count > 0) {
                        modulated_slices = calloc (modulated_count, sizeof (itty_exec_buffer_slice_t));
                        if (!modulated_slices)
                                return false;
                }

                plan->modulated_slices_by_node[node_index] = modulated_slices;
                plan->modulated_counts[node_index] = modulated_count;

                for (size_t i = 0; i < modulated_count; i++) {
                        itty_bit_string_t *input_bit_string = itty_bit_string_list_fetch (current_input, i);
                        itty_bit_string_t *mask_bit_string = itty_bit_string_list_fetch (node->feed.modulation_masks, i);
                        size_t input_words = itty_bit_string_get_number_of_words (input_bit_string);
                        size_t mask_words = itty_bit_string_get_number_of_words (mask_bit_string);
                        size_t modulated_words = input_words > mask_words ? input_words : mask_words;

                        if (modulated_words > max_modulated_words)
                                max_modulated_words = modulated_words;

                        itty_exec_buffer_id_t input_buffer = itty_exec_buffer_register_words (plan->exec_buffer,
                                                                                              itty_bit_string_get_words (input_bit_string),
                                                                                              input_words,
                                                                                              ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                              ITTY_NETWORK_BUFFER_INPUT);
                        itty_exec_buffer_id_t mask_buffer = itty_exec_buffer_register_words (plan->exec_buffer,
                                                                                             itty_bit_string_get_words (mask_bit_string),
                                                                                             mask_words,
                                                                                             ITTY_EXEC_BUFFER_ACCESS_READ_ONLY,
                                                                                             ITTY_NETWORK_BUFFER_MASK);
                        itty_exec_buffer_id_t modulated_buffer = itty_exec_buffer_allocate_words (plan->exec_buffer,
                                                                                                  modulated_words,
                                                                                                  ITTY_NETWORK_BUFFER_MODULATED_INPUT);
                        if (input_buffer == ITTY_EXEC_BUFFER_INVALID_ID ||
                            mask_buffer == ITTY_EXEC_BUFFER_INVALID_ID ||
                            modulated_buffer == ITTY_EXEC_BUFFER_INVALID_ID)
                                return false;

                        modulated_slices[i] = itty_exec_buffer_get_word_slice (modulated_buffer, 0, modulated_words);
                        if (!itty_exec_buffer_add_xor (plan->exec_buffer,
                                                       modulated_slices[i],
                                                       itty_exec_buffer_get_word_slice (input_buffer, 0, input_words),
                                                       itty_exec_buffer_get_word_slice (mask_buffer, 0, mask_words)))
                                return false;
                }
                plan->max_modulated_words_by_node[node_index] = max_modulated_words;
        }

        return true;
}

static bool
build_feed_layer_condense_stage (itty_network_feed_layer_plan_t *plan)
{
        itty_exec_buffer_begin_named_stage (plan->exec_buffer, ITTY_NETWORK_STAGE_CONDENSE_NODES);
        for (size_t node_index = 0; node_index < plan->number_of_nodes; node_index++) {
                if (plan->modulated_counts[node_index] == 0)
                        continue;

                itty_exec_buffer_id_t condensed_buffer = itty_exec_buffer_allocate_words (plan->exec_buffer,
                                                                                          plan->max_modulated_words_by_node[node_index],
                                                                                          ITTY_NETWORK_BUFFER_CONDENSED_OUTPUT);
                if (condensed_buffer == ITTY_EXEC_BUFFER_INVALID_ID)
                        return false;

                plan->condensed_slices[node_index] = itty_exec_buffer_get_word_slice (condensed_buffer,
                                                                                      0,
                                                                                      plan->max_modulated_words_by_node[node_index]);
                if (!itty_exec_buffer_add_condense (plan->exec_buffer,
                                                    plan->condensed_slices[node_index],
                                                    plan->modulated_slices_by_node[node_index],
                                                    plan->modulated_counts[node_index]))
                        return false;
        }

        return true;
}

static bool
build_feed_layer_output_stage (itty_network_feed_layer_plan_t *plan)
{
        itty_exec_buffer_begin_named_stage (plan->exec_buffer, ITTY_NETWORK_STAGE_DOUBLE_OUTPUTS);
        for (size_t node_index = 0; node_index < plan->number_of_nodes; node_index++) {
                itty_bit_string_t *output = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
                if (!output)
                        return false;

                if (plan->modulated_counts[node_index] == 0) {
                        itty_bit_string_list_append (plan->layer_outputs, output);
                        continue;
                }

                itty_bit_string_append_zeros (output, plan->max_modulated_words_by_node[node_index] * 2);
                itty_exec_buffer_id_t output_buffer = itty_exec_buffer_register_bit_string (plan->exec_buffer,
                                                                                            output,
                                                                                            ITTY_EXEC_BUFFER_ACCESS_READ_WRITE,
                                                                                            ITTY_NETWORK_BUFFER_LAYER_OUTPUT);
                if (output_buffer == ITTY_EXEC_BUFFER_INVALID_ID ||
                    !itty_exec_buffer_add_double (plan->exec_buffer,
                                                  itty_exec_buffer_get_word_slice (output_buffer, 0, itty_bit_string_get_number_of_words (output)),
                                                  plan->condensed_slices[node_index])) {
                        itty_bit_string_free (output);
                        return false;
                }

                itty_bit_string_list_append (plan->layer_outputs, output);
        }

        return true;
}

static itty_network_feed_layer_plan_t *
itty_network_feed_layer_plan_new (itty_network_layer_t   *layer,
                                  itty_bit_string_list_t *current_input)
{
        itty_network_feed_layer_plan_t *plan = calloc (1, sizeof (itty_network_feed_layer_plan_t));
        if (!plan)
                return NULL;

        plan->number_of_nodes = layer->number_of_nodes;
        plan->layer_outputs = itty_bit_string_list_new ();
        plan->exec_buffer = itty_exec_buffer_new ();

        if (plan->number_of_nodes > 0) {
                plan->modulated_slices_by_node = calloc (plan->number_of_nodes, sizeof (itty_exec_buffer_slice_t *));
                plan->modulated_counts = calloc (plan->number_of_nodes, sizeof (size_t));
                plan->max_modulated_words_by_node = calloc (plan->number_of_nodes, sizeof (size_t));
                plan->condensed_slices = calloc (plan->number_of_nodes, sizeof (itty_exec_buffer_slice_t));
        }

        if (!plan->layer_outputs || !plan->exec_buffer ||
            (plan->number_of_nodes > 0 &&
             (!plan->modulated_slices_by_node ||
              !plan->modulated_counts ||
              !plan->max_modulated_words_by_node ||
              !plan->condensed_slices)) ||
            !build_feed_layer_modulation_stage (plan, layer, current_input) ||
            !build_feed_layer_condense_stage (plan) ||
            !build_feed_layer_output_stage (plan)) {
                itty_network_feed_layer_plan_free (plan, true);
                return NULL;
        }

        return plan;
}

static itty_network_layer_runner_kind_t
itty_network_layer_get_runner_kind (itty_network_layer_t *layer)
{
        for (size_t i = 0; i < layer->number_of_nodes; i++) {
                if (layer->nodes[i]->kind != ITTY_NETWORK_NODE_FEED)
                        return ITTY_NETWORK_LAYER_RUNNER_MIXED;
                if (layer->nodes[i]->feed.rotation != 0)
                        return ITTY_NETWORK_LAYER_RUNNER_MIXED;
        }

        return ITTY_NETWORK_LAYER_RUNNER_FEED_PLAN;
}

static itty_bit_string_list_t *
itty_network_layer_run_mixed_with_manager (itty_network_layer_t   *layer,
                                           itty_bit_string_list_t *current_input,
                                           itty_manager_t         *manager)
{
        itty_bit_string_list_t *layer_outputs = itty_bit_string_list_new ();
        if (!layer_outputs)
                return NULL;

        for (size_t node_index = 0; node_index < layer->number_of_nodes; node_index++) {
                itty_network_node_t *node = layer->nodes[node_index];

                if (node->kind == ITTY_NETWORK_NODE_FEED) {
                        itty_bit_string_t *output = itty_network_feed_node_run (node, current_input);
                        if (!output) {
                                itty_bit_string_list_free (layer_outputs);
                                return NULL;
                        }
                        itty_bit_string_list_append (layer_outputs, output);
                } else {
                        itty_bit_string_list_t *outputs = itty_affinity_probe_list_with_manager (node->affinity.affinity,
                                                                                                  current_input,
                                                                                                  &node->affinity.options,
                                                                                                  manager);
                        if (!outputs) {
                                itty_bit_string_list_free (layer_outputs);
                                return NULL;
                        }

                        itty_bit_string_list_append_steal_all (layer_outputs, outputs);
                        itty_bit_string_list_free (outputs);
                }
        }

        return layer_outputs;
}

static itty_bit_string_list_t *
itty_network_feed_layer_run_with_manager (itty_network_layer_t   *layer,
                                          itty_bit_string_list_t *current_input,
                                          itty_manager_t         *manager)
{
        itty_network_feed_layer_plan_t *plan = itty_network_feed_layer_plan_new (layer, current_input);
        if (!plan)
                return NULL;

        if (!itty_exec_buffer_run_with_manager (plan->exec_buffer, manager)) {
                itty_network_feed_layer_plan_free (plan, true);
                return NULL;
        }

        itty_bit_string_list_t *layer_outputs = plan->layer_outputs;
        plan->layer_outputs = NULL;
        itty_network_feed_layer_plan_free (plan, false);

        return layer_outputs;
}

static itty_bit_string_list_t *
itty_network_layer_run_with_manager (itty_network_layer_t   *layer,
                                     itty_bit_string_list_t *current_input,
                                     itty_manager_t         *manager)
{
        switch (itty_network_layer_get_runner_kind (layer)) {
        case ITTY_NETWORK_LAYER_RUNNER_FEED_PLAN:
                return itty_network_feed_layer_run_with_manager (layer, current_input, manager);
        case ITTY_NETWORK_LAYER_RUNNER_MIXED:
                return itty_network_layer_run_mixed_with_manager (layer, current_input, manager);
        }

        return NULL;
}

static bool
append_to_network_plan_presentation (char       **presentation,
                                     size_t      *presentation_length,
                                     const char  *text)
{
        size_t text_length = strlen (text);
        char *new_presentation = realloc (*presentation, *presentation_length + text_length + 1);
        if (!new_presentation)
                return false;

        memcpy (new_presentation + *presentation_length, text, text_length + 1);
        *presentation = new_presentation;
        *presentation_length += text_length;

        return true;
}

static size_t
network_affinity_score_bit_length_for_input (itty_network_node_t      *node,
                                             itty_bit_string_list_t  *current_input)
{
        if (node->affinity.options.score_bit_length != 0)
                return node->affinity.options.score_bit_length;

        size_t score_bit_length = itty_bit_string_list_get_bit_length (current_input);
        if (score_bit_length != 0)
                return score_bit_length;

        for (size_t i = 0; i < itty_bit_string_list_get_length (current_input); i++) {
                itty_bit_string_t *probe = itty_bit_string_list_fetch (current_input, i);
                size_t bit_capacity = itty_bit_string_get_bit_capacity (probe);
                if (bit_capacity > score_bit_length)
                        score_bit_length = bit_capacity;
        }

        return score_bit_length != 0 ? score_bit_length : 1;
}

static bool
append_mixed_network_layer_plan_presentation (char                    **presentation,
                                              size_t                   *presentation_length,
                                              size_t                    layer_index,
                                              itty_network_layer_t     *layer,
                                              itty_bit_string_list_t   *current_input)
{
        char header[80];
        snprintf (header, sizeof (header), "network layer %zu\n", layer_index);
        if (!append_to_network_plan_presentation (presentation, presentation_length, header))
                return false;

        for (size_t node_index = 0; node_index < layer->number_of_nodes; node_index++) {
                itty_network_node_t *node = layer->nodes[node_index];

                if (node->kind == ITTY_NETWORK_NODE_FEED) {
                        snprintf (header, sizeof (header), "network feed node %zu\n", node_index);
                        if (!append_to_network_plan_presentation (presentation, presentation_length, header))
                                return false;
                        continue;
                }

                snprintf (header, sizeof (header), "network affinity node %zu\n", node_index);
                if (!append_to_network_plan_presentation (presentation, presentation_length, header))
                        return false;

                size_t score_bit_length = network_affinity_score_bit_length_for_input (node, current_input);
                itty_affinity_plan_t *affinity_plan = itty_affinity_plan_new (node->affinity.affinity,
                                                                              score_bit_length);
                if (!affinity_plan)
                        return false;

                char *affinity_presentation = itty_affinity_plan_present (affinity_plan);
                itty_affinity_plan_free (affinity_plan);
                if (!affinity_presentation)
                        return false;

                bool appended = append_to_network_plan_presentation (presentation,
                                                                     presentation_length,
                                                                     affinity_presentation);
                free (affinity_presentation);
                if (!appended)
                        return false;
        }

        return true;
}

static bool
append_feed_network_layer_plan_presentation (char                    **presentation,
                                             size_t                   *presentation_length,
                                             size_t                    layer_index,
                                             itty_network_layer_t     *layer,
                                             itty_bit_string_list_t   *current_input,
                                             itty_bit_string_list_t  **layer_outputs)
{
        itty_network_feed_layer_plan_t *plan = itty_network_feed_layer_plan_new (layer,
                                                                                 current_input);
        if (!plan)
                return false;

        char header[64];
        snprintf (header, sizeof (header), "network layer %zu\n", layer_index);
        char *layer_presentation = itty_exec_buffer_present (plan->exec_buffer);
        if (!layer_presentation ||
            !append_to_network_plan_presentation (presentation, presentation_length, header) ||
            !append_to_network_plan_presentation (presentation, presentation_length, layer_presentation)) {
                free (layer_presentation);
                itty_network_feed_layer_plan_free (plan, true);
                return false;
        }
        free (layer_presentation);

        *layer_outputs = plan->layer_outputs;
        plan->layer_outputs = NULL;
        itty_network_feed_layer_plan_free (plan, false);

        return true;
}

static bool
append_network_layer_plan_presentation (char                    **presentation,
                                        size_t                   *presentation_length,
                                        size_t                    layer_index,
                                        itty_network_layer_t     *layer,
                                        itty_bit_string_list_t   *current_input,
                                        itty_bit_string_list_t  **layer_outputs)
{
        *layer_outputs = NULL;

        switch (itty_network_layer_get_runner_kind (layer)) {
        case ITTY_NETWORK_LAYER_RUNNER_FEED_PLAN:
                return append_feed_network_layer_plan_presentation (presentation,
                                                                    presentation_length,
                                                                    layer_index,
                                                                    layer,
                                                                    current_input,
                                                                    layer_outputs);
        case ITTY_NETWORK_LAYER_RUNNER_MIXED:
                if (!append_mixed_network_layer_plan_presentation (presentation,
                                                                   presentation_length,
                                                                   layer_index,
                                                                   layer,
                                                                   current_input))
                        return false;

                *layer_outputs = itty_network_layer_run_mixed_with_manager (layer,
                                                                            current_input,
                                                                            NULL);
                return *layer_outputs != NULL;
        }

        return false;
}

char *
itty_network_layer_present (itty_network_layer_t   *layer,
                            itty_bit_string_list_t *input)
{
        if (!layer || !input)
                return NULL;

        char *presentation = NULL;
        size_t presentation_length = 0;
        itty_bit_string_list_t *layer_outputs = NULL;
        if (!append_network_layer_plan_presentation (&presentation,
                                                    &presentation_length,
                                                    0,
                                                    layer,
                                                    input,
                                                    &layer_outputs)) {
                free (presentation);
                return NULL;
        }

        itty_bit_string_list_free (layer_outputs);

        return presentation;
}

char *
itty_network_present_plan (itty_network_t         *network,
                           itty_bit_string_list_t *input)
{
        if (!network || !input)
                return NULL;

        char *presentation = NULL;
        size_t presentation_length = 0;
        char header[64];

        snprintf (header, sizeof (header), "network plan: layers=%zu\n", network->number_of_layers);
        if (!append_to_network_plan_presentation (&presentation, &presentation_length, header))
                return NULL;

        itty_bit_string_list_t *current_input = input;
        for (size_t layer_index = 0; layer_index < network->number_of_layers; layer_index++) {
                itty_bit_string_list_t *layer_outputs = NULL;
                if (!append_network_layer_plan_presentation (&presentation,
                                                            &presentation_length,
                                                            layer_index,
                                                            network->layers[layer_index],
                                                            current_input,
                                                            &layer_outputs)) {
                        if (current_input != input)
                                itty_bit_string_list_free (current_input);
                        free (presentation);
                        return NULL;
                }

                if (current_input != input)
                        itty_bit_string_list_free (current_input);
                current_input = layer_outputs;
        }

        if (current_input != input)
                itty_bit_string_list_free (current_input);

        return presentation;
}

itty_bit_string_list_t *
itty_network_feed_with_manager (itty_network_t         *network,
                                itty_bit_string_list_t *input,
                                itty_manager_t         *manager)
{
        if (!manager)
                return itty_network_feed (network, input);

        itty_bit_string_list_t *current_input = input;
        itty_network_iterator_t net_iterator;
        itty_network_iterator_init (network, &net_iterator);
        itty_network_layer_t *layer;

        while (itty_network_iterator_next (&net_iterator, &layer)) {
                itty_bit_string_list_t *layer_outputs = itty_network_layer_run_with_manager (layer,
                                                                                             current_input,
                                                                                             manager);
                if (!layer_outputs) {
                        if (current_input != input)
                                itty_bit_string_list_free (current_input);
                        return NULL;
                }

                if (current_input != input)
                        itty_bit_string_list_free (current_input);
                current_input = layer_outputs;
        }

        return current_input;
}

itty_bit_string_list_t *
itty_network_layer_feed (itty_network_layer_t   *layer,
                         itty_bit_string_list_t *input)
{
        if (!layer || !input)
                return NULL;

        return itty_network_layer_run (layer, input);
}

itty_bit_string_list_t *
itty_network_layer_feed_with_manager (itty_network_layer_t   *layer,
                                      itty_bit_string_list_t *input,
                                      itty_manager_t         *manager)
{
        if (!layer || !input)
                return NULL;
        if (!manager)
                return itty_network_layer_feed (layer, input);

        return itty_network_layer_run_with_manager (layer, input, manager);
}

bool
itty_network_select_output (itty_bit_string_list_t *outputs,
                            size_t                 *index)
{
        if (!outputs)
                return false;

        return itty_bit_string_list_popcount_argmax (outputs,
                                                     itty_bit_string_list_get_max_number_of_words (outputs),
                                                     index);
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
        free (network->layers);
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
