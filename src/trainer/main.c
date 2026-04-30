#include "itty-bit-string-map.h"
#include "itty-bit-string-list.h"
#include "itty-bit-string.h"
#include "itty-decoder.h"
#include "itty-manager.h"
#include "itty-network.h"
#include "itty-trainer.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
print_usage (const char *program_name)
{
        fprintf (stderr,
                 "Usage:\n"
                 "  %s help\n"
                 "  %s init\n"
                 "  %s run <model_file> <context_file> <expected_payload_file> <payload_bit_count> <number_of_layers> <nodes_per_layer>\n",
                 program_name,
                 program_name,
                 program_name);
}

static size_t
parse_size_t_or_die (const char *label,
                     const char *value)
{
        char *end = NULL;
        unsigned long long parsed_value;

        errno = 0;
        parsed_value = strtoull (value, &end, 10);
        if (errno != 0 || !end || *end != '\0') {
                fprintf (stderr, "Invalid %s: %s\n", label, value);
                exit (EXIT_FAILURE);
        }

        return (size_t) parsed_value;
}

static itty_network_t *
build_feed_network_or_die (itty_bit_string_map_file_t *model_map_file,
                           itty_bit_string_list_t     *input_list,
                           size_t                      number_of_layers,
                           size_t                      nodes_per_layer)
{
        size_t previous_output_count = itty_bit_string_list_get_length (input_list);
        size_t words_per_mask = itty_bit_string_list_get_max_number_of_words (input_list);
        itty_network_t *network = itty_network_new ();

        if (!network) {
                fprintf (stderr, "Failed to create network\n");
                exit (EXIT_FAILURE);
        }

        for (size_t layer_index = 0; layer_index < number_of_layers; layer_index++) {
                size_t layer_node_count = layer_index == 0 ? previous_output_count : nodes_per_layer;
                size_t inputs_per_node = layer_index == 0 ? 1 : previous_output_count;
                itty_network_layer_t *layer = itty_network_layer_new ();

                if (!layer) {
                        fprintf (stderr, "Failed to create network layer\n");
                        itty_network_free (network);
                        exit (EXIT_FAILURE);
                }

                for (size_t node_index = 0; node_index < layer_node_count; node_index++) {
                        itty_network_node_t *node;

                        if (layer_index == 0) {
                                itty_bit_string_t *bit_string = itty_bit_string_map_file_next (model_map_file,
                                                                                                words_per_mask);

                                if (!bit_string) {
                                        fprintf (stderr, "Model file ended before the requested network was filled\n");
                                        itty_network_layer_free (layer);
                                        itty_network_free (network);
                                        exit (EXIT_FAILURE);
                                }

                                node = itty_network_adapter_node_new (bit_string);
                        } else {
                                itty_bit_string_list_t *mask_list = itty_bit_string_list_new ();

                                if (!mask_list) {
                                        fprintf (stderr, "Failed to create mask list\n");
                                        itty_network_layer_free (layer);
                                        itty_network_free (network);
                                        exit (EXIT_FAILURE);
                                }

                                for (size_t input_index = 0; input_index < inputs_per_node; input_index++) {
                                        itty_bit_string_t *bit_string = itty_bit_string_map_file_next (model_map_file,
                                                                                                        words_per_mask);

                                        if (!bit_string) {
                                                fprintf (stderr, "Model file ended before the requested network was filled\n");
                                                itty_bit_string_list_free (mask_list);
                                                itty_network_layer_free (layer);
                                                itty_network_free (network);
                                                exit (EXIT_FAILURE);
                                        }

                                        itty_bit_string_list_append (mask_list, bit_string);
                                }

                                node = itty_network_feed_node_new (mask_list);
                        }

                        if (!node) {
                                fprintf (stderr, "Failed to create network node\n");
                                itty_network_layer_free (layer);
                                itty_network_free (network);
                                exit (EXIT_FAILURE);
                        }

                        itty_network_layer_append (layer, node);
                }

                itty_network_append (network, layer);
                previous_output_count = layer_node_count;
                if (layer_index > 0)
                        words_per_mask *= 2;
        }

        return network;
}

static itty_bit_string_list_t *
load_context_or_die (itty_bit_string_map_file_t *context_map_file)
{
        itty_bit_string_list_t *input_list = itty_bit_string_list_new ();
        itty_bit_string_t *input_bit_string;

        if (!input_list) {
                fprintf (stderr, "Failed to create input list\n");
                exit (EXIT_FAILURE);
        }

        while ((input_bit_string = itty_bit_string_map_file_next (context_map_file, 1)) != NULL)
                itty_bit_string_list_append (input_list, input_bit_string);

        if (itty_bit_string_list_get_length (input_list) == 0) {
                fprintf (stderr, "Failed to read any input bit strings from context file\n");
                itty_bit_string_list_free (input_list);
                exit (EXIT_FAILURE);
        }

        return input_list;
}

static itty_bit_string_t *
load_expected_payload_or_die (itty_bit_string_map_file_t *expected_payload_map_file)
{
        itty_bit_string_t *expected_payload = itty_bit_string_map_file_next (expected_payload_map_file, 1);

        if (!expected_payload) {
                fprintf (stderr, "Failed to read expected payload bit string\n");
                exit (EXIT_FAILURE);
        }

        return expected_payload;
}

static int
run_init (void)
{
        printf ("Trainer scaffold\n");
        printf ("- training commands will be added here\n");
        printf ("- this binary is intended to populate and update stored-entry tables\n");
        printf ("- no training workflow is implemented yet\n");
        return EXIT_SUCCESS;
}

static int
run_training (const char *model_file,
              const char *context_file,
              const char *expected_payload_file,
              size_t      payload_bit_count,
              size_t      number_of_layers,
              size_t      nodes_per_layer)
{
        itty_bit_string_map_file_t *model_map_file = itty_bit_string_map_file_new (model_file);
        itty_bit_string_map_file_t *context_map_file = itty_bit_string_map_file_new (context_file);
        itty_bit_string_map_file_t *expected_payload_map_file = itty_bit_string_map_file_new (expected_payload_file);
        itty_bit_string_list_t *input_list;
        itty_bit_string_t *expected_payload;
        itty_network_t *network;
        itty_manager_t *manager;
        itty_trainer_result_t *result;
        itty_bit_string_list_t *outputs;

        if (!model_map_file || !context_map_file || !expected_payload_map_file) {
                fprintf (stderr, "Failed to map one or more input files\n");
                exit (EXIT_FAILURE);
        }

        input_list = load_context_or_die (context_map_file);
        expected_payload = load_expected_payload_or_die (expected_payload_map_file);
        network = build_feed_network_or_die (model_map_file, input_list, number_of_layers, nodes_per_layer);
        manager = itty_manager_new ();
        if (!manager) {
                fprintf (stderr, "Failed to create work manager\n");
                exit (EXIT_FAILURE);
        }

        result = itty_trainer_run (network,
                                   input_list,
                                   expected_payload,
                                   payload_bit_count,
                                   manager);
        if (!result) {
                fprintf (stderr, "Failed to run trainer\n");
                exit (EXIT_FAILURE);
        }

        outputs = itty_trainer_result_get_outputs (result);
        printf ("Trainer forward pass complete\n");
        printf ("Input bit strings: %zu\n", itty_bit_string_list_get_length (input_list));
        printf ("Output bit strings: %zu\n", itty_bit_string_list_get_length (outputs));
        printf ("Selected output index: %zu\n", itty_trainer_result_get_selected_index (result));
        printf ("Header valid: %s\n", itty_trainer_result_header_valid (result) ? "yes" : "no");
        printf ("Chosen score: %zu\n", itty_trainer_result_get_score_pop_count (result));
        printf ("Chosen payload start: %zu\n", itty_trainer_result_get_payload_start (result));
        printf ("Distance to expected payload: %zu\n", itty_trainer_result_get_payload_distance (result));
        printf ("Payload start is diagnostic hidden state; training update step not implemented yet\n");

        itty_trainer_result_free (result);
        itty_manager_free (manager);
        itty_network_free (network);
        itty_bit_string_list_free (input_list);
        itty_bit_string_free (expected_payload);
        itty_bit_string_map_file_free (model_map_file);
        itty_bit_string_map_file_free (context_map_file);
        itty_bit_string_map_file_free (expected_payload_map_file);
        return EXIT_SUCCESS;
}

int
main (int    argc,
      char **argv)
{
        if (argc == 1 || (argc == 2 && strcmp (argv[1], "help") == 0)) {
                print_usage (argv[0]);
                return argc == 1 ? EXIT_FAILURE : EXIT_SUCCESS;
        }

        if (argc == 2 && strcmp (argv[1], "init") == 0)
                return run_init ();

        if (argc == 8 && strcmp (argv[1], "run") == 0)
                return run_training (argv[2],
                                     argv[3],
                                     argv[4],
                                     parse_size_t_or_die ("payload_bit_count", argv[5]),
                                     parse_size_t_or_die ("number_of_layers", argv[6]),
                                     parse_size_t_or_die ("nodes_per_layer", argv[7]));

        print_usage (argv[0]);
        return EXIT_FAILURE;
}
