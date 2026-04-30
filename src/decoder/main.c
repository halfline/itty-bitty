#include "itty-bit-string-map.h"
#include "itty-bit-string-list.h"
#include "itty-bit-string.h"
#include "itty-decoder.h"
#include "itty-manager.h"
#include "itty-network.h"
#include "itty-vocabulary.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
print_usage (const char *program_name)
{
        fprintf (stderr,
                 "Usage:\n"
                 "  %s context <vocabulary_text_file> <vocabulary_bit_string_file> <context_output_file>\n"
                 "  %s infer <vocabulary_text_file> <vocabulary_bit_string_file> <model_file> <context_file> <payload_bit_count> <number_of_layers> <nodes_per_layer>\n",
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

static void
generate_context (const char *vocabulary_text_file,
                  const char *vocabulary_bit_string_file,
                  const char *context_output_file)
{
        itty_vocabulary_t *vocabulary = itty_vocabulary_new (vocabulary_text_file, vocabulary_bit_string_file);
        char *input_text = NULL;
        size_t input_text_size = 0;
        char *line = NULL;
        size_t len = 0;
        ssize_t read;

        if (!vocabulary) {
                fprintf (stderr, "Failed to load vocabulary files\n");
                exit (EXIT_FAILURE);
        }

        while ((read = getline (&line, &len, stdin)) != -1) {
                char *new_input_text = realloc (input_text, input_text_size + read + 1);

                if (!new_input_text) {
                        fprintf (stderr, "Failed to read input text\n");
                        free (line);
                        free (input_text);
                        itty_vocabulary_free (vocabulary);
                        exit (EXIT_FAILURE);
                }

                input_text = new_input_text;
                memcpy (input_text + input_text_size, line, (size_t) read);
                input_text_size += (size_t) read;
                input_text[input_text_size] = '\0';
        }

        free (line);

        if (input_text_size > 0 && input_text[input_text_size - 1] == '\n') {
                input_text[input_text_size - 1] = '\0';
        } else if (!input_text) {
                input_text = strdup ("");
                if (!input_text) {
                        fprintf (stderr, "Failed to read input text\n");
                        itty_vocabulary_free (vocabulary);
                        exit (EXIT_FAILURE);
                }
        }

        if (!itty_vocabulary_write_to_file (vocabulary, input_text, context_output_file)) {
                fprintf (stderr, "Failed to write context to output file\n");
                itty_vocabulary_free (vocabulary);
                free (input_text);
                exit (EXIT_FAILURE);
        }

        free (input_text);
        itty_vocabulary_free (vocabulary);
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
load_context_or_die (itty_bit_string_map_file_t *context_map_file,
                     itty_vocabulary_t          *vocabulary)
{
        itty_bit_string_list_t *input_list = itty_bit_string_list_new ();
        itty_bit_string_t *input_bit_string;

        if (!input_list) {
            fprintf (stderr, "Failed to create input list\n");
            exit (EXIT_FAILURE);
        }

        printf ("Context: ");
        while ((input_bit_string = itty_bit_string_map_file_next (context_map_file, 1)) != NULL) {
                char *text = itty_vocabulary_translate_to_text (vocabulary, input_bit_string);

                if (text) {
                        printf ("%s", text);
                        free (text);
                }

                itty_bit_string_list_append (input_list, input_bit_string);
        }
        printf ("\n");

        if (itty_bit_string_list_get_length (input_list) == 0) {
                fprintf (stderr, "Failed to read any input bit strings from context file\n");
                itty_bit_string_list_free (input_list);
                exit (EXIT_FAILURE);
        }

        return input_list;
}

static void
run_inference (const char *vocabulary_text_file,
               const char *vocabulary_bit_string_file,
               const char *model_file,
               const char *context_file,
               size_t      payload_bit_count,
               size_t      number_of_layers,
               size_t      nodes_per_layer)
{
        itty_bit_string_map_file_t *model_map_file = itty_bit_string_map_file_new (model_file);
        itty_bit_string_map_file_t *context_map_file = itty_bit_string_map_file_new (context_file);
        itty_vocabulary_t *vocabulary;
        itty_bit_string_list_t *input_list;
        itty_network_t *network;
        itty_manager_t *manager;
        itty_bit_string_list_t *outputs;
        size_t selected_index;
        itty_bit_string_t *selected_activation;
        itty_decoder_result_t *decoder_result;
        itty_bit_string_t *decoded_payload;
        char *text = NULL;
        size_t distance = 0;
        itty_bit_string_list_t *output_list;
        itty_bit_string_list_iterator_t iterator;
        itty_bit_string_t *current_bit_string;
        size_t output_index = 0;

        if (!model_map_file || !context_map_file) {
                fprintf (stderr, "Failed to map one or more input files\n");
                exit (EXIT_FAILURE);
        }

        vocabulary = itty_vocabulary_new (vocabulary_text_file, vocabulary_bit_string_file);
        if (!vocabulary) {
                fprintf (stderr, "Failed to load vocabulary files\n");
                exit (EXIT_FAILURE);
        }

        input_list = load_context_or_die (context_map_file, vocabulary);
        network = build_feed_network_or_die (model_map_file, input_list, number_of_layers, nodes_per_layer);
        manager = itty_manager_new ();
        if (!manager) {
                fprintf (stderr, "Failed to create work manager\n");
                exit (EXIT_FAILURE);
        }

        outputs = itty_network_feed_with_manager (network, input_list, manager);
        if (!outputs || !itty_decoder_select_output (outputs, payload_bit_count, &selected_index)) {
                fprintf (stderr, "Failed to run decoder forward pass\n");
                exit (EXIT_FAILURE);
        }

        selected_activation = itty_bit_string_list_fetch (outputs, selected_index);
        decoder_result = itty_decoder_decode (selected_activation, payload_bit_count);
        if (!decoder_result) {
                fprintf (stderr, "Failed to decode selected activation\n");
                exit (EXIT_FAILURE);
        }

        if (!itty_decoder_result_header_valid (decoder_result)) {
                fprintf (stderr, "Selected activation did not contain a valid payload header\n");
                exit (EXIT_FAILURE);
        }

        decoded_payload = itty_decoder_result_get_payload (decoder_result);
        if (!itty_vocabulary_decode_nearest (vocabulary, decoded_payload, &text, &distance)) {
                fprintf (stderr, "Failed to decode payload through vocabulary\n");
                exit (EXIT_FAILURE);
        }

        printf ("Output: %s\n", text);
        printf ("Output distance: %zu\n", distance);
        printf ("Chosen score: %zu\n", itty_decoder_result_get_score_pop_count (decoder_result));
        printf ("Chosen payload start: %zu\n", itty_decoder_result_get_payload_start (decoder_result));

        output_list = outputs;

        printf ("Output bit strings:\n");
        itty_bit_string_list_iterator_init (output_list, &iterator);
        while (itty_bit_string_list_iterator_next (&iterator, &current_bit_string)) {
                char *text = itty_vocabulary_translate_to_text (vocabulary, current_bit_string);

                if (text) {
                        printf ("%s%s\n",
                                selected_index == output_index ? "❯ " : "  ",
                                text);
                        free (text);
                }

                output_index++;
        }

        free (text);
        itty_decoder_result_free (decoder_result);
        if (outputs != input_list)
                itty_bit_string_list_free (outputs);
        itty_manager_free (manager);
        itty_network_free (network);
        itty_bit_string_list_free (input_list);
        itty_vocabulary_free (vocabulary);
        itty_bit_string_map_file_free (model_map_file);
        itty_bit_string_map_file_free (context_map_file);
}

int
main (int    argc,
      char **argv)
{
        if (argc >= 2 && strcmp (argv[1], "context") == 0) {
                if (argc != 5) {
                        print_usage (argv[0]);
                        return EXIT_FAILURE;
                }

                generate_context (argv[2], argv[3], argv[4]);
                return EXIT_SUCCESS;
        }

        if (argc >= 2 && strcmp (argv[1], "infer") == 0) {
                size_t payload_bit_count;
                size_t number_of_layers;
                size_t nodes_per_layer;

                if (argc != 9) {
                        print_usage (argv[0]);
                        return EXIT_FAILURE;
                }

                payload_bit_count = parse_size_t_or_die ("payload_bit_count", argv[6]);
                number_of_layers = parse_size_t_or_die ("number_of_layers", argv[7]);
                nodes_per_layer = parse_size_t_or_die ("nodes_per_layer", argv[8]);
                run_inference (argv[2], argv[3], argv[4], argv[5], payload_bit_count, number_of_layers, nodes_per_layer);
                return EXIT_SUCCESS;
        }

        print_usage (argv[0]);
        return EXIT_FAILURE;
}
