#include "itty-vocabulary.h"
#include "itty-bit-string.h"
#include "itty-bit-string-list.h"
#include "itty-bit-string-map.h"
#include "itty-network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void
generate_context (const char *vocabulary_text_file,
                  const char *vocabulary_bit_string_file,
                  const char *context_output_file)
{
        itty_vocabulary_t *vocabulary = itty_vocabulary_new (vocabulary_text_file, vocabulary_bit_string_file);
        if (!vocabulary) {
                fprintf (stderr, "Failed to load vocabulary files\n");
                exit (EXIT_FAILURE);
        }

        char *input_text = NULL;
        size_t len = 0;
        ssize_t read;
        size_t input_text_size = 0;

        while ((read = getline (&input_text, &len, stdin)) != -1) {
                input_text_size += read;
        }

        if (input_text) {
                input_text[input_text_size - 1] = '\0';
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

void
run_inference (const char *vocabulary_text_file,
               const char *vocabulary_bit_string_file,
               const char *inference_model_file,
               const char *context_file,
               size_t number_of_layers,
               size_t nodes_per_layer)
{
        size_t inputs_per_node = nodes_per_layer;

        itty_bit_string_map_file_t *model_map_file = itty_bit_string_map_file_new (inference_model_file);
        itty_bit_string_map_file_t *context_map_file = itty_bit_string_map_file_new (context_file);

        if (!model_map_file || !context_map_file) {
                fprintf (stderr, "Failed to map one or more input files\n");
                exit (EXIT_FAILURE);
        }

        itty_vocabulary_t *vocabulary = itty_vocabulary_new (vocabulary_text_file, vocabulary_bit_string_file);
        if (!vocabulary) {
                fprintf (stderr, "Failed to load vocabulary files\n");
                exit (EXIT_FAILURE);
        }

        printf ("Context: ");
        itty_bit_string_list_t *input_list = itty_bit_string_list_new ();
        itty_bit_string_t *input_bit_string;
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
                fprintf (stderr, "Failed to read any input bit strings from context file.\n");
                exit (EXIT_FAILURE);
        }

        itty_network_t *network = itty_network_new ();

        for (size_t i = 0; i < number_of_layers; i++) {
                itty_bit_string_list_t *bit_string_list;
                size_t number_of_nodes = 0;
                size_t number_of_words = 1 << i;

                itty_network_layer_t *layer = itty_network_layer_new ();
                while (number_of_nodes < nodes_per_layer) {
                        bit_string_list = itty_bit_string_list_new ();
                        for (size_t j = 0; j < inputs_per_node; j++) {
                                itty_bit_string_t *bit_string = itty_bit_string_map_file_next (model_map_file, number_of_words);
                                if (!bit_string) {
                                        fprintf (stderr, "Model insufficient size\n");
                                        exit (EXIT_FAILURE);
                                }

                                itty_bit_string_list_append (bit_string_list, bit_string);
                        }

                        itty_network_node_t *node = itty_network_node_new (bit_string_list);
                        itty_network_layer_append (layer, node);
                        number_of_nodes++;
                }

                itty_network_append (network, layer);
        }

        size_t index;
        itty_bit_string_list_t *output_list = itty_network_feed (network, input_list);
        itty_bit_string_list_popcount_argmax (output_list, itty_bit_string_list_get_max_number_of_words (output_list), &index);

        printf ("Output bit strings:\n");
        itty_bit_string_list_iterator_t iterator;
        itty_bit_string_list_iterator_init (output_list, &iterator);
        itty_bit_string_t *current_bit_string;
        size_t i = 0;
        while (itty_bit_string_list_iterator_next (&iterator, &current_bit_string)) {
                char *text = itty_vocabulary_translate_to_text (vocabulary, current_bit_string);
                if (text) {
                        printf ("%s%s\n",
                                index == i ? "❯ " : "  ",
                                text);
                        free (text);
                }
                i++;
        }

        itty_bit_string_list_free (output_list);
        itty_bit_string_list_free (input_list);
        itty_bit_string_map_file_free (model_map_file);
        itty_bit_string_map_file_free (context_map_file);
        itty_network_free (network);
        itty_vocabulary_free (vocabulary);
}

int
main (int    argc,
      char **argv)
{
        if (argc == 4) {
                const char *vocabulary_text_file = argv[1];
                const char *vocabulary_bit_string_file = argv[2];
                const char *context_output_file = argv[3];
                generate_context (vocabulary_text_file, vocabulary_bit_string_file, context_output_file);
                return EXIT_SUCCESS;
        }

        if (argc == 7) {
                const char *vocabulary_text_file = argv[1];
                const char *vocabulary_bit_string_file = argv[2];
                const char *inference_model_file = argv[3];
                const char *context_file = argv[4];
                size_t number_of_layers = atoi (argv[5]);
                size_t nodes_per_layer = atoi (argv[6]);
                run_inference (vocabulary_text_file, vocabulary_bit_string_file, inference_model_file, context_file, number_of_layers, nodes_per_layer);
                return EXIT_SUCCESS;
        }

        fprintf (stderr, "Usage: %s <vocabulary_text_file> <vocabulary_bit_string_file> <context_output_file> | <vocabulary_text_file> <vocabulary_bit_string_file> <inference_model_file> <context_file> <number_of_layers> <nodes_per_layer>\n", argv[0]);
        return EXIT_FAILURE;
}

