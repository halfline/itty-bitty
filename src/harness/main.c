#include "itty-bit-string-map.h"
#include "itty-bit-string.h"
#include "itty-decoder.h"
#include "itty-vocabulary.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

static void
print_usage (const char *program_name)
{
        fprintf (stderr,
                 "Usage:\n"
                 "  %s gray-demo <output_dir>\n",
                 program_name);
}

static void
mkdir_or_die (const char *path)
{
        if (mkdir (path, 0755) == 0)
                return;

        if (errno == EEXIST)
                return;

        fprintf (stderr, "Failed to create directory: %s\n", path);
        exit (EXIT_FAILURE);
}

static char *
join_path_or_die (const char *directory,
                  const char *name)
{
        size_t directory_length = strlen (directory);
        size_t name_length = strlen (name);
        size_t needs_separator = directory_length > 0 && directory[directory_length - 1] != '/';
        char *path = malloc (directory_length + needs_separator + name_length + 1);

        if (!path) {
                fprintf (stderr, "Failed to allocate path\n");
                exit (EXIT_FAILURE);
        }

        memcpy (path, directory, directory_length);
        if (needs_separator)
                path[directory_length++] = '/';
        memcpy (path + directory_length, name, name_length);
        path[directory_length + name_length] = '\0';
        return path;
}

static void
write_text_file_or_die (const char *path,
                        const char *text)
{
        FILE *file = fopen (path, "w");

        if (!file) {
                fprintf (stderr, "Failed to open file for writing: %s\n", path);
                exit (EXIT_FAILURE);
        }

        if (fputs (text, file) == EOF) {
                fprintf (stderr, "Failed to write text file: %s\n", path);
                fclose (file);
                exit (EXIT_FAILURE);
        }

        fclose (file);
}

static void
write_word_map_file_or_die (const char *path,
                            size_t     *words,
                            size_t      number_of_words)
{
        itty_bit_string_map_file_t *map_file = itty_bit_string_map_file_new (path);
        char *mapped_data;

        if (!map_file) {
                fprintf (stderr, "Failed to create mapped file: %s\n", path);
                exit (EXIT_FAILURE);
        }

        if (!itty_bit_string_map_file_resize (map_file, number_of_words * sizeof (size_t))) {
                fprintf (stderr, "Failed to resize mapped file: %s\n", path);
                itty_bit_string_map_file_free (map_file);
                exit (EXIT_FAILURE);
        }

        mapped_data = itty_bit_string_map_file_get_mapped_data (map_file);
        if (!mapped_data) {
                fprintf (stderr, "Failed to map file contents: %s\n", path);
                itty_bit_string_map_file_free (map_file);
                exit (EXIT_FAILURE);
        }

        memcpy (mapped_data, words, number_of_words * sizeof (size_t));
        itty_bit_string_map_file_free (map_file);
}

static void
generate_gray_demo_or_die (const char *output_dir)
{
        static const char *token_text =
                "a\n"
                "b\n"
                "c\n"
                "d\n"
                "e\n"
                "f\n"
                "g\n"
                "h\n"
                "i\n"
                "j\n"
                "k\n"
                "l\n"
                "m\n"
                "n\n"
                "o\n"
                "p\n"
                "q\n"
                "r\n"
                "s\n"
                "t\n"
                "u\n"
                "v\n"
                "w\n"
                "x\n"
                "y\n"
                "z\n";
        static const char *context_text = "d";
        size_t header_word = itty_decoder_gray_code (64) & 0xff;
        size_t payload_words[26];
        size_t expected_index = 3;
        size_t expected_payload_word;
        size_t model_words[] = {
                0,
                0, 0,
                0, 0, 0, 0
        };
        size_t number_of_layers = 3;
        size_t nodes_per_layer = 1;
        char *vocabulary_text_path = join_path_or_die (output_dir, "vocabulary.txt");
        char *vocabulary_bits_path = join_path_or_die (output_dir, "vocabulary.bin");
        char *context_path = join_path_or_die (output_dir, "context.bin");
        char *model_path = join_path_or_die (output_dir, "model.bin");
        char *expected_payload_path = join_path_or_die (output_dir, "expected-payload.bin");
        char *readme_path = join_path_or_die (output_dir, "README.txt");
        FILE *readme;
        itty_vocabulary_t *vocabulary;

        for (size_t i = 0; i < 26; i++)
                payload_words[i] = header_word | (((size_t) 1) << (8 + i));
        expected_payload_word = payload_words[expected_index];

        mkdir_or_die (output_dir);

        write_text_file_or_die (vocabulary_text_path, token_text);
        write_word_map_file_or_die (vocabulary_bits_path, payload_words, 26);
        write_word_map_file_or_die (model_path,
                                    model_words,
                                    sizeof (model_words) / sizeof (model_words[0]));
        write_word_map_file_or_die (expected_payload_path, &expected_payload_word, 1);

        vocabulary = itty_vocabulary_new (vocabulary_text_path, vocabulary_bits_path);
        if (!vocabulary) {
                fprintf (stderr, "Failed to build vocabulary for harness\n");
                exit (EXIT_FAILURE);
        }

        if (!itty_vocabulary_write_to_file (vocabulary, context_text, context_path)) {
                fprintf (stderr, "Failed to write context file\n");
                itty_vocabulary_free (vocabulary);
                exit (EXIT_FAILURE);
        }

        itty_vocabulary_free (vocabulary);

        readme = fopen (readme_path, "w");
        if (!readme) {
                fprintf (stderr, "Failed to write harness readme\n");
                exit (EXIT_FAILURE);
        }

        fprintf (readme,
                 "Generated gray-payload demo files\n"
                 "\n"
                 "Files:\n"
                 "  vocabulary.txt         alphabet token text file\n"
                 "  vocabulary.bin         one payload bit string per letter\n"
                 "  context.bin            encoded input context\n"
                 "  model.bin              three-layer one-node zero mask model\n"
                 "  expected-payload.bin   expected decoded payload\n"
                 "\n"
                 "Example input:\n"
                 "  context text: %s\n"
                 "  expected decoded output: %s\n"
                 "\n"
                 "Network shape:\n"
                 "  layer 1: input adapter\n"
                 "  layer 2: middle layer\n"
                 "  layer 3: final output layer\n"
                 "\n"
                 "Suggested commands:\n"
                 "  itty-bitty infer %s %s %s %s 64 %zu %zu\n"
                 "  itty-bitty-trainer run %s %s %s 64 %zu %zu\n",
                 context_text,
                 context_text,
                 vocabulary_text_path,
                 vocabulary_bits_path,
                 model_path,
                 context_path,
                 number_of_layers,
                 nodes_per_layer,
                 model_path,
                 context_path,
                 expected_payload_path,
                 number_of_layers,
                 nodes_per_layer);
        fclose (readme);

        printf ("Generated gray-payload demo in %s\n", output_dir);
        printf ("Expected token: %s\n", context_text);
        printf ("Expected payload word: 0x%zx\n", expected_payload_word);
        printf ("Model words: %zu zero masks across %zu layers\n",
                sizeof (model_words) / sizeof (model_words[0]),
                number_of_layers);
        printf ("Try:\n");
        printf ("  itty-bitty infer %s %s %s %s 64 %zu %zu\n",
                vocabulary_text_path,
                vocabulary_bits_path,
                model_path,
                context_path,
                number_of_layers,
                nodes_per_layer);
        printf ("  itty-bitty-trainer run %s %s %s 64 %zu %zu\n",
                model_path,
                context_path,
                expected_payload_path,
                number_of_layers,
                nodes_per_layer);

        free (vocabulary_text_path);
        free (vocabulary_bits_path);
        free (context_path);
        free (model_path);
        free (expected_payload_path);
        free (readme_path);
}

int
main (int    argc,
      char **argv)
{
        if (argc == 3 && strcmp (argv[1], "gray-demo") == 0) {
                generate_gray_demo_or_die (argv[2]);
                return EXIT_SUCCESS;
        }

        print_usage (argv[0]);
        return EXIT_FAILURE;
}
