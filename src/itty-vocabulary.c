#include "itty-vocabulary.h"
#include "itty-bit-string-list.h"
#include "itty-bit-string-map.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct itty_vocabulary_t {
        itty_bit_string_map_file_t *bit_string_map;
        char **texts;
        itty_bit_string_list_t *bit_strings;
        size_t count;
};

itty_vocabulary_t *
itty_vocabulary_new (const char *text_file,
                     const char *bit_string_file)
{
        FILE *fp = fopen (text_file, "r");
        if (!fp) {
                return NULL;
        }

        itty_bit_string_map_file_t *bit_string_map = itty_bit_string_map_file_new (bit_string_file);
        if (!bit_string_map) {
                fclose (fp);
                return NULL;
        }

        itty_vocabulary_t *vocabulary = malloc (sizeof (itty_vocabulary_t));

        vocabulary->bit_string_map = bit_string_map;

        vocabulary->texts = NULL;
        vocabulary->bit_strings = itty_bit_string_list_new ();
        vocabulary->count = 0;

        char *line = NULL;
        size_t len = 0;
        ssize_t read;

        while ((read = getline (&line, &len, fp)) != -1) {
                line[strcspn (line, "\n")] = '\0';
                itty_bit_string_t *bit_string = itty_bit_string_map_file_next (bit_string_map, 1);
                if (!bit_string) {
                        break;
                }
                vocabulary->texts = realloc (vocabulary->texts, (vocabulary->count + 1) * sizeof (char *));
                vocabulary->texts[vocabulary->count] = strdup (line);
                itty_bit_string_list_append (vocabulary->bit_strings, bit_string);
                vocabulary->count++;
        }

        free (line);
        fclose (fp);

        return vocabulary;
}

void
itty_vocabulary_free (itty_vocabulary_t *vocabulary)
{
        if (!vocabulary) {
                return;
        }
        for (size_t i = 0; i < vocabulary->count; i++) {
                free (vocabulary->texts[i]);
        }
        free (vocabulary->texts);
        itty_bit_string_list_free (vocabulary->bit_strings);
        itty_bit_string_map_file_free (vocabulary->bit_string_map);
        free (vocabulary);
}

itty_bit_string_t *
itty_vocabulary_translate_to_bit_string (itty_vocabulary_t *vocabulary,
                                         const char        *text)
{
        size_t text_length = strlen (text);
        size_t longest_match_length = 0;
        itty_bit_string_t *best_match = NULL;

        for (size_t i = 0; i < vocabulary->count; i++) {
                size_t entry_length = strlen (vocabulary->texts[i]);

                for (size_t j = 0; j < text_length; j++) {
                        if (strncmp (&text[j], vocabulary->texts[i], entry_length) == 0) {
                                if (entry_length > longest_match_length) {
                                        longest_match_length = entry_length;
                                        best_match = itty_bit_string_list_fetch (vocabulary->bit_strings, i);
                                }
                        }
                }
        }

        return best_match;
}

char *
itty_vocabulary_translate_to_text (itty_vocabulary_t *vocabulary,
                                   itty_bit_string_t *bit_string)
{
        for (size_t i = 0; i < vocabulary->count; i++) {
                itty_bit_string_t *current_bit_string = itty_bit_string_list_fetch (vocabulary->bit_strings, i);
                if (current_bit_string && itty_bit_string_compare (current_bit_string, bit_string) == 0) {
                        return strdup (vocabulary->texts[i]);
                }
        }
        return NULL;
}

bool
itty_vocabulary_write_to_file (itty_vocabulary_t *vocabulary,
                               const char        *input_text,
                               const char        *output_file)
{
        size_t max_size = strlen (input_text) * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES;

        itty_bit_string_map_file_t *output_map_file = itty_bit_string_map_file_new (output_file);

        if (!output_map_file) {
                return false;
        }

        itty_bit_string_map_file_resize (output_map_file, max_size);

        void *mapped_data = itty_bit_string_map_file_get_mapped_data (output_map_file);
        if (mapped_data == NULL) {
                itty_bit_string_map_file_free (output_map_file);
                return false;
        }

        size_t offset = 0;
        const char *position = input_text;
        while (*position) {
                for (size_t i = 0; i < vocabulary->count; i++) {
                        size_t len = strlen (vocabulary->texts[i]);
                        if (strncmp (position, vocabulary->texts[i], len) == 0) {
                                itty_bit_string_t *bit_string = itty_vocabulary_translate_to_bit_string (vocabulary, vocabulary->texts[i]);
                                if (bit_string) {
                                        void *words = itty_bit_string_get_words (bit_string);
                                        size_t number_of_words = itty_bit_string_get_number_of_words (bit_string);
                                        memcpy (((char *) mapped_data) + offset, words, number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
                                        offset += number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES;
                                }
                                position += len;
                                break;
                        }
                }
        }

        itty_bit_string_map_file_resize (output_map_file, offset);
        itty_bit_string_map_file_free (output_map_file);

        return true;
}

