#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "itty-bit-string.c"
#include "itty-bit-string-list.c"
#include "itty-bit-string-map.c"
#include "itty-vocabulary.c"

static char *
create_temp_file (const char *content, size_t size)
{
        char *filename = strdup ("/tmp/test-XXXXXX");
        int fd = mkstemp (filename);
        if (fd == -1) {
                free (filename);
                return NULL;
        }

        if (write (fd, content, size) != size) {
                close (fd);
                free (filename);
                return NULL;
        }

        close (fd);
        return filename;
}

void
test_itty_vocabulary_new (void)
{
        const char *text_content = " apple\n banana\n cherry\n";
        const char bit_string_content[] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                           0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                           0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

        char *text_file = create_temp_file (text_content, strlen (text_content));
        char *bit_string_file = create_temp_file (bit_string_content, sizeof (bit_string_content));

        itty_vocabulary_t *vocabulary = itty_vocabulary_new (text_file, bit_string_file);
        assert (vocabulary != NULL);
        assert (vocabulary->count == 3);

        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

void
test_itty_vocabulary_translate_to_bit_string (void)
{
        const char *text_content = " apple\n banana\n cherry\n";
        const char bit_string_content[] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                           0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                           0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

        char *text_file = create_temp_file (text_content, strlen (text_content));
        char *bit_string_file = create_temp_file (bit_string_content, sizeof (bit_string_content));

        itty_vocabulary_t *vocabulary = itty_vocabulary_new (text_file, bit_string_file);
        assert (vocabulary != NULL);

        itty_bit_string_t *bit_string = itty_vocabulary_translate_to_bit_string (vocabulary, " banana");
        assert (bit_string != NULL);
        char *representation = itty_bit_string_present (bit_string, ITTY_BIT_STRING_PRESENTATION_FORMAT_HEXADECIMAL);
        assert (strcmp (representation, "0000000000000002") == 0);
        free (representation);

        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

void
test_itty_vocabulary_translate_to_text (void)
{
        const char *text_content = " apple\n banana\n cherry\n";
        const char bit_string_content[] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                           0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                           0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

        char *text_file = create_temp_file (text_content, strlen (text_content));
        char *bit_string_file = create_temp_file (bit_string_content, sizeof (bit_string_content));

        itty_vocabulary_t *vocabulary = itty_vocabulary_new (text_file, bit_string_file);
        assert (vocabulary != NULL);

        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        itty_bit_string_append_word (bit_string, 0x02);
        char *text = itty_vocabulary_translate_to_text (vocabulary, bit_string);
        assert (text != NULL);
        assert (strcmp (text, " banana") == 0);
        free (text);

        itty_bit_string_free (bit_string);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

void
test_itty_vocabulary_write_to_file (void)
{
        const char *text_content = " apple\n banana\n cherry\n";
        const char bit_string_content[] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                           0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                           0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

        char *text_file = create_temp_file (text_content, strlen (text_content));
        char *bit_string_file = create_temp_file (bit_string_content, sizeof (bit_string_content));

        itty_vocabulary_t *vocabulary = itty_vocabulary_new (text_file, bit_string_file);
        assert (vocabulary != NULL);

        const char *input_text = " apple banana apple";
        char *output_file = "/tmp/test-output.bin";

        bool result = itty_vocabulary_write_to_file (vocabulary, input_text, output_file);
        assert (result);

        itty_bit_string_map_file_t *output_map_file = itty_bit_string_map_file_new (output_file);
        assert (output_map_file != NULL);

        void *mapped_data = itty_bit_string_map_file_get_mapped_data (output_map_file);
        assert (mapped_data != NULL);

        const char expected_output[] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                        0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

        size_t total_size = sizeof (expected_output);
        assert (memcmp (mapped_data, expected_output, total_size) == 0);

        itty_bit_string_map_file_free (output_map_file);
        remove (output_file);

        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

int
main (void)
{
        test_itty_vocabulary_new ();
        test_itty_vocabulary_translate_to_bit_string ();
        test_itty_vocabulary_translate_to_text ();
        test_itty_vocabulary_write_to_file ();

        printf ("All tests passed.\n");
        return 0;
}

