#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "itty-bit-string.h"
#include "itty-bit-string-private.h"
#include "itty-bit-string-map.h"
#include "itty-bit-string-map-private.h"

void
test_itty_bit_string_map_file_new_and_free (void)
{
        const char *file_name = "testfile.bin";
        FILE *file = fopen (file_name, "w");
        fprintf (file, "Test data");
        fclose (file);

        itty_bit_string_map_file_t *mapped_file = itty_bit_string_map_file_new (file_name);
        assert (mapped_file != NULL);
        itty_bit_string_map_file_free (mapped_file);

        remove (file_name);
}

void
test_itty_bit_string_map_file_next (void)
{
        const char *file_name = "testfile.bin";
        FILE *file = fopen (file_name, "w");
        size_t expected_value = ~0UL;

        fwrite (&expected_value, sizeof (size_t), 1, file);
        fclose (file);

        itty_bit_string_map_file_t *mapped_file = itty_bit_string_map_file_new (file_name);
        assert (mapped_file != NULL);

        itty_bit_string_t *bit_string;
        while ((bit_string = itty_bit_string_map_file_next (mapped_file, 1)) != NULL) {
                size_t *words = (size_t *) itty_bit_string_get_words (bit_string);
                assert (words[0] == expected_value);
                expected_value++;
                itty_bit_string_free (bit_string);
        }

        itty_bit_string_map_file_free (mapped_file);
        remove (file_name);
}

void
test_itty_bit_string_map_file_resize (void)
{
        const char *file_name = "testfile.bin";
        FILE *file = fopen (file_name, "w");
        fprintf (file, "Test data");
        fclose (file);

        itty_bit_string_map_file_t *mapped_file = itty_bit_string_map_file_new (file_name);
        assert (mapped_file != NULL);

        size_t new_size = 1024;
        bool success = itty_bit_string_map_file_resize (mapped_file, new_size);
        assert (success);
        assert (mapped_file->file_size == new_size);

        itty_bit_string_map_file_free (mapped_file);
        remove (file_name);
}

int
main (void)
{
        test_itty_bit_string_map_file_new_and_free ();
        test_itty_bit_string_map_file_next ();
        test_itty_bit_string_map_file_resize ();

        printf ("All itty-bit-string-map tests passed.\n");
        return 0;
}

