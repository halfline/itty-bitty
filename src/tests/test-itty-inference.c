#include "itty-bit-string.h"
#include "itty-bit-string-list.h"
#include "itty-inference.h"
#include "itty-manager.h"
#include "itty-network.h"
#include "itty-vocabulary.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char *
create_temp_file (const char *content,
                  size_t      size)
{
        char *filename = strdup ("/tmp/test-XXXXXX");
        int fd = mkstemp (filename);
        assert (fd != -1);
        assert (write (fd, content, size) == (ssize_t) size);
        close (fd);
        return filename;
}

static itty_bit_string_t *
create_bit_string (size_t word)
{
        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        itty_bit_string_append_word (bit_string, word);
        return bit_string;
}

static itty_vocabulary_t *
create_vocabulary (char **text_file,
                   char **bit_string_file)
{
        const char *text_content = " apple\n banana\n cherry\n";
        size_t bit_string_words[] = { 0x01, 0x0a, 0x0c };

        *text_file = create_temp_file (text_content, strlen (text_content));
        *bit_string_file = create_temp_file ((char *) bit_string_words, sizeof (bit_string_words));

        return itty_vocabulary_new (*text_file, *bit_string_file);
}

static itty_network_t *
create_feed_network (void)
{
        itty_network_t *network = itty_network_new ();
        itty_network_layer_t *layer = itty_network_layer_new ();
        itty_bit_string_list_t *masks = itty_bit_string_list_new ();

        itty_bit_string_list_append (masks, create_bit_string (0));
        itty_network_layer_append (layer, itty_network_feed_node_new (masks));
        itty_network_append (network, layer);

        return network;
}

static itty_bit_string_list_t *
create_input (void)
{
        itty_bit_string_list_t *input = itty_bit_string_list_new ();
        itty_bit_string_list_append (input, create_bit_string (0x0a));
        return input;
}

static void
test_itty_inference_run_decodes_selected_activation (void)
{
        char *text_file = NULL;
        char *bit_string_file = NULL;
        itty_vocabulary_t *vocabulary = create_vocabulary (&text_file, &bit_string_file);
        itty_network_t *network = create_feed_network ();
        itty_bit_string_list_t *input = create_input ();
        itty_manager_t *manager = itty_manager_new ();

        itty_inference_result_t *result = itty_inference_run (network,
                                                              input,
                                                              vocabulary,
                                                              manager);
        assert (result != NULL);
        assert (strcmp (itty_inference_result_get_text (result), " banana") == 0);
        assert (itty_inference_result_get_distance (result) == 0);
        assert (itty_inference_result_get_selected_index (result) == 0);
        assert (itty_bit_string_list_get_length (itty_inference_result_get_outputs (result)) == 1);
        assert (itty_inference_result_get_selected_activation (result) != NULL);

        itty_inference_result_free (result);
        itty_manager_free (manager);
        itty_bit_string_list_free (input);
        itty_network_free (network);
        itty_vocabulary_free (vocabulary);
        remove (text_file);
        remove (bit_string_file);
        free (text_file);
        free (bit_string_file);
}

int
main (void)
{
        test_itty_inference_run_decodes_selected_activation ();
        printf ("All itty-inference tests passed.\n");
        return 0;
}
