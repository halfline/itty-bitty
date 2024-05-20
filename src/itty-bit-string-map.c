#include "itty-bit-string-map.h"
#include "itty-bit-string-map-private.h"
#include "itty-bit-string.h"
#include "itty-bit-string-private.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

itty_bit_string_map_file_t *
itty_bit_string_map_file_new (const char *file_name)
{
        itty_bit_string_map_file_t *mapped_file = malloc (sizeof (itty_bit_string_map_file_t));

        mapped_file->mapped_data = MAP_FAILED;
        mapped_file->fd = open (file_name, O_RDWR | O_CREAT, 0644);
        if (mapped_file->fd == -1) {
                free (mapped_file);
                return NULL;
        }

        struct stat sb;
        if (fstat (mapped_file->fd, &sb) == -1) {
                close (mapped_file->fd);
                free (mapped_file);
                return NULL;
        }

        if (sb.st_size != 0) {
                mapped_file->file_size = sb.st_size;
                mapped_file->mapped_data = mmap (NULL, mapped_file->file_size, PROT_READ | PROT_WRITE, MAP_SHARED, mapped_file->fd, 0);
                if (mapped_file->mapped_data == MAP_FAILED) {
                        close (mapped_file->fd);
                        free (mapped_file);
                        return NULL;
                }

        }
        mapped_file->current_index = 0;
        mapped_file->bit_string_list = itty_bit_string_list_new ();

        return mapped_file;
}

void
itty_bit_string_map_file_free (itty_bit_string_map_file_t *mapped_file)
{
        if (!mapped_file) {
                return;
        }

        itty_bit_string_list_iterator_t iterator;
        itty_bit_string_list_iterator_init (mapped_file->bit_string_list, &iterator);
        itty_bit_string_t *bit_string;
        while (itty_bit_string_list_iterator_next (&iterator, &bit_string)) {
                bit_string->words = NULL;  // Clear the words pointer
        }

        itty_bit_string_list_free (mapped_file->bit_string_list);
        munmap (mapped_file->mapped_data, mapped_file->file_size);
        close (mapped_file->fd);
        free (mapped_file);
}

itty_bit_string_t *
itty_bit_string_map_file_next (itty_bit_string_map_file_t  *mapped_file,
                               size_t                       number_of_words)
{
        size_t total_words = mapped_file->file_size / ITTY_BIT_STRING_WORD_SIZE_IN_BYTES;
        if (mapped_file->current_index >= total_words) {
                return NULL;
        }

        itty_bit_string_t *bit_string = itty_bit_string_new ();
        bit_string->words = (size_t *) (mapped_file->mapped_data) + mapped_file->current_index;
        bit_string->number_of_words = number_of_words;
        bit_string->pop_count_computed = false;

        mapped_file->current_index += bit_string->number_of_words;

        return bit_string;
}

char *
itty_bit_string_map_file_get_mapped_data (itty_bit_string_map_file_t *mapped_file)
{
        if (mapped_file->mapped_data == MAP_FAILED)
                return NULL;

        return mapped_file->mapped_data;
}

bool
itty_bit_string_map_file_resize (itty_bit_string_map_file_t *mapped_file,
                                 size_t                      new_size)
{
        if (mapped_file->file_size > 0 && mapped_file->file_size > new_size) {
                if (mapped_file->mapped_data != MAP_FAILED)
                    mapped_file->mapped_data = mremap (mapped_file->mapped_data, mapped_file->file_size, new_size, MREMAP_MAYMOVE);
                else
                    mapped_file->mapped_data = mmap (NULL, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, mapped_file->fd, 0);

                if (mapped_file->mapped_data == MAP_FAILED)
                        return false;
        }

        if (ftruncate (mapped_file->fd, new_size) < 0)
                return false;

        if (mapped_file->file_size < new_size && new_size > 0) {
                mremap (mapped_file->mapped_data, mapped_file->file_size, new_size, 0);
        } else if (new_size == 0 && mapped_file->file_size != 0) {
                munmap (mapped_file->mapped_data, mapped_file->file_size);
                mapped_file->mapped_data = MAP_FAILED;
        }

        mapped_file->file_size = new_size;

        return true;
}

