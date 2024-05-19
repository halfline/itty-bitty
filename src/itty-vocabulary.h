#pragma once

#include "itty-bit-string.h"

typedef struct itty_vocabulary_t itty_vocabulary_t;

itty_vocabulary_t *itty_vocabulary_new (const char *text_file,
                                        const char *bit_string_file);

void itty_vocabulary_free (itty_vocabulary_t *vocabulary);

itty_bit_string_t *itty_vocabulary_translate_to_bit_string (itty_vocabulary_t *vocabulary,
                                                            const char *text);

char *itty_vocabulary_translate_to_text (itty_vocabulary_t *vocabulary,
                                         itty_bit_string_t *bit_string);

bool itty_vocabulary_write_to_file (itty_vocabulary_t *vocabulary,
                                    const char        *input_text,
                                    const char        *output_file);
