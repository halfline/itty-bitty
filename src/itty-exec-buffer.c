#include "itty-exec-buffer.h"
#include "itty-bit-string-list.h"
#include "itty-bit-string-list-private.h"
#include "itty-bit-string-private.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
        size_t                     *words;
        size_t                      number_of_words;
        itty_exec_buffer_storage_t  storage;
        itty_exec_buffer_access_t   access;
        itty_bit_string_t          *bit_string;
        char                       *debug_name;
} itty_exec_buffer_descriptor_t;

typedef enum {
        ITTY_EXEC_COMMAND_XOR,
        ITTY_EXEC_COMMAND_XNOR,
        ITTY_EXEC_COMMAND_POPCOUNT,
        ITTY_EXEC_COMMAND_CONDENSE,
        ITTY_EXEC_COMMAND_WEIGHTED_CONDENSE,
        ITTY_EXEC_COMMAND_DOUBLE,
        ITTY_EXEC_COMMAND_CLEAR_ARRAY_RANGE
} itty_exec_command_kind_t;

typedef struct {
        itty_exec_command_kind_t kind;
        union {
                struct {
                        itty_exec_buffer_slice_t destination;
                        itty_exec_buffer_slice_t a;
                        itty_exec_buffer_slice_t b;
                } binary;
                struct {
                        itty_exec_buffer_slice_t destination;
                        size_t                   input_start;
                        size_t                   input_count;
                } condense;
                struct {
                        itty_exec_buffer_slice_t destination;
                        itty_exec_buffer_slice_t votes;
                        size_t                   input_start;
                        size_t                   input_count;
                } weighted_condense;
                struct {
                        itty_exec_buffer_array_t destination;
                        itty_exec_buffer_value_t start;
                        itty_exec_buffer_value_t count;
                } clear_array_range;
        };
} itty_exec_command_t;

typedef struct {
        size_t first_command;
        size_t command_count;
        char  *debug_name;
} itty_exec_stage_t;

typedef struct {
        itty_exec_buffer_t  *exec_buffer;
        itty_exec_command_t *command;
        bool                *success;
} itty_exec_command_task_data_t;

struct itty_exec_buffer_t {
        itty_exec_buffer_descriptor_t *descriptors;
        size_t                         descriptor_count;
        size_t                         descriptor_capacity;

        itty_exec_buffer_slice_t      *slices;
        size_t                         slice_count;
        size_t                         slice_capacity;

        itty_exec_command_t           *commands;
        size_t                         command_count;
        size_t                         command_capacity;

        itty_exec_stage_t             *stages;
        size_t                         stage_count;
        size_t                         stage_capacity;
};

static bool
ensure_descriptor_capacity (itty_exec_buffer_t *exec_buffer)
{
        if (exec_buffer->descriptor_count < exec_buffer->descriptor_capacity)
                return true;

        size_t new_capacity = exec_buffer->descriptor_capacity == 0 ? 16 : exec_buffer->descriptor_capacity * 2;
        itty_exec_buffer_descriptor_t *descriptors = realloc (exec_buffer->descriptors, new_capacity * sizeof (itty_exec_buffer_descriptor_t));
        if (!descriptors)
                return false;

        exec_buffer->descriptors = descriptors;
        exec_buffer->descriptor_capacity = new_capacity;
        return true;
}

static bool
ensure_slice_capacity (itty_exec_buffer_t *exec_buffer,
                       size_t              additional_slices)
{
        if (exec_buffer->slice_count + additional_slices <= exec_buffer->slice_capacity)
                return true;

        size_t new_capacity = exec_buffer->slice_capacity == 0 ? 32 : exec_buffer->slice_capacity * 2;
        while (new_capacity < exec_buffer->slice_count + additional_slices)
                new_capacity *= 2;

        itty_exec_buffer_slice_t *slices = realloc (exec_buffer->slices, new_capacity * sizeof (itty_exec_buffer_slice_t));
        if (!slices)
                return false;

        exec_buffer->slices = slices;
        exec_buffer->slice_capacity = new_capacity;
        return true;
}

static bool
ensure_command_capacity (itty_exec_buffer_t *exec_buffer)
{
        if (exec_buffer->command_count < exec_buffer->command_capacity)
                return true;

        size_t new_capacity = exec_buffer->command_capacity == 0 ? 32 : exec_buffer->command_capacity * 2;
        itty_exec_command_t *commands = realloc (exec_buffer->commands, new_capacity * sizeof (itty_exec_command_t));
        if (!commands)
                return false;

        exec_buffer->commands = commands;
        exec_buffer->command_capacity = new_capacity;
        return true;
}

static bool
ensure_stage_capacity (itty_exec_buffer_t *exec_buffer)
{
        if (exec_buffer->stage_count < exec_buffer->stage_capacity)
                return true;

        size_t new_capacity = exec_buffer->stage_capacity == 0 ? 8 : exec_buffer->stage_capacity * 2;
        itty_exec_stage_t *stages = realloc (exec_buffer->stages, new_capacity * sizeof (itty_exec_stage_t));
        if (!stages)
                return false;

        exec_buffer->stages = stages;
        exec_buffer->stage_capacity = new_capacity;
        return true;
}

static itty_exec_buffer_descriptor_t *
get_descriptor (itty_exec_buffer_t       *exec_buffer,
                itty_exec_buffer_slice_t  slice)
{
        if (slice.buffer_id >= exec_buffer->descriptor_count)
                return NULL;

        itty_exec_buffer_descriptor_t *descriptor = &exec_buffer->descriptors[slice.buffer_id];
        if (slice.word_offset > descriptor->number_of_words ||
            slice.number_of_words > descriptor->number_of_words - slice.word_offset)
                return NULL;

        return descriptor;
}

static size_t *
get_words (itty_exec_buffer_t       *exec_buffer,
           itty_exec_buffer_slice_t  slice)
{
        itty_exec_buffer_descriptor_t *descriptor = get_descriptor (exec_buffer, slice);
        if (!descriptor)
                return NULL;

        return descriptor->words + slice.word_offset;
}

static bool
slice_is_writable (itty_exec_buffer_t       *exec_buffer,
                   itty_exec_buffer_slice_t  slice)
{
        itty_exec_buffer_descriptor_t *descriptor = get_descriptor (exec_buffer, slice);
        return descriptor && descriptor->access == ITTY_EXEC_BUFFER_ACCESS_READ_WRITE;
}

static void
invalidate_slice_cache (itty_exec_buffer_t       *exec_buffer,
                        itty_exec_buffer_slice_t  slice)
{
        itty_exec_buffer_descriptor_t *descriptor = get_descriptor (exec_buffer, slice);
        if (!descriptor || !descriptor->bit_string)
                return;

        descriptor->bit_string->pop_count_computed = false;
}

static size_t
slice_bit_capacity (itty_exec_buffer_slice_t slice)
{
        return slice.number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
}

static size_t
slice_logical_bit_length (itty_exec_buffer_slice_t slice)
{
        return slice.bit_length == 0 ? slice_bit_capacity (slice) : slice.bit_length;
}

static bool
slice_has_valid_bit_length (itty_exec_buffer_slice_t slice)
{
        return slice.bit_length <= slice_bit_capacity (slice);
}

static bool
slice_is_valid (itty_exec_buffer_t       *exec_buffer,
                itty_exec_buffer_slice_t  slice)
{
        return get_descriptor (exec_buffer, slice) && slice_has_valid_bit_length (slice);
}

static bool
slice_is_valid_writable (itty_exec_buffer_t       *exec_buffer,
                         itty_exec_buffer_slice_t  slice)
{
        return slice_is_valid (exec_buffer, slice) && slice_is_writable (exec_buffer, slice);
}

static bool
array_is_valid_writable (itty_exec_buffer_t       *exec_buffer,
                         itty_exec_buffer_array_t  array)
{
        return array.bit_length == 0 && slice_is_valid_writable (exec_buffer, array);
}

static bool
value_is_valid (itty_exec_buffer_t       *exec_buffer,
                itty_exec_buffer_value_t  value)
{
        return value.number_of_words == 1 &&
               value.bit_length == 0 &&
               slice_is_valid (exec_buffer, value);
}

static bool
slices_are_valid (itty_exec_buffer_t             *exec_buffer,
                  itty_exec_buffer_slice_t const *slices,
                  size_t                          slice_count)
{
        for (size_t i = 0; i < slice_count; i++) {
                if (!slice_is_valid (exec_buffer, slices[i]))
                        return false;
        }

        return true;
}

static bool
ensure_command_stage (itty_exec_buffer_t *exec_buffer)
{
        if (!ensure_command_capacity (exec_buffer))
                return false;

        if (exec_buffer->stage_count == 0)
                itty_exec_buffer_begin_stage (exec_buffer);

        return exec_buffer->stage_count > 0;
}

static bool
execute_xor (itty_exec_buffer_t       *exec_buffer,
             itty_exec_buffer_slice_t  destination,
             itty_exec_buffer_slice_t  a,
             itty_exec_buffer_slice_t  b)
{
        if (!slice_is_writable (exec_buffer, destination))
                return false;

        size_t *destination_words = get_words (exec_buffer, destination);
        size_t *a_words = get_words (exec_buffer, a);
        size_t *b_words = get_words (exec_buffer, b);
        if (!destination_words || !a_words || !b_words)
                return false;

        for (size_t i = 0; i < destination.number_of_words; i++) {
                size_t a_word = i < a.number_of_words ? a_words[i] : 0;
                size_t b_word = i < b.number_of_words ? b_words[i] : 0;
                destination_words[i] = a_word ^ b_word;
        }

        invalidate_slice_cache (exec_buffer, destination);

        return true;
}

static bool
execute_xnor (itty_exec_buffer_t       *exec_buffer,
              itty_exec_buffer_slice_t  destination,
              itty_exec_buffer_slice_t  a,
              itty_exec_buffer_slice_t  b)
{
        if (!slice_is_writable (exec_buffer, destination))
                return false;

        size_t *destination_words = get_words (exec_buffer, destination);
        size_t *a_words = get_words (exec_buffer, a);
        size_t *b_words = get_words (exec_buffer, b);
        if (!destination_words || !a_words || !b_words)
                return false;

        for (size_t i = 0; i < destination.number_of_words; i++) {
                size_t a_word = i < a.number_of_words ? a_words[i] : 0;
                size_t b_word = i < b.number_of_words ? b_words[i] : 0;
                destination_words[i] = ~(a_word ^ b_word);
        }

        invalidate_slice_cache (exec_buffer, destination);

        return true;
}

static bool
execute_popcount (itty_exec_buffer_t       *exec_buffer,
                  itty_exec_buffer_slice_t  destination,
                  itty_exec_buffer_slice_t  source)
{
        if (!slice_is_writable (exec_buffer, destination) ||
            destination.number_of_words == 0)
                return false;

        size_t *destination_words = get_words (exec_buffer, destination);
        size_t *source_words = get_words (exec_buffer, source);
        if (!destination_words || !source_words)
                return false;

        size_t bit_length = slice_logical_bit_length (source);
        if (bit_length > slice_bit_capacity (source))
                return false;

        size_t popcount = 0;
        size_t full_words = bit_length / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t trailing_bits = bit_length % ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        for (size_t i = 0; i < full_words; i++)
                popcount += __builtin_popcountl (source_words[i]);
        if (trailing_bits > 0) {
                size_t mask = (1UL << trailing_bits) - 1;
                popcount += __builtin_popcountl (source_words[full_words] & mask);
        }

        memset (destination_words, 0, destination.number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
        destination_words[0] = popcount;

        invalidate_slice_cache (exec_buffer, destination);

        return true;
}

static bool
execute_double (itty_exec_buffer_t       *exec_buffer,
                itty_exec_buffer_slice_t  destination,
                itty_exec_buffer_slice_t  source)
{
        if (!slice_is_writable (exec_buffer, destination) ||
            destination.number_of_words < source.number_of_words * 2)
                return false;

        size_t *destination_words = get_words (exec_buffer, destination);
        size_t *source_words = get_words (exec_buffer, source);
        if (!destination_words || !source_words)
                return false;

        memcpy (destination_words, source_words, source.number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
        memcpy (destination_words + source.number_of_words, source_words, source.number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);

        invalidate_slice_cache (exec_buffer, destination);

        return true;
}

static bool
execute_clear_array_range (itty_exec_buffer_t       *exec_buffer,
                           itty_exec_buffer_array_t  destination,
                           itty_exec_buffer_value_t  start,
                           itty_exec_buffer_value_t  count)
{
        if (!slice_is_writable (exec_buffer, destination))
                return false;

        size_t *destination_words = get_words (exec_buffer, destination);
        size_t *start_words = get_words (exec_buffer, start);
        size_t *count_words = get_words (exec_buffer, count);
        if (!destination_words || !start_words || !count_words)
                return false;

        size_t start_word = start_words[0];
        size_t count_word = count_words[0];
        if (count_word == 0 || start_word >= destination.number_of_words)
                return true;

        size_t available = destination.number_of_words - start_word;
        size_t clear_count = count_word < available ? count_word : available;
        memset (destination_words + start_word, 0, clear_count * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);

        invalidate_slice_cache (exec_buffer, destination);

        return true;
}

static bool
execute_condense (itty_exec_buffer_t       *exec_buffer,
                  itty_exec_buffer_slice_t  destination,
                  size_t                    input_start,
                  size_t                    input_count)
{
        if (!slice_is_writable (exec_buffer, destination) ||
            input_start > exec_buffer->slice_count ||
            input_count > exec_buffer->slice_count - input_start)
                return false;

        size_t *destination_words = get_words (exec_buffer, destination);
        if (!destination_words)
                return false;

        itty_bit_string_list_t *list = itty_bit_string_list_new ();
        if (!list)
                return false;

        for (size_t i = 0; i < input_count; i++) {
                itty_exec_buffer_slice_t slice = exec_buffer->slices[input_start + i];
                size_t *words = get_words (exec_buffer, slice);
                if (!words) {
                        itty_bit_string_list_free (list);
                        return false;
                }

                itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_ONLY);
                if (!bit_string) {
                        itty_bit_string_list_free (list);
                        return false;
                }
                bit_string->words = words;
                bit_string->number_of_words = slice.number_of_words;
                itty_bit_string_list_append (list, bit_string);
        }

        itty_bit_string_t *condensed = itty_bit_string_list_condense (list);
        memset (destination_words, 0, destination.number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);

        if (condensed) {
                size_t words_to_copy = condensed->number_of_words < destination.number_of_words ? condensed->number_of_words : destination.number_of_words;
                memcpy (destination_words, condensed->words, words_to_copy * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
                itty_bit_string_free (condensed);
        }

        itty_bit_string_list_free (list);
        invalidate_slice_cache (exec_buffer, destination);

        return true;
}

static bool
words_get_bit (size_t const *words,
               size_t        number_of_words,
               size_t        bit_index)
{
        size_t word_index = bit_index / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t bit_position = bit_index % ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        if (word_index >= number_of_words)
                return false;

        return (words[word_index] & (1UL << bit_position)) != 0;
}

static void
words_set_bit (size_t *words,
               size_t  bit_index)
{
        size_t word_index = bit_index / ITTY_BIT_STRING_WORD_SIZE_IN_BITS;
        size_t bit_position = bit_index % ITTY_BIT_STRING_WORD_SIZE_IN_BITS;

        words[word_index] |= 1UL << bit_position;
}

static bool
execute_weighted_condense (itty_exec_buffer_t       *exec_buffer,
                           itty_exec_buffer_slice_t  destination,
                           itty_exec_buffer_slice_t  votes,
                           size_t                    input_start,
                           size_t                    input_count)
{
        if (!slice_is_writable (exec_buffer, destination) ||
            input_count == 0 ||
            input_start > exec_buffer->slice_count ||
            input_count > exec_buffer->slice_count - input_start ||
            votes.number_of_words < input_count)
                return false;

        size_t bit_length = slice_logical_bit_length (destination);
        if (bit_length > slice_bit_capacity (destination))
                return false;

        size_t *destination_words = get_words (exec_buffer, destination);
        size_t *vote_words = get_words (exec_buffer, votes);
        if (!destination_words || !vote_words)
                return false;

        memset (destination_words, 0, destination.number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);

        __uint128_t total_votes = 0;
        for (size_t i = 0; i < input_count; i++)
                total_votes += vote_words[i];

        if (total_votes == 0) {
                invalidate_slice_cache (exec_buffer, destination);
                return true;
        }

        __uint128_t majority_threshold = total_votes / 2 + 1;
        for (size_t bit_index = 0; bit_index < bit_length; bit_index++) {
                __uint128_t one_votes = 0;

                for (size_t input_index = 0; input_index < input_count; input_index++) {
                        itty_exec_buffer_slice_t input = exec_buffer->slices[input_start + input_index];
                        size_t *input_words = get_words (exec_buffer, input);
                        if (!input_words)
                                return false;

                        if (words_get_bit (input_words, input.number_of_words, bit_index))
                                one_votes += vote_words[input_index];
                }

                if (one_votes >= majority_threshold)
                        words_set_bit (destination_words, bit_index);
        }

        invalidate_slice_cache (exec_buffer, destination);

        return true;
}

static bool
execute_command (itty_exec_buffer_t *exec_buffer,
                 itty_exec_command_t *command)
{
        switch (command->kind) {
        case ITTY_EXEC_COMMAND_XOR:
                return execute_xor (exec_buffer,
                                    command->binary.destination,
                                    command->binary.a,
                                    command->binary.b);
        case ITTY_EXEC_COMMAND_XNOR:
                return execute_xnor (exec_buffer,
                                     command->binary.destination,
                                     command->binary.a,
                                     command->binary.b);
        case ITTY_EXEC_COMMAND_POPCOUNT:
                return execute_popcount (exec_buffer,
                                         command->binary.destination,
                                         command->binary.a);
        case ITTY_EXEC_COMMAND_CONDENSE:
                return execute_condense (exec_buffer,
                                         command->condense.destination,
                                         command->condense.input_start,
                                         command->condense.input_count);
        case ITTY_EXEC_COMMAND_WEIGHTED_CONDENSE:
                return execute_weighted_condense (exec_buffer,
                                                 command->weighted_condense.destination,
                                                 command->weighted_condense.votes,
                                                 command->weighted_condense.input_start,
                                                 command->weighted_condense.input_count);
        case ITTY_EXEC_COMMAND_DOUBLE:
                return execute_double (exec_buffer,
                                       command->binary.destination,
                                       command->binary.a);
        case ITTY_EXEC_COMMAND_CLEAR_ARRAY_RANGE:
                return execute_clear_array_range (exec_buffer,
                                                  command->clear_array_range.destination,
                                                  command->clear_array_range.start,
                                                  command->clear_array_range.count);
        }

        return false;
}

static void *
execute_command_task (void *data)
{
        itty_exec_command_task_data_t *task_data = data;

        *task_data->success = execute_command (task_data->exec_buffer, task_data->command);

        return NULL;
}

static const char *
command_kind_name (itty_exec_command_kind_t kind)
{
        switch (kind) {
        case ITTY_EXEC_COMMAND_XOR:
                return "XOR";
        case ITTY_EXEC_COMMAND_XNOR:
                return "XNOR";
        case ITTY_EXEC_COMMAND_POPCOUNT:
                return "POPCOUNT";
        case ITTY_EXEC_COMMAND_CONDENSE:
                return "CONDENSE";
        case ITTY_EXEC_COMMAND_WEIGHTED_CONDENSE:
                return "WEIGHTED_CONDENSE";
        case ITTY_EXEC_COMMAND_DOUBLE:
                return "DOUBLE";
        case ITTY_EXEC_COMMAND_CLEAR_ARRAY_RANGE:
                return "CLEAR_ARRAY_RANGE";
        }

        return "UNKNOWN";
}

static const char *
storage_name (itty_exec_buffer_storage_t storage)
{
        switch (storage) {
        case ITTY_EXEC_BUFFER_STORAGE_EXTERNAL:
                return "external";
        case ITTY_EXEC_BUFFER_STORAGE_OWNED:
                return "owned";
        }

        return "unknown";
}

static const char *
access_name (itty_exec_buffer_access_t access)
{
        switch (access) {
        case ITTY_EXEC_BUFFER_ACCESS_READ_ONLY:
                return "read-only";
        case ITTY_EXEC_BUFFER_ACCESS_READ_WRITE:
                return "read-write";
        }

        return "unknown";
}

static void
describe_slice (FILE                    *stream,
                itty_exec_buffer_slice_t slice)
{
        fprintf (stream,
                 "buffer=%zu word_offset=%zu words=%zu bit_length=%zu",
                 slice.buffer_id,
                 slice.word_offset,
                 slice.number_of_words,
                 slice.bit_length);
}

static void
describe_command (FILE                *stream,
                  itty_exec_buffer_t  *exec_buffer,
                  size_t               command_index,
                  itty_exec_command_t *command)
{
        fprintf (stream, "  command %zu: %s ", command_index, command_kind_name (command->kind));

        switch (command->kind) {
        case ITTY_EXEC_COMMAND_XOR:
        case ITTY_EXEC_COMMAND_XNOR:
                fprintf (stream, "destination=(");
                describe_slice (stream, command->binary.destination);
                fprintf (stream, ") a=(");
                describe_slice (stream, command->binary.a);
                fprintf (stream, ") b=(");
                describe_slice (stream, command->binary.b);
                fprintf (stream, ")\n");
                break;
        case ITTY_EXEC_COMMAND_POPCOUNT:
        case ITTY_EXEC_COMMAND_DOUBLE:
                fprintf (stream, "destination=(");
                describe_slice (stream, command->binary.destination);
                fprintf (stream, ") source=(");
                describe_slice (stream, command->binary.a);
                fprintf (stream, ")\n");
                break;
        case ITTY_EXEC_COMMAND_CLEAR_ARRAY_RANGE:
                fprintf (stream, "destination=(");
                describe_slice (stream, command->clear_array_range.destination);
                fprintf (stream, ") start=(");
                describe_slice (stream, command->clear_array_range.start);
                fprintf (stream, ") count=(");
                describe_slice (stream, command->clear_array_range.count);
                fprintf (stream, ")\n");
                break;
        case ITTY_EXEC_COMMAND_CONDENSE:
                fprintf (stream, "destination=(");
                describe_slice (stream, command->condense.destination);
                fprintf (stream, ") inputs=%zu", command->condense.input_count);
                for (size_t i = 0; i < command->condense.input_count; i++) {
                        fprintf (stream, " input[%zu]=(", i);
                        describe_slice (stream, exec_buffer->slices[command->condense.input_start + i]);
                        fprintf (stream, ")");
                }
                fprintf (stream, "\n");
                break;
        case ITTY_EXEC_COMMAND_WEIGHTED_CONDENSE:
                fprintf (stream, "destination=(");
                describe_slice (stream, command->weighted_condense.destination);
                fprintf (stream, ") votes=(");
                describe_slice (stream, command->weighted_condense.votes);
                fprintf (stream, ") inputs=%zu", command->weighted_condense.input_count);
                for (size_t i = 0; i < command->weighted_condense.input_count; i++) {
                        fprintf (stream, " input[%zu]=(", i);
                        describe_slice (stream, exec_buffer->slices[command->weighted_condense.input_start + i]);
                        fprintf (stream, ")");
                }
                fprintf (stream, "\n");
                break;
        }
}

static void
update_stage_command_counts (itty_exec_buffer_t *exec_buffer)
{
        for (size_t i = 0; i < exec_buffer->stage_count; i++) {
                itty_exec_stage_t *stage = &exec_buffer->stages[i];
                size_t next_command = i + 1 < exec_buffer->stage_count ? exec_buffer->stages[i + 1].first_command : exec_buffer->command_count;
                stage->command_count = next_command - stage->first_command;
        }
}

static bool
execute_stage (itty_exec_buffer_t *exec_buffer,
               itty_exec_stage_t  *stage)
{
        for (size_t i = 0; i < stage->command_count; i++) {
                if (!execute_command (exec_buffer, &exec_buffer->commands[stage->first_command + i]))
                        return false;
        }

        return true;
}

static bool
execute_stage_with_manager (itty_exec_buffer_t *exec_buffer,
                            itty_exec_stage_t  *stage,
                            itty_manager_t     *manager)
{
        if (!manager)
                return execute_stage (exec_buffer, stage);

        itty_task_group_t *task_group = itty_manager_create_task_group (manager);
        bool *successes = calloc (stage->command_count, sizeof (bool));
        itty_exec_command_task_data_t *task_data = calloc (stage->command_count, sizeof (itty_exec_command_task_data_t));

        if (!task_group || !successes || !task_data) {
                itty_manager_free_task_group (task_group);
                free (successes);
                free (task_data);
                return false;
        }

        for (size_t i = 0; i < stage->command_count; i++) {
                task_data[i].exec_buffer = exec_buffer;
                task_data[i].command = &exec_buffer->commands[stage->first_command + i];
                task_data[i].success = &successes[i];
                if (!itty_manager_task_group_submit (task_group, execute_command_task, &task_data[i]))
                        successes[i] = execute_command (exec_buffer, &exec_buffer->commands[stage->first_command + i]);
        }

        itty_manager_wait_for_task_group (task_group);

        bool stage_success = true;
        for (size_t i = 0; i < stage->command_count; i++) {
                if (!successes[i])
                        stage_success = false;
        }

        itty_manager_free_task_group (task_group);
        free (successes);
        free (task_data);

        return stage_success;
}

itty_exec_buffer_t *
itty_exec_buffer_new (void)
{
        return calloc (1, sizeof (itty_exec_buffer_t));
}

void
itty_exec_buffer_free (itty_exec_buffer_t *exec_buffer)
{
        if (!exec_buffer)
                return;

        for (size_t i = 0; i < exec_buffer->descriptor_count; i++) {
                if (exec_buffer->descriptors[i].storage == ITTY_EXEC_BUFFER_STORAGE_OWNED)
                        free (exec_buffer->descriptors[i].words);
                free (exec_buffer->descriptors[i].debug_name);
        }

        free (exec_buffer->descriptors);
        for (size_t i = 0; i < exec_buffer->stage_count; i++)
                free (exec_buffer->stages[i].debug_name);
        free (exec_buffer->slices);
        free (exec_buffer->commands);
        free (exec_buffer->stages);
        free (exec_buffer);
}

itty_exec_buffer_id_t
itty_exec_buffer_register_words (itty_exec_buffer_t          *exec_buffer,
                                 size_t                      *words,
                                 size_t                       number_of_words,
                                 itty_exec_buffer_access_t    access,
                                 const char                  *debug_name)
{
        if (!exec_buffer || !words || !ensure_descriptor_capacity (exec_buffer))
                return ITTY_EXEC_BUFFER_INVALID_ID;

        itty_exec_buffer_id_t id = exec_buffer->descriptor_count++;
        exec_buffer->descriptors[id].words = words;
        exec_buffer->descriptors[id].number_of_words = number_of_words;
        exec_buffer->descriptors[id].storage = ITTY_EXEC_BUFFER_STORAGE_EXTERNAL;
        exec_buffer->descriptors[id].access = access;
        exec_buffer->descriptors[id].bit_string = NULL;
        exec_buffer->descriptors[id].debug_name = debug_name ? strdup (debug_name) : NULL;

        return id;
}

itty_exec_buffer_id_t
itty_exec_buffer_register_bit_string (itty_exec_buffer_t          *exec_buffer,
                                      itty_bit_string_t           *bit_string,
                                      itty_exec_buffer_access_t    access,
                                      const char                  *debug_name)
{
        if (!bit_string)
                return ITTY_EXEC_BUFFER_INVALID_ID;

        itty_exec_buffer_id_t id = itty_exec_buffer_register_words (exec_buffer,
                                                                    itty_bit_string_get_words (bit_string),
                                                                    itty_bit_string_get_number_of_words (bit_string),
                                                                    access,
                                                                    debug_name);
        if (id == ITTY_EXEC_BUFFER_INVALID_ID)
                return ITTY_EXEC_BUFFER_INVALID_ID;

        exec_buffer->descriptors[id].bit_string = bit_string;

        return id;
}

bool
itty_exec_buffer_rebind_words (itty_exec_buffer_t       *exec_buffer,
                               itty_exec_buffer_id_t     buffer_id,
                               size_t                   *words,
                               size_t                    number_of_words)
{
        if (!exec_buffer || !words || buffer_id >= exec_buffer->descriptor_count)
                return false;

        itty_exec_buffer_descriptor_t *descriptor = &exec_buffer->descriptors[buffer_id];
        if (descriptor->storage != ITTY_EXEC_BUFFER_STORAGE_EXTERNAL)
                return false;

        descriptor->words = words;
        descriptor->number_of_words = number_of_words;
        descriptor->bit_string = NULL;

        return true;
}

bool
itty_exec_buffer_rebind_bit_string (itty_exec_buffer_t       *exec_buffer,
                                    itty_exec_buffer_id_t     buffer_id,
                                    itty_bit_string_t        *bit_string)
{
        if (!bit_string)
                return false;

        if (!itty_exec_buffer_rebind_words (exec_buffer,
                                            buffer_id,
                                            itty_bit_string_get_words (bit_string),
                                            itty_bit_string_get_number_of_words (bit_string)))
                return false;

        exec_buffer->descriptors[buffer_id].bit_string = bit_string;

        return true;
}

itty_exec_buffer_id_t
itty_exec_buffer_allocate_words (itty_exec_buffer_t *exec_buffer,
                                 size_t              number_of_words,
                                 const char         *debug_name)
{
        if (!exec_buffer || number_of_words == 0 || !ensure_descriptor_capacity (exec_buffer))
                return ITTY_EXEC_BUFFER_INVALID_ID;

        size_t *words = calloc (number_of_words, ITTY_BIT_STRING_WORD_SIZE_IN_BYTES);
        if (!words)
                return ITTY_EXEC_BUFFER_INVALID_ID;

        itty_exec_buffer_id_t id = exec_buffer->descriptor_count++;
        exec_buffer->descriptors[id].words = words;
        exec_buffer->descriptors[id].number_of_words = number_of_words;
        exec_buffer->descriptors[id].storage = ITTY_EXEC_BUFFER_STORAGE_OWNED;
        exec_buffer->descriptors[id].access = ITTY_EXEC_BUFFER_ACCESS_READ_WRITE;
        exec_buffer->descriptors[id].bit_string = NULL;
        exec_buffer->descriptors[id].debug_name = debug_name ? strdup (debug_name) : NULL;

        return id;
}

bool
itty_exec_buffer_find_descriptor (itty_exec_buffer_t    *exec_buffer,
                                  const char            *debug_name,
                                  itty_exec_buffer_id_t *buffer_id)
{
        if (!exec_buffer || !debug_name || !buffer_id)
                return false;

        bool found = false;
        for (size_t i = 0; i < exec_buffer->descriptor_count; i++) {
                if (exec_buffer->descriptors[i].debug_name &&
                    strcmp (exec_buffer->descriptors[i].debug_name, debug_name) == 0) {
                        if (found)
                                return false;
                        *buffer_id = i;
                        found = true;
                }
        }

        return found;
}

itty_exec_buffer_slice_t
itty_exec_buffer_get_word_slice (itty_exec_buffer_id_t buffer_id,
                                 size_t                word_offset,
                                 size_t                number_of_words)
{
        return itty_exec_buffer_get_bit_slice (buffer_id, word_offset, number_of_words, 0);
}

itty_exec_buffer_slice_t
itty_exec_buffer_get_bit_slice (itty_exec_buffer_id_t buffer_id,
                                size_t                word_offset,
                                size_t                number_of_words,
                                size_t                bit_length)
{
        return (itty_exec_buffer_slice_t) {
                buffer_id,
                word_offset,
                number_of_words,
                bit_length
        };
}

itty_exec_buffer_value_t
itty_exec_buffer_get_value (itty_exec_buffer_id_t buffer_id,
                            size_t                word_offset)
{
        return itty_exec_buffer_get_word_slice (buffer_id, word_offset, 1);
}

itty_exec_buffer_array_t
itty_exec_buffer_get_array (itty_exec_buffer_id_t buffer_id,
                            size_t                word_offset,
                            size_t                number_of_words)
{
        return itty_exec_buffer_get_word_slice (buffer_id, word_offset, number_of_words);
}

itty_exec_buffer_bits_t
itty_exec_buffer_get_bits (itty_exec_buffer_id_t buffer_id,
                           size_t                word_offset,
                           size_t                number_of_words,
                           size_t                bit_length)
{
        return itty_exec_buffer_get_bit_slice (buffer_id, word_offset, number_of_words, bit_length);
}

void
itty_exec_buffer_begin_stage (itty_exec_buffer_t *exec_buffer)
{
        itty_exec_buffer_begin_named_stage (exec_buffer, NULL);
}

void
itty_exec_buffer_begin_named_stage (itty_exec_buffer_t *exec_buffer,
                                    const char         *debug_name)
{
        if (!exec_buffer || !ensure_stage_capacity (exec_buffer))
                return;

        if (exec_buffer->stage_count > 0) {
                itty_exec_stage_t *previous_stage = &exec_buffer->stages[exec_buffer->stage_count - 1];
                previous_stage->command_count = exec_buffer->command_count - previous_stage->first_command;
        }

        exec_buffer->stages[exec_buffer->stage_count++] = (itty_exec_stage_t) {
                exec_buffer->command_count,
                0,
                debug_name ? strdup (debug_name) : NULL
        };
}

bool
itty_exec_buffer_add_xor (itty_exec_buffer_t       *exec_buffer,
                          itty_exec_buffer_slice_t  destination,
                          itty_exec_buffer_slice_t  a,
                          itty_exec_buffer_slice_t  b)
{
        if (!exec_buffer ||
            !slice_is_valid_writable (exec_buffer, destination) ||
            !slice_is_valid (exec_buffer, a) ||
            !slice_is_valid (exec_buffer, b) ||
            !ensure_command_stage (exec_buffer))
                return false;

        exec_buffer->commands[exec_buffer->command_count++] = (itty_exec_command_t) {
                ITTY_EXEC_COMMAND_XOR,
                .binary = {
                        destination,
                        a,
                        b
                }
        };

        return true;
}

bool
itty_exec_buffer_add_xnor (itty_exec_buffer_t       *exec_buffer,
                           itty_exec_buffer_slice_t  destination,
                           itty_exec_buffer_slice_t  a,
                           itty_exec_buffer_slice_t  b)
{
        if (!exec_buffer ||
            !slice_is_valid_writable (exec_buffer, destination) ||
            !slice_is_valid (exec_buffer, a) ||
            !slice_is_valid (exec_buffer, b) ||
            !ensure_command_stage (exec_buffer))
                return false;

        exec_buffer->commands[exec_buffer->command_count++] = (itty_exec_command_t) {
                ITTY_EXEC_COMMAND_XNOR,
                .binary = {
                        destination,
                        a,
                        b
                }
        };

        return true;
}

bool
itty_exec_buffer_add_popcount (itty_exec_buffer_t       *exec_buffer,
                               itty_exec_buffer_slice_t  destination,
                               itty_exec_buffer_slice_t  source)
{
        if (!exec_buffer ||
            destination.number_of_words == 0 ||
            !slice_is_valid_writable (exec_buffer, destination) ||
            !slice_is_valid (exec_buffer, source) ||
            !ensure_command_stage (exec_buffer))
                return false;

        exec_buffer->commands[exec_buffer->command_count++] = (itty_exec_command_t) {
                ITTY_EXEC_COMMAND_POPCOUNT,
                .binary = {
                        destination,
                        source,
                        { ITTY_EXEC_BUFFER_INVALID_ID, 0, 0, 0 }
                }
        };

        return true;
}

bool
itty_exec_buffer_add_condense (itty_exec_buffer_t             *exec_buffer,
                               itty_exec_buffer_slice_t        destination,
                               itty_exec_buffer_slice_t const *inputs,
                               size_t                          input_count)
{
        if (!exec_buffer || !inputs || input_count == 0 ||
            !slice_is_valid_writable (exec_buffer, destination) ||
            !slices_are_valid (exec_buffer, inputs, input_count) ||
            !ensure_command_stage (exec_buffer) ||
            !ensure_slice_capacity (exec_buffer, input_count))
                return false;

        size_t input_start = exec_buffer->slice_count;
        memcpy (&exec_buffer->slices[input_start], inputs, input_count * sizeof (itty_exec_buffer_slice_t));
        exec_buffer->slice_count += input_count;

        exec_buffer->commands[exec_buffer->command_count++] = (itty_exec_command_t) {
                ITTY_EXEC_COMMAND_CONDENSE,
                .condense = {
                        destination,
                        input_start,
                        input_count
                }
        };

        return true;
}

bool
itty_exec_buffer_add_weighted_condense (itty_exec_buffer_t             *exec_buffer,
                                        itty_exec_buffer_slice_t        destination,
                                        itty_exec_buffer_slice_t const *inputs,
                                        itty_exec_buffer_slice_t        votes,
                                        size_t                          input_count)
{
        if (!exec_buffer || !inputs || input_count == 0 ||
            !slice_is_valid_writable (exec_buffer, destination) ||
            !slices_are_valid (exec_buffer, inputs, input_count) ||
            !slice_is_valid (exec_buffer, votes) ||
            votes.number_of_words < input_count ||
            !ensure_command_stage (exec_buffer) ||
            !ensure_slice_capacity (exec_buffer, input_count))
                return false;

        size_t input_start = exec_buffer->slice_count;
        memcpy (&exec_buffer->slices[input_start], inputs, input_count * sizeof (itty_exec_buffer_slice_t));
        exec_buffer->slice_count += input_count;

        exec_buffer->commands[exec_buffer->command_count++] = (itty_exec_command_t) {
                ITTY_EXEC_COMMAND_WEIGHTED_CONDENSE,
                .weighted_condense = {
                        destination,
                        votes,
                        input_start,
                        input_count
                }
        };

        return true;
}

bool
itty_exec_buffer_add_double (itty_exec_buffer_t       *exec_buffer,
                             itty_exec_buffer_slice_t  destination,
                             itty_exec_buffer_slice_t  source)
{
        if (!exec_buffer ||
            !slice_is_valid_writable (exec_buffer, destination) ||
            !slice_is_valid (exec_buffer, source) ||
            destination.number_of_words < source.number_of_words * 2 ||
            !ensure_command_stage (exec_buffer))
                return false;

        exec_buffer->commands[exec_buffer->command_count++] = (itty_exec_command_t) {
                ITTY_EXEC_COMMAND_DOUBLE,
                .binary = {
                        destination,
                        source,
                        { ITTY_EXEC_BUFFER_INVALID_ID, 0, 0, 0 }
                }
        };

        return true;
}

bool
itty_exec_buffer_add_clear_array_range (itty_exec_buffer_t        *exec_buffer,
                                        itty_exec_buffer_array_t   destination,
                                        itty_exec_buffer_value_t   start,
                                        itty_exec_buffer_value_t   count)
{
        if (!exec_buffer ||
            !array_is_valid_writable (exec_buffer, destination) ||
            !value_is_valid (exec_buffer, start) ||
            !value_is_valid (exec_buffer, count) ||
            !ensure_command_stage (exec_buffer))
                return false;

        exec_buffer->commands[exec_buffer->command_count++] = (itty_exec_command_t) {
                ITTY_EXEC_COMMAND_CLEAR_ARRAY_RANGE,
                .clear_array_range = {
                        destination,
                        start,
                        count
                }
        };

        return true;
}

bool
itty_exec_buffer_run (itty_exec_buffer_t *exec_buffer)
{
        if (!exec_buffer)
                return false;

        update_stage_command_counts (exec_buffer);

        for (size_t stage_index = 0; stage_index < exec_buffer->stage_count; stage_index++) {
                if (!execute_stage (exec_buffer, &exec_buffer->stages[stage_index]))
                        return false;
        }

        return true;
}

bool
itty_exec_buffer_run_with_manager (itty_exec_buffer_t *exec_buffer,
                                   itty_manager_t     *manager)
{
        if (!manager)
                return itty_exec_buffer_run (exec_buffer);

        if (!exec_buffer)
                return false;

        update_stage_command_counts (exec_buffer);

        for (size_t stage_index = 0; stage_index < exec_buffer->stage_count; stage_index++) {
                if (!execute_stage_with_manager (exec_buffer, &exec_buffer->stages[stage_index], manager))
                        return false;
        }

        return true;
}

size_t
itty_exec_buffer_get_stage_count (itty_exec_buffer_t *exec_buffer)
{
        if (!exec_buffer)
                return 0;

        update_stage_command_counts (exec_buffer);

        return exec_buffer->stage_count;
}

bool
itty_exec_buffer_find_stage (itty_exec_buffer_t *exec_buffer,
                             const char         *debug_name,
                             size_t             *stage_index)
{
        if (!exec_buffer || !debug_name || !stage_index)
                return false;

        update_stage_command_counts (exec_buffer);

        bool found = false;
        for (size_t i = 0; i < exec_buffer->stage_count; i++) {
                if (exec_buffer->stages[i].debug_name &&
                    strcmp (exec_buffer->stages[i].debug_name, debug_name) == 0) {
                        if (found)
                                return false;
                        *stage_index = i;
                        found = true;
                }
        }

        return found;
}

bool
itty_exec_buffer_run_stage (itty_exec_buffer_t *exec_buffer,
                            size_t              stage_index)
{
        if (!exec_buffer || stage_index >= exec_buffer->stage_count)
                return false;

        update_stage_command_counts (exec_buffer);

        return execute_stage (exec_buffer, &exec_buffer->stages[stage_index]);
}

bool
itty_exec_buffer_run_stage_with_manager (itty_exec_buffer_t *exec_buffer,
                                         size_t              stage_index,
                                         itty_manager_t     *manager)
{
        if (!manager)
                return itty_exec_buffer_run_stage (exec_buffer, stage_index);

        if (!exec_buffer || stage_index >= exec_buffer->stage_count)
                return false;

        update_stage_command_counts (exec_buffer);

        return execute_stage_with_manager (exec_buffer, &exec_buffer->stages[stage_index], manager);
}

bool
itty_exec_buffer_run_named_stage (itty_exec_buffer_t *exec_buffer,
                                  const char         *debug_name)
{
        size_t stage_index = 0;
        if (!itty_exec_buffer_find_stage (exec_buffer, debug_name, &stage_index))
                return false;

        return itty_exec_buffer_run_stage (exec_buffer, stage_index);
}

bool
itty_exec_buffer_run_named_stage_with_manager (itty_exec_buffer_t *exec_buffer,
                                               const char         *debug_name,
                                               itty_manager_t     *manager)
{
        size_t stage_index = 0;
        if (!itty_exec_buffer_find_stage (exec_buffer, debug_name, &stage_index))
                return false;

        return itty_exec_buffer_run_stage_with_manager (exec_buffer, stage_index, manager);
}

char *
itty_exec_buffer_present (itty_exec_buffer_t *exec_buffer)
{
        if (!exec_buffer)
                return NULL;

        update_stage_command_counts (exec_buffer);

        char *description = NULL;
        size_t description_length = 0;
        FILE *stream = open_memstream (&description, &description_length);
        if (!stream)
                return NULL;

        fprintf (stream,
                 "exec_buffer descriptors=%zu stages=%zu commands=%zu\n",
                 exec_buffer->descriptor_count,
                 exec_buffer->stage_count,
                 exec_buffer->command_count);

        for (size_t i = 0; i < exec_buffer->descriptor_count; i++) {
                itty_exec_buffer_descriptor_t *descriptor = &exec_buffer->descriptors[i];
                fprintf (stream,
                         "descriptor %zu: name=%s storage=%s access=%s words=%zu bit_capacity=%zu\n",
                         i,
                         descriptor->debug_name ? descriptor->debug_name : "(unnamed)",
                         storage_name (descriptor->storage),
                         access_name (descriptor->access),
                         descriptor->number_of_words,
                         descriptor->number_of_words * ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        }

        for (size_t stage_index = 0; stage_index < exec_buffer->stage_count; stage_index++) {
                itty_exec_stage_t *stage = &exec_buffer->stages[stage_index];
                fprintf (stream,
                         "stage %zu: name=%s first_command=%zu command_count=%zu\n",
                         stage_index,
                         stage->debug_name ? stage->debug_name : "(unnamed)",
                         stage->first_command,
                         stage->command_count);

                for (size_t i = 0; i < stage->command_count; i++) {
                        size_t command_index = stage->first_command + i;
                        describe_command (stream, exec_buffer, command_index, &exec_buffer->commands[command_index]);
                }
        }

        if (fclose (stream) != 0) {
                free (description);
                return NULL;
        }

        return description;
}
