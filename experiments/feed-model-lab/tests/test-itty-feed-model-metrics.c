#include "itty-bit-string.h"
#include "itty-bit-string-list.h"
#include "itty-feed-model.h"
#include "itty-model-metrics.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

static itty_bit_string_t *
create_bit_string (size_t word)
{
        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        itty_bit_string_append_word (bit_string, word);
        return bit_string;
}

static size_t
create_half_populated_word (void)
{
        return ((size_t) 1 << (ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2)) - 1;
}

static void
assert_near (double actual,
             double expected)
{
        assert (fabs (actual - expected) < 0.000001);
}

static void
test_itty_feed_model_measure_masks (void)
{
        itty_feed_model_t *model = itty_feed_model_new (1, 1, 1, 1);
        itty_bit_string_list_t *input = itty_bit_string_list_new ();
        itty_bit_string_t *target = create_bit_string (create_half_populated_word ());
        itty_model_metrics_bit_summary_t summary;

        itty_bit_string_list_append (input, create_bit_string (0));

        assert (itty_feed_model_measure_masks (model, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert (summary.set_bits == 0);
        assert_near (summary.entropy, 0.0);

        assert (itty_feed_model_train_one (model, input, target));

        assert (itty_feed_model_measure_layer_masks (model, 0, &summary));
        assert (summary.bit_count == ITTY_BIT_STRING_WORD_SIZE_IN_BITS);
        assert (summary.set_bits == ITTY_BIT_STRING_WORD_SIZE_IN_BITS / 2);
        assert_near (summary.entropy, 1.0);

        itty_bit_string_free (target);
        itty_bit_string_list_free (input);
        itty_feed_model_free (model);
}

int
main (void)
{
        test_itty_feed_model_measure_masks ();
        printf ("All itty-feed-model-metrics tests passed.\n");
        return 0;
}
