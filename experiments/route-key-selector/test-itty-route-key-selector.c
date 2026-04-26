#include "itty-route-key-selector.h"

#include <assert.h>
#include <stdio.h>

static itty_bit_string_t *
create_bit_string (size_t word)
{
        itty_bit_string_t *bit_string = itty_bit_string_new (ITTY_BIT_STRING_MUTABILITY_READ_WRITE);
        itty_bit_string_append_word (bit_string, word);
        return bit_string;
}

static void
assert_selects (itty_route_key_selector_t *selector,
                itty_bit_string_t         *probe,
                size_t                     expected_route)
{
        itty_route_key_selection_t selection;

        assert (itty_route_key_selector_select (selector, probe, &selection));
        assert (selection.has_selection);
        assert (selection.selected_route == expected_route);
        assert (selection.selected_gap > 0);
}

static void
test_itty_route_key_selector_ab_stability (void)
{
        itty_route_key_selector_t *selector = itty_route_key_selector_new ();
        itty_bit_string_t *probe_a = create_bit_string (0b00001111);
        itty_bit_string_t *probe_b = create_bit_string (0b11110000);

        assert (selector != NULL);
        assert (itty_route_key_selector_append_route (selector, probe_b));
        assert (itty_route_key_selector_append_route (selector, probe_a));

        assert_selects (selector, probe_a, 1);
        assert_selects (selector, probe_b, 0);

        itty_bit_string_free (probe_a);
        itty_bit_string_free (probe_b);
        itty_route_key_selector_free (selector);
}

static void
test_itty_route_key_selector_abc_stability (void)
{
        itty_route_key_selector_t *selector = itty_route_key_selector_new ();
        itty_bit_string_t *probe_a = create_bit_string (0b00001111);
        itty_bit_string_t *probe_b = create_bit_string (0b11110000);
        itty_bit_string_t *probe_c = create_bit_string (0b00110011);

        assert (selector != NULL);
        assert (itty_route_key_selector_append_route (selector, probe_c));
        assert (itty_route_key_selector_append_route (selector, probe_b));
        assert (itty_route_key_selector_append_route (selector, probe_a));

        assert_selects (selector, probe_a, 2);
        assert_selects (selector, probe_b, 1);
        assert_selects (selector, probe_c, 0);

        itty_bit_string_free (probe_a);
        itty_bit_string_free (probe_b);
        itty_bit_string_free (probe_c);
        itty_route_key_selector_free (selector);
}

static void
test_itty_route_key_selector_abcd_no_regression_when_adding_route (void)
{
        itty_route_key_selector_t *selector = itty_route_key_selector_new ();
        itty_bit_string_t *probe_a = create_bit_string (0x0f0f0f0f);
        itty_bit_string_t *probe_b = create_bit_string (0xf0f0f0f0);
        itty_bit_string_t *probe_c = create_bit_string (0x33333333);
        itty_bit_string_t *probe_d = create_bit_string (0x5555aaaa);

        assert (selector != NULL);
        assert (itty_route_key_selector_append_route (selector, probe_b));
        assert (itty_route_key_selector_append_route (selector, probe_a));
        assert_selects (selector, probe_a, 1);
        assert_selects (selector, probe_b, 0);

        assert (itty_route_key_selector_append_route (selector, probe_c));
        assert_selects (selector, probe_a, 1);
        assert_selects (selector, probe_b, 0);
        assert_selects (selector, probe_c, 2);

        assert (itty_route_key_selector_append_route (selector, probe_d));
        assert_selects (selector, probe_a, 1);
        assert_selects (selector, probe_b, 0);
        assert_selects (selector, probe_c, 2);
        assert_selects (selector, probe_d, 3);

        itty_bit_string_free (probe_a);
        itty_bit_string_free (probe_b);
        itty_bit_string_free (probe_c);
        itty_bit_string_free (probe_d);
        itty_route_key_selector_free (selector);
}

int
main (void)
{
        test_itty_route_key_selector_ab_stability ();
        test_itty_route_key_selector_abc_stability ();
        test_itty_route_key_selector_abcd_no_regression_when_adding_route ();
        printf ("All itty-route-key-selector tests passed.\n");
        return 0;
}
