#include "itty-manager-private.h"
#include "itty-work-queue-private.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <unistd.h>

typedef struct {
        int input;
        int *result;
        atomic_bool completed;
} test_data_t;

void *
test_callback (void *user_data)
{
        test_data_t *data = (test_data_t *) user_data;
        data->result = malloc (sizeof (int));
        *(data->result) = data->input * 2;

        atomic_store (&data->completed, true);

        return data->result;
}

bool
all_work_completed (test_data_t *items, int count)
{
        for (int i = 0; i < count; i++) {
                if (!atomic_load (&items[i].completed)) {
                        return false;
                }
        }
        return true;
}

void
test_itty_manager (void)
{
        itty_manager_t *manager = itty_manager_new ();
        assert (manager != NULL);

        test_data_t test_items[3] = {
            {1, NULL, ATOMIC_VAR_INIT (false)},
            {2, NULL, ATOMIC_VAR_INIT (false)},
            {3, NULL, ATOMIC_VAR_INIT (false)}
        };

        itty_work_t work1 = { test_callback, &test_items[0], NULL, NULL };
        itty_work_t work2 = { test_callback, &test_items[1], NULL, NULL };
        itty_work_t work3 = { test_callback, &test_items[2], NULL, NULL };

        itty_manager_enqueue_work (manager, &work1);
        itty_manager_enqueue_work (manager, &work2);
        itty_manager_enqueue_work (manager, &work3);

        while (!all_work_completed (test_items, 3)) {
            usleep (20);
        }

        for (int i = 0; i < 3; i++) {
            assert (test_items[i].result != NULL);
            assert (* (test_items[i].result) == test_items[i].input * 2);
            printf ("Work result %d: %d\n", i+1, * (test_items[i].result));
            free (test_items[i].result);
        }

        itty_manager_free (manager);
        printf ("All tests passed!\n");
}

int
main (void)
{
        test_itty_manager ();
        return 0;
}

