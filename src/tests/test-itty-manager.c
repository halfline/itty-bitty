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

        itty_work_t work1 = { test_callback, &test_items[0], NULL, NULL, false };
        itty_work_t work2 = { test_callback, &test_items[1], NULL, NULL, false };
        itty_work_t work3 = { test_callback, &test_items[2], NULL, NULL, false };

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

void *
multiply_callback (void *user_data)
{
        int *value = user_data;
        int *result = malloc (sizeof (int));
        *result = *value * 3;
        return result;
}

void
test_itty_manager_tasks (void)
{
        itty_manager_t *manager = itty_manager_new ();
        assert (manager != NULL);
        assert (itty_manager_get_queue_count (manager) > 0);

        int value = 7;
        itty_task_t *task = itty_manager_submit (manager, multiply_callback, &value);
        assert (task != NULL);
        itty_manager_wait_for_task (manager, task);
        assert (itty_manager_task_is_complete (task));

        int *result = itty_manager_task_get_result (task);
        assert (result != NULL);
        assert (*result == 21);
        free (result);
        itty_manager_free_task (task);

        itty_manager_free (manager);
}

void
test_itty_manager_task_wait_allows_immediate_free (void)
{
        itty_manager_t *manager = itty_manager_new ();
        assert (manager != NULL);

        for (size_t i = 0; i < 1000; i++) {
                int value = (int) i;
                itty_task_t *task = itty_manager_submit (manager, multiply_callback, &value);
                assert (task != NULL);

                itty_manager_wait_for_task (manager, task);
                int *result = itty_manager_task_get_result (task);
                assert (result != NULL);
                assert (*result == value * 3);
                free (result);
                itty_manager_free_task (task);
        }

        itty_manager_free (manager);
}

void
test_itty_manager_task_groups (void)
{
        itty_manager_t *manager = itty_manager_new ();
        assert (manager != NULL);

        itty_task_group_t *group = itty_manager_create_task_group (manager);
        assert (group != NULL);

        int values[4] = { 1, 2, 3, 4 };
        for (size_t i = 0; i < 4; i++) {
                assert (itty_manager_task_group_submit (group, multiply_callback, &values[i]) != NULL);
        }

        itty_manager_wait_for_task_group (group);
        assert (itty_manager_task_group_get_size (group) == 4);

        for (size_t i = 0; i < 4; i++) {
                itty_task_t *task = itty_manager_task_group_get_task (group, i);
                int *result = itty_manager_task_get_result (task);
                assert (result != NULL);
                assert (*result == values[i] * 3);
                free (result);
        }

        itty_manager_free_task_group (group);
        itty_manager_wait_for_all (manager);
        itty_manager_free (manager);
}

int
main (void)
{
        test_itty_manager ();
        test_itty_manager_tasks ();
        test_itty_manager_task_wait_allows_immediate_free ();
        test_itty_manager_task_groups ();
        return 0;
}
