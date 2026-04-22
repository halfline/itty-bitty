#include "itty-work-queue.h"
#include "itty-work-queue-private.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdatomic.h>
#include <unistd.h>

typedef struct {
        int input;
        int result;
        atomic_bool completed;
} test_data_t;

void *
test_callback (void *user_data)
{
        test_data_t *data = user_data;
        data->result = data->input * 2;
        atomic_store (&data->completed, true);
        return NULL;
}

void
test_itty_work_queue (void)
{
        itty_work_queue_t *queue = itty_work_queue_new ();
        assert (queue != NULL);

        test_data_t data1 = { 1, 0, ATOMIC_VAR_INIT (false) };
        test_data_t data2 = { 2, 0, ATOMIC_VAR_INIT (false) };
        test_data_t data3 = { 3, 0, ATOMIC_VAR_INIT (false) };
        itty_work_t work1 = { test_callback, &data1, NULL, NULL, false };
        itty_work_t work2 = { test_callback, &data2, NULL, NULL, false };
        itty_work_t work3 = { test_callback, &data3, NULL, NULL, false };

        itty_work_queue_enqueue (queue, &work1);
        itty_work_queue_enqueue (queue, &work2);
        itty_work_queue_enqueue (queue, &work3);

        while (!atomic_load (&data1.completed) ||
               !atomic_load (&data2.completed) ||
               !atomic_load (&data3.completed)) {
                usleep (20);
        }

        assert (data1.result == 2);
        assert (data2.result == 4);
        assert (data3.result == 6);

        itty_work_queue_free (queue);

        printf ("All tests passed!\n");
}

int
main (void)
{
        test_itty_work_queue ();
        return 0;
}
