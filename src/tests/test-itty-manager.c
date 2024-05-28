#include "itty-manager-private.h"
#include "itty-work-queue-private.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void *
test_callback (void *user_data)
{
        int *data = (int *) user_data;
        int *result = malloc (sizeof (int));
        *result = *data * 2;
        return result;
}

void
test_itty_manager (void)
{
        itty_manager_t *manager = itty_manager_new ();
        assert (manager != NULL);

        int data1 = 1, data2 = 2, data3 = 3;
        itty_work_t work1 = { test_callback, &data1, NULL, NULL };
        itty_work_t work2 = { test_callback, &data2, NULL, NULL };
        itty_work_t work3 = { test_callback, &data3, NULL, NULL };

        itty_manager_enqueue_work (manager, &work1);
        itty_manager_enqueue_work (manager, &work2);
        itty_manager_enqueue_work (manager, &work3);

        // Dequeue and process work from each queue
        for (int i = 0; i < 3; i++)
        {
                itty_work_t *dequeued_work = itty_work_queue_dequeue (manager->queues[i]);
                if (dequeued_work != NULL)
                {
                        dequeued_work->result = dequeued_work->callback (dequeued_work->user_data);
                        printf ("Work result: %d\n", *(int *) dequeued_work->result);
                        free (dequeued_work->result);
                }
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

