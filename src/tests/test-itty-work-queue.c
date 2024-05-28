#include "itty-work-queue.h"
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
test_itty_work_queue (void)
{
        itty_work_queue_t *queue = itty_work_queue_new ();
        assert (queue != NULL);

        int data1 = 1, data2 = 2, data3 = 3;
        itty_work_t work1 = { test_callback, &data1, NULL, NULL };
        itty_work_t work2 = { test_callback, &data2, NULL, NULL };
        itty_work_t work3 = { test_callback, &data3, NULL, NULL };

        itty_work_queue_enqueue (queue, &work1);
        itty_work_queue_enqueue (queue, &work2);
        itty_work_queue_enqueue (queue, &work3);

        itty_work_t *dequeued_work;
        dequeued_work = itty_work_queue_dequeue (queue);
        assert (dequeued_work == &work1);
        dequeued_work->result = dequeued_work->callback (dequeued_work->user_data);
        assert (*(int *) dequeued_work->result == 2);

        dequeued_work = itty_work_queue_dequeue (queue);
        assert (dequeued_work == &work2);
        dequeued_work->result = dequeued_work->callback (dequeued_work->user_data);
        assert (*(int *) dequeued_work->result == 4);

        dequeued_work = itty_work_queue_dequeue (queue);
        assert (dequeued_work == &work3);
        dequeued_work->result = dequeued_work->callback (dequeued_work->user_data);
        assert (*(int *) dequeued_work->result == 6);

        itty_work_queue_free (queue);

        printf ("All tests passed!\n");
}

int
main (void)
{
        test_itty_work_queue ();
        return 0;
}

