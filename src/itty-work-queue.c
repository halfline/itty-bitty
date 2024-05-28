#include <stdlib.h>
#include "itty-work-queue-private.h"
#include "itty-work-queue.h"

itty_work_queue_t *
itty_work_queue_new (void)
{
        itty_work_queue_t *queue = malloc (sizeof (itty_work_queue_t));

        queue->head = NULL;
        queue->tail = NULL;
        pthread_mutex_init (&queue->mutex, NULL);
        pthread_cond_init (&queue->condition, NULL);

        return queue;
}

void
itty_work_queue_free (itty_work_queue_t *queue)
{
        if (queue == NULL)
                return;

        pthread_mutex_destroy (&queue->mutex);
        pthread_cond_destroy (&queue->condition);
        free (queue);
}

void
itty_work_queue_enqueue (itty_work_queue_t *queue,
                         itty_work_t       *work)
{
        pthread_mutex_lock (&queue->mutex);
        if (queue->tail) {
                queue->tail->next = work;
        } else {
                queue->head = work;
        }
        queue->tail = work;
        pthread_cond_signal (&queue->condition);
        pthread_mutex_unlock (&queue->mutex);
}

itty_work_t *
itty_work_queue_dequeue (itty_work_queue_t *queue)
{
        itty_work_t *work;
        pthread_mutex_lock (&queue->mutex);
        while (queue->head == NULL) {
                pthread_cond_wait (&queue->condition, &queue->mutex);
        }
        work = queue->head;
        queue->head = work->next;
        if (queue->head == NULL) {
                queue->tail = NULL;
        }
        pthread_mutex_unlock (&queue->mutex);
        return work;
}

