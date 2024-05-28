#include <stdbool.h>
#include <stdlib.h>
#include <pthread.h>
#include "itty-work-queue-private.h"
#include "itty-work-queue.h"

static void *
itty_work_queue_thread (void *data)
{
        itty_work_queue_t *queue = (itty_work_queue_t *) data;
        while ("It's time to make the donuts") {
                itty_work_t *work = itty_work_queue_dequeue (queue);

                if (work == NULL) {
                        break;
                }

                work->callback (work->user_data);
        }
        return NULL;
}

itty_work_queue_t *
itty_work_queue_new (void)
{
        itty_work_queue_t *queue = malloc (sizeof (itty_work_queue_t));

        queue->head = NULL;
        queue->tail = NULL;
        pthread_mutex_init (&queue->mutex, NULL);
        pthread_cond_init (&queue->condition, NULL);
        queue->running = true;

        pthread_create (&queue->thread, NULL, itty_work_queue_thread, queue);

        return queue;
}

void
itty_work_queue_free (itty_work_queue_t *queue)
{
        if (queue == NULL)
                return;

        pthread_mutex_lock(&queue->mutex);
        queue->running = false;
        pthread_cond_broadcast (&queue->condition);
        pthread_mutex_unlock(&queue->mutex);
        pthread_join (queue->thread, NULL);

        pthread_mutex_destroy (&queue->mutex);
        pthread_cond_destroy (&queue->condition);
        free (queue);
}

void
itty_work_queue_enqueue (itty_work_queue_t *queue,
                         itty_work_t       *work)
{
        work->next = NULL;
        pthread_mutex_lock (&queue->mutex);
        if (queue->tail) {
                queue->tail->next = work;
        } else {
                queue->head = work;
        }
        queue->tail = work;
        pthread_cond_broadcast (&queue->condition);
        pthread_mutex_unlock (&queue->mutex);
}

itty_work_t *
itty_work_queue_dequeue (itty_work_queue_t *queue)
{
        itty_work_t *work;
        pthread_mutex_lock (&queue->mutex);
        while (queue->head == NULL && queue->running) {
                pthread_cond_wait (&queue->condition, &queue->mutex);
        }

        if (queue->head == NULL) {
                pthread_mutex_unlock (&queue->mutex);
                return NULL;
        }

        work = queue->head;
        queue->head = work->next;
        if (queue->head == NULL) {
                queue->tail = NULL;
        }
        pthread_mutex_unlock (&queue->mutex);
        return work;
}

