#pragma once

#include <pthread.h>

typedef struct itty_work_queue_t itty_work_queue_t;
typedef struct itty_work_t itty_work_t;

struct itty_work_t
{
        void *(*callback) (void *user_data);
        void *user_data;
        void *result;
        struct itty_work_t *next;
};

itty_work_queue_t *itty_work_queue_new (void);
void itty_work_queue_free (itty_work_queue_t *queue);
void itty_work_queue_enqueue (itty_work_queue_t *queue,
                              itty_work_t       *work);
itty_work_t *itty_work_queue_dequeue (itty_work_queue_t *queue);

