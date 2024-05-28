#pragma once

#include <pthread.h>
#include <stdbool.h>
#include "itty-work-queue.h"

struct itty_work_queue_t
{
        itty_work_t *head;
        itty_work_t *tail;

        pthread_mutex_t mutex;
        pthread_cond_t condition;

        pthread_t thread;
        bool running;
};
