#pragma once

#include <pthread.h>
#include "itty-manager.h"
#include "itty-work-queue.h"

struct itty_manager_t
{
        itty_work_queue_t **queues;
        int number_of_queues;
        int next_queue;
        pthread_mutex_t mutex;
};
