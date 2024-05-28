#pragma once

#include <pthread.h>
#include "itty-manager.h"
#include "itty-work-queue.h"

struct itty_condition_t {
        itty_condition_check_handler_t check_handler;
        void *data;

        pthread_mutex_t mutex;
        pthread_cond_t variable;
};

struct itty_manager_t
{
        itty_work_queue_t **queues;
        int number_of_queues;
        int next_queue;
        pthread_mutex_t mutex;

        itty_condition_t **conditions;
        int number_of_conditions;
};
