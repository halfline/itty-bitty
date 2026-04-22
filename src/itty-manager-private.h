#pragma once

#include <pthread.h>
#include <stddef.h>
#include "itty-manager.h"
#include "itty-work-queue.h"

struct itty_condition_t {
        itty_condition_check_handler_t check_handler;
        void *data;

        pthread_mutex_t mutex;
        pthread_cond_t variable;
};

struct itty_task_t {
        itty_manager_t *manager;
        itty_work_handler_t handler;
        void *data;
        void *result;
        bool complete;

        pthread_mutex_t mutex;
        pthread_cond_t condition;
};

struct itty_task_group_t {
        itty_manager_t *manager;
        itty_task_t **tasks;
        size_t count;
        size_t capacity;
};

struct itty_manager_t
{
        itty_work_queue_t **queues;
        int number_of_queues;
        int next_queue;
        pthread_mutex_t mutex;
        pthread_cond_t idle_condition;
        size_t active_tasks;

        itty_condition_t **conditions;
        int number_of_conditions;
};
