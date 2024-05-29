#pragma once

#include <stdatomic.h>
#include <stdlib.h>
#include "itty-pipeline.h"
#include "itty-manager.h"

struct itty_pipeline_t {
        void **elements;
        int elements_count;
        itty_pipeline_operation_t **operations;
        int operations_count;
        int fence_operation_id;
        int next_operation_id;
        int next_fence_id;
        itty_manager_t *manager;
};

struct itty_pipeline_operation_t {
        int id;
        itty_pipeline_handler_t handler;
        void *data;
        itty_manager_t *manager;
};

struct itty_pipeline_fence_t {
        int id;
        atomic_bool passed;
        struct itty_pipeline_t *pipeline;
        itty_condition_t *condition;
};
