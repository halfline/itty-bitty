#pragma once

#include <stdbool.h>

#include "itty-operation.h"

typedef struct itty_pipeline_t itty_pipeline_t;
typedef struct itty_pipeline_fence_t itty_pipeline_fence_t;
typedef struct itty_pipeline_operation_t itty_pipeline_operation_t;
typedef void * (* itty_pipeline_handler_t) (void *data);

itty_pipeline_t *itty_pipeline_new (void);

void itty_pipeline_free (itty_pipeline_t *pipeline);

int itty_pipeline_register_operation (itty_pipeline_t *pipeline,
                                      itty_pipeline_handler_t handler);

void itty_pipeline_add_operation (itty_pipeline_t *pipeline,
                                  int              operation_id,
                                  void            *data);

itty_pipeline_fence_t *itty_pipeline_add_fence (itty_pipeline_t *pipeline);

void itty_pipeline_process (itty_pipeline_t *pipeline);

bool itty_pipeline_fence_is_passed (itty_pipeline_fence_t *fence);

void itty_pipeline_fence_wait (itty_pipeline_fence_t *fence);

