#include <unistd.h>
#include "itty-pipeline.h"
#include "itty-pipeline-private.h"
#include "itty-work-queue.h"

static void *
itty_fence_operation (void *data)
{
        itty_pipeline_fence_t *fence = (itty_pipeline_fence_t *) data;
        fence->passed = 1;
        return NULL;
}

itty_pipeline_t *
itty_pipeline_new (void)
{
        itty_pipeline_t *pipeline;

        pipeline = (itty_pipeline_t *) malloc (sizeof (itty_pipeline_t));
        pipeline->elements = NULL;
        pipeline->elements_count = 0;
        pipeline->operations = NULL;
        pipeline->operations_count = 0;
        pipeline->next_operation_id = 1;
        pipeline->next_fence_id = 1;
        pipeline->manager = itty_manager_new ();

        itty_pipeline_register_operation (pipeline, (itty_pipeline_handler_t) itty_fence_operation);

        return pipeline;
}

void
itty_pipeline_free (itty_pipeline_t *pipeline)
{
        size_t i;

        for (i = 0; i < pipeline->elements_count; i++) {
                free (pipeline->elements[i]);
        }
        free (pipeline->elements);

        for (i = 0; i < pipeline->operations_count; i++) {
                free (pipeline->operations[i]);
        }
        free (pipeline->operations);

        itty_manager_free (pipeline->manager);
        free (pipeline);
}

int
itty_pipeline_register_operation (itty_pipeline_t *pipeline,
                                  itty_pipeline_handler_t handler)
{
        itty_pipeline_operation_t *operation;

        operation = (itty_pipeline_operation_t *) malloc (sizeof (itty_pipeline_operation_t));
        operation->id = pipeline->next_operation_id++;
        operation->handler = handler;
        operation->manager = pipeline->manager;

        pipeline->operations_count++;
        pipeline->operations = (itty_pipeline_operation_t **) realloc (pipeline->operations, sizeof (itty_pipeline_operation_t *) * pipeline->operations_count);
        pipeline->operations[pipeline->operations_count - 1] = operation;

        return operation->id;
}

void
itty_pipeline_add_operation (itty_pipeline_t *pipeline,
                             int              operation_id,
                             void            *data)
{
        size_t i;
        itty_pipeline_operation_t *operation;
        itty_pipeline_operation_t *pipeline_operation;

        for (i = 0; i < pipeline->operations_count; i++) {
                if (pipeline->operations[i]->id == operation_id) {
                        operation = pipeline->operations[i];
                        pipeline_operation = (itty_pipeline_operation_t *) malloc (sizeof (itty_pipeline_operation_t));
                        *pipeline_operation = *operation;
                        pipeline_operation->data = data;

                        pipeline->elements_count++;
                        pipeline->elements = (void **) realloc (pipeline->elements, sizeof (void *) * pipeline->elements_count);
                        pipeline->elements[pipeline->elements_count - 1] = pipeline_operation;
                        break;
                }
        }
}

itty_pipeline_fence_t *
itty_pipeline_add_fence (itty_pipeline_t *pipeline)
{
        itty_pipeline_fence_t *fence;

        fence = (itty_pipeline_fence_t *) malloc (sizeof (itty_pipeline_fence_t));
        fence->id = pipeline->next_fence_id++;
        fence->passed = 0;

        itty_pipeline_add_operation (pipeline, ITTY_OPERATION_FENCE_ID, fence);

        return fence;
}

void
itty_pipeline_process (itty_pipeline_t *pipeline)
{
        size_t i;
        itty_pipeline_operation_t *operation;

        for (i = 0; i < pipeline->elements_count; i++) {
                operation = (itty_pipeline_operation_t *) pipeline->elements[i];
                itty_work_t *work = (itty_work_t *) malloc (sizeof (itty_work_t));
                work->callback = (void *(*)(void *))operation->handler;
                work->user_data = operation->data;
                itty_manager_enqueue_work (operation->manager, work);
        }
}

int
itty_pipeline_fence_is_passed (itty_pipeline_fence_t *fence)
{
        return fence->passed;
}

void
itty_pipeline_fence_wait (itty_pipeline_fence_t *fence)
{
        while (!itty_pipeline_fence_is_passed (fence)) {
                usleep (50);
        }
}

