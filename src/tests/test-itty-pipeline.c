#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "itty-pipeline-private.h"
#include "itty-pipeline.h"

// Global variables to track execution order
static int execution_counter = 0;
static int op1_execution_order = 0;
static int op2_execution_order = 0;

void *
sample_operation_1 (void *data)
{
        usleep (10000);
        op1_execution_order = ++execution_counter;
        printf ("Operation 1 executed with data: %s (order: %d)\n",
               (char *) data, op1_execution_order);
        return NULL;
}

void *
sample_operation_2 (void *data)
{
        usleep (10000);
        op2_execution_order = ++execution_counter;
        printf ("Operation 2 executed with data: %s (order: %d)\n",
               (char *) data, op2_execution_order);
        return NULL;
}

int
main (void)
{
        itty_pipeline_t *pipeline;
        itty_pipeline_fence_t *fence;
        int op1_id, op2_id;

        pipeline = itty_pipeline_new ();

        op1_id = itty_pipeline_register_operation (pipeline, sample_operation_1);
        op2_id = itty_pipeline_register_operation (pipeline, sample_operation_2);

        itty_pipeline_add_operation (pipeline, op1_id, "First operation");
        itty_pipeline_add_operation (pipeline, op1_id, "Second operation");

        fence = itty_pipeline_add_fence (pipeline);

        itty_pipeline_add_operation (pipeline, op2_id, "Third operation");
        itty_pipeline_add_operation (pipeline, op2_id, "Fourth operation");

        itty_pipeline_process (pipeline);

        printf ("Waiting for fence...\n");
        itty_pipeline_fence_wait (fence);
        printf ("Fence has passed!\n");

        if (op1_execution_order < op2_execution_order) {
                printf ("TEST PASSED: Operations executed in correct order\n");
        } else {
                printf ("TEST FAILED: Operations executed in wrong order\n");
        }

        itty_pipeline_free (pipeline);

        return 0;
}

