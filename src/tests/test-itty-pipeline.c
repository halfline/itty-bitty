#include <stdio.h>
#include <stdlib.h>
#include "itty-pipeline-private.h"
#include "itty-pipeline.h"

void *
sample_operation (void *data)
{
        printf ("Sample operation executed with data: %s\n", (char *) data);
        return NULL;
}

int
main (void)
{
        itty_pipeline_t *pipeline;
        itty_pipeline_fence_t *fence;
        int op_id;

        pipeline = itty_pipeline_new ();

        op_id = itty_pipeline_register_operation (pipeline, sample_operation);

        itty_pipeline_add_operation (pipeline, op_id, "Hello, World!");
        fence = itty_pipeline_add_fence (pipeline);
        itty_pipeline_add_operation (pipeline, op_id, "Goodbye, World!");

        itty_pipeline_process (pipeline);

        if (itty_pipeline_fence_is_passed (fence)) {
                printf ("Fence has passed\n");
        } else {
                printf ("Fence has not passed\n");
        }

        itty_pipeline_free (pipeline);

        return 0;
}

