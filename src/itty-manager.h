#pragma once

#include "itty-work-queue.h"

typedef struct itty_manager_t itty_manager_t;

itty_manager_t *itty_manager_new (void);
void itty_manager_free (itty_manager_t *manager);
void itty_manager_enqueue_work (itty_manager_t *manager,
                                itty_work_t    *work);
