#pragma once

#include "itty-work-queue.h"
#include <pthread.h>
#include <stdbool.h>

typedef struct itty_manager_t itty_manager_t;
typedef struct itty_condition_t itty_condition_t;

typedef bool (*itty_condition_check_handler_t) (void *data);

itty_manager_t *itty_manager_new (void);
void itty_manager_free (itty_manager_t *manager);
void itty_manager_enqueue_work (itty_manager_t *manager,
                                itty_work_t    *work);

itty_condition_t *itty_manager_register_condition (itty_manager_t *manager,
                                                  itty_condition_check_handler_t handler,
                                                  void *data);

void itty_manager_wait_for_condition (itty_manager_t   *manager,
                                      itty_condition_t *condition);
void itty_manager_signal_condition (itty_manager_t   *manager,
                                    itty_condition_t *condition);
void itty_manager_free_condition (itty_manager_t   *manager,
                                  itty_condition_t *condition);
