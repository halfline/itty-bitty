#pragma once

#include "itty-work-queue.h"
#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>

typedef struct itty_manager_t itty_manager_t;
typedef struct itty_condition_t itty_condition_t;
typedef struct itty_task_t itty_task_t;
typedef struct itty_task_group_t itty_task_group_t;

typedef bool (*itty_condition_check_handler_t) (void *data);

itty_manager_t *itty_manager_new (void);
void itty_manager_free (itty_manager_t *manager);
void itty_manager_enqueue_work (itty_manager_t *manager,
                                itty_work_t    *work);

itty_task_t *itty_manager_submit (itty_manager_t     *manager,
                                  itty_work_handler_t handler,
                                  void               *data);
void itty_manager_wait_for_task (itty_manager_t *manager,
                                 itty_task_t    *task);
void *itty_manager_task_get_result (itty_task_t *task);
bool itty_manager_task_is_complete (itty_task_t *task);
void itty_manager_free_task (itty_task_t *task);

itty_task_group_t *itty_manager_create_task_group (itty_manager_t *manager);
itty_task_t *itty_manager_task_group_submit (itty_task_group_t  *group,
                                             itty_work_handler_t handler,
                                             void               *data);
void itty_manager_wait_for_task_group (itty_task_group_t *group);
size_t itty_manager_task_group_get_size (itty_task_group_t *group);
itty_task_t *itty_manager_task_group_get_task (itty_task_group_t *group,
                                               size_t             index);
void itty_manager_free_task_group (itty_task_group_t *group);

void itty_manager_wait_for_all (itty_manager_t *manager);
size_t itty_manager_get_queue_count (itty_manager_t *manager);

itty_condition_t *itty_manager_register_condition (itty_manager_t *manager,
                                                  itty_condition_check_handler_t handler,
                                                  void *data);

void itty_manager_wait_for_condition (itty_manager_t   *manager,
                                      itty_condition_t *condition);
void itty_manager_signal_condition (itty_manager_t   *manager,
                                    itty_condition_t *condition);
void itty_manager_free_condition (itty_manager_t   *manager,
                                  itty_condition_t *condition);
