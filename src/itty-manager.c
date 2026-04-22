#include "itty-manager-private.h"
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysinfo.h>

static void *
itty_manager_run_task (void *data)
{
        itty_task_t *task = data;
        void *result = task->handler (task->data);

        pthread_mutex_lock (&task->manager->mutex);
        task->manager->active_tasks--;
        pthread_cond_broadcast (&task->manager->idle_condition);
        pthread_mutex_unlock (&task->manager->mutex);

        pthread_mutex_lock (&task->mutex);
        task->result = result;
        task->complete = true;
        pthread_cond_broadcast (&task->condition);
        pthread_mutex_unlock (&task->mutex);

        return result;
}

itty_manager_t *
itty_manager_new (void)
{
        itty_manager_t *manager = malloc (sizeof (itty_manager_t));
        if (manager != NULL) {
                manager->number_of_queues = get_nprocs ();
                if (manager->number_of_queues <= 0)
                        manager->number_of_queues = 1;

                manager->queues = malloc (manager->number_of_queues * sizeof (itty_work_queue_t *));
                if (!manager->queues) {
                        free (manager);
                        return NULL;
                }

                for (int i = 0; i < manager->number_of_queues; i++) {
                        manager->queues[i] = itty_work_queue_new ();
                        if (!manager->queues[i]) {
                                for (int j = 0; j < i; j++)
                                        itty_work_queue_free (manager->queues[j]);
                                free (manager->queues);
                                free (manager);
                                return NULL;
                        }
                }

                manager->next_queue = 0;
                pthread_mutex_init (&manager->mutex, NULL);
                pthread_cond_init (&manager->idle_condition, NULL);
                manager->active_tasks = 0;

                manager->conditions = NULL;
                manager->number_of_conditions = 0;
        }
        return manager;
}

void
itty_manager_free (itty_manager_t *manager)
{
        if (manager == NULL)
                return;

        for (int i = 0; i < manager->number_of_conditions; i++) {
                pthread_mutex_destroy (&manager->conditions[i]->mutex);
                pthread_cond_destroy (&manager->conditions[i]->variable);
                free (manager->conditions[i]);
        }
        free (manager->conditions);

        for (int i = 0; i < manager->number_of_queues; i++) {
                itty_work_queue_free (manager->queues[i]);
        }
        free (manager->queues);

        pthread_cond_destroy (&manager->idle_condition);
        pthread_mutex_destroy (&manager->mutex);
        free (manager);
}

void
itty_manager_enqueue_work (itty_manager_t *manager,
                           itty_work_t    *work)
{
        pthread_mutex_lock (&manager->mutex);
        int queue_index = manager->next_queue;
        manager->next_queue = (manager->next_queue + 1) % manager->number_of_queues;
        pthread_mutex_unlock (&manager->mutex);

        itty_work_queue_enqueue (manager->queues[queue_index], work);
}

itty_task_t *
itty_manager_submit (itty_manager_t     *manager,
                     itty_work_handler_t handler,
                     void               *data)
{
        if (!manager || !handler)
                return NULL;

        itty_task_t *task = malloc (sizeof (itty_task_t));
        if (!task)
                return NULL;

        task->manager = manager;
        task->handler = handler;
        task->data = data;
        task->result = NULL;
        task->complete = false;
        pthread_mutex_init (&task->mutex, NULL);
        pthread_cond_init (&task->condition, NULL);

        itty_work_t *work = malloc (sizeof (itty_work_t));
        if (!work) {
                pthread_cond_destroy (&task->condition);
                pthread_mutex_destroy (&task->mutex);
                free (task);
                return NULL;
        }

        work->callback = itty_manager_run_task;
        work->user_data = task;
        work->result = NULL;
        work->next = NULL;
        work->free_after_run = true;

        pthread_mutex_lock (&manager->mutex);
        manager->active_tasks++;
        pthread_mutex_unlock (&manager->mutex);

        itty_manager_enqueue_work (manager, work);

        return task;
}

void
itty_manager_wait_for_task (itty_manager_t *manager,
                            itty_task_t    *task)
{
        (void) manager;

        if (!task)
                return;

        pthread_mutex_lock (&task->mutex);
        while (!task->complete) {
                pthread_cond_wait (&task->condition, &task->mutex);
        }
        pthread_mutex_unlock (&task->mutex);
}

void *
itty_manager_task_get_result (itty_task_t *task)
{
        if (!task)
                return NULL;

        pthread_mutex_lock (&task->mutex);
        void *result = task->result;
        pthread_mutex_unlock (&task->mutex);

        return result;
}

bool
itty_manager_task_is_complete (itty_task_t *task)
{
        if (!task)
                return false;

        pthread_mutex_lock (&task->mutex);
        bool complete = task->complete;
        pthread_mutex_unlock (&task->mutex);

        return complete;
}

void
itty_manager_free_task (itty_task_t *task)
{
        if (!task)
                return;

        itty_manager_wait_for_task (task->manager, task);
        pthread_cond_destroy (&task->condition);
        pthread_mutex_destroy (&task->mutex);
        free (task);
}

itty_task_group_t *
itty_manager_create_task_group (itty_manager_t *manager)
{
        if (!manager)
                return NULL;

        itty_task_group_t *group = malloc (sizeof (itty_task_group_t));
        if (!group)
                return NULL;

        group->manager = manager;
        group->tasks = NULL;
        group->count = 0;
        group->capacity = 0;

        return group;
}

itty_task_t *
itty_manager_task_group_submit (itty_task_group_t  *group,
                                itty_work_handler_t handler,
                                void               *data)
{
        if (!group)
                return NULL;

        if (group->count == group->capacity) {
                size_t new_capacity = group->capacity == 0 ? 8 : group->capacity * 2;
                itty_task_t **tasks = realloc (group->tasks, new_capacity * sizeof (itty_task_t *));
                if (!tasks)
                        return NULL;

                group->tasks = tasks;
                group->capacity = new_capacity;
        }

        itty_task_t *task = itty_manager_submit (group->manager, handler, data);
        if (!task)
                return NULL;

        group->tasks[group->count++] = task;

        return task;
}

void
itty_manager_wait_for_task_group (itty_task_group_t *group)
{
        if (!group)
                return;

        for (size_t i = 0; i < group->count; i++) {
                itty_manager_wait_for_task (group->manager, group->tasks[i]);
        }
}

size_t
itty_manager_task_group_get_size (itty_task_group_t *group)
{
        if (!group)
                return 0;

        return group->count;
}

itty_task_t *
itty_manager_task_group_get_task (itty_task_group_t *group,
                                  size_t             index)
{
        if (!group || index >= group->count)
                return NULL;

        return group->tasks[index];
}

void
itty_manager_free_task_group (itty_task_group_t *group)
{
        if (!group)
                return;

        for (size_t i = 0; i < group->count; i++) {
                itty_manager_free_task (group->tasks[i]);
        }
        free (group->tasks);
        free (group);
}

void
itty_manager_wait_for_all (itty_manager_t *manager)
{
        if (!manager)
                return;

        pthread_mutex_lock (&manager->mutex);
        while (manager->active_tasks > 0) {
                pthread_cond_wait (&manager->idle_condition, &manager->mutex);
        }
        pthread_mutex_unlock (&manager->mutex);
}

size_t
itty_manager_get_queue_count (itty_manager_t *manager)
{
        if (!manager)
                return 0;

        return manager->number_of_queues;
}

itty_condition_t *
itty_manager_register_condition (itty_manager_t *manager,
                                 itty_condition_check_handler_t check_handler,
                                 void *data)
{
        itty_condition_t *condition = malloc (sizeof (itty_condition_t));

        condition->check_handler = check_handler;
        condition->data = data;
        pthread_mutex_init (&condition->mutex, NULL);
        pthread_cond_init (&condition->variable, NULL);

        pthread_mutex_lock (&manager->mutex);

        manager->number_of_conditions++;
        manager->conditions = realloc (manager->conditions,
                                      manager->number_of_conditions * sizeof (itty_condition_t *));
        manager->conditions[manager->number_of_conditions - 1] = condition;

        pthread_mutex_unlock (&manager->mutex);

        return condition;
}

void
itty_manager_wait_for_condition (itty_manager_t   *manager,
                                 itty_condition_t *condition)
{
        (void) manager;

        pthread_mutex_lock (&condition->mutex);

        while (!condition->check_handler (condition->data)) {
                pthread_cond_wait (&condition->variable, &condition->mutex);
        }

        pthread_mutex_unlock (&condition->mutex);
}

void itty_manager_signal_condition (itty_manager_t   *manager,
                                   itty_condition_t *condition)
{
        (void) manager;

        pthread_mutex_lock (&condition->mutex);
        pthread_cond_broadcast (&condition->variable);
        pthread_mutex_unlock (&condition->mutex);
}

void
itty_manager_free_condition (itty_manager_t   *manager,
                             itty_condition_t *condition)
{
        pthread_mutex_lock (&manager->mutex);

        for (int i = 0; i < manager->number_of_conditions; i++) {
                if (manager->conditions[i] == condition) {
                        pthread_mutex_destroy (&condition->mutex);
                        pthread_cond_destroy (&condition->variable);
                        free (condition);
                        if (i < manager->number_of_conditions - 1) {
                                manager->conditions[i] =
                                        manager->conditions[manager->number_of_conditions - 1];
                        }
                        manager->number_of_conditions--;
                        break;
                }
        }

        pthread_mutex_unlock (&manager->mutex);
}
