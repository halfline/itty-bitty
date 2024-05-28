#include "itty-manager-private.h"
#include <stdlib.h>
#include <unistd.h>
#include <sys/sysinfo.h>

itty_manager_t *
itty_manager_new (void)
{
        itty_manager_t *manager = malloc (sizeof (itty_manager_t));
        if (manager != NULL) {
                manager->number_of_queues = get_nprocs ();
                manager->queues = malloc (manager->number_of_queues * sizeof (itty_work_queue_t *));
                for (int i = 0; i < manager->number_of_queues; i++) {
                        manager->queues[i] = itty_work_queue_new ();
                }

                manager->next_queue = 0;
                pthread_mutex_init (&manager->mutex, NULL);

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
        pthread_mutex_lock (&condition->mutex);

        while (!condition->check_handler (condition->data)) {
                pthread_cond_wait (&condition->variable, &condition->mutex);
        }

        pthread_mutex_unlock (&condition->mutex);
}

void itty_manager_signal_condition (itty_manager_t   *manager,
                                   itty_condition_t *condition)
{
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

