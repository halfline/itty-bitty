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
        }
        return manager;
}

void
itty_manager_free (itty_manager_t *manager)
{
        if (manager == NULL)
                return;

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

