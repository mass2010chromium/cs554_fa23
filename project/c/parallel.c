#include "parallel.h"
#include <stdbool.h>
#include <stdlib.h>

void* worker(void* argp) {
    worker_spec* spec = argp;
    pthread_mutex_lock(&spec->lock);

    pthread_mutex_lock(&spec->owner->ret_lock);
    spec->owner->ret_counter += 1;
    pthread_cond_signal(&spec->owner->ret_cond);
    pthread_mutex_unlock(&spec->owner->ret_lock);

    while (true) {
        pthread_cond_wait(&spec->cond, &spec->lock);
        int status = spec->job_status;
        if (status == 0) { continue; }
        
        if (status == -1) { 
            pthread_mutex_unlock(&spec->lock);
            break;
        }
        if (status == 1) {
            // process job.
            spec->job_data->ret = spec->job_fn(spec->job_data);

            pthread_mutex_lock(&spec->owner->ret_lock);
            spec->owner->ret_counter += 1;
            pthread_cond_signal(&spec->owner->ret_cond);
            pthread_mutex_unlock(&spec->owner->ret_lock);
        }
        // if status == 0, then we loop again. Spurious wakeup!
    }
    return NULL;
}

worker_batch* make_worker_batch(size_t n_workers) {
    worker_batch* ret = malloc(sizeof(worker_batch));
    ret->workers = malloc(sizeof(worker_spec) * n_workers);

    ret->ret_counter = 0;
    ret->n_workers = n_workers;
    pthread_mutex_init(&ret->ret_lock, NULL);
    pthread_cond_init(&ret->ret_cond, NULL);

    for (int i = 0; i < n_workers; ++i) {
        pthread_mutex_init(&ret->workers[i].lock, NULL);
        pthread_cond_init(&ret->workers[i].cond, NULL);
        ret->workers[i].owner = ret;
        ret->workers[i].job_status = 1;

        pthread_create(&ret->workers[i].thread, NULL, worker, &ret->workers[i]);
    }
    // Make sure they finished initialization.
    batch_wait(ret);
    return ret;
}

void send_job_homogenous(worker_batch* workers, int (* job_fn)(job_spec*), job_spec* jobs, size_t job_size) {
    char* jobp = (char*) jobs;
    for (size_t i = 0; i < workers->n_workers; ++i) {
        workers->workers[i].job_fn = job_fn;
        workers->workers[i].job_data = (job_spec*) jobp;
        workers->workers[i].job_status = 1;
        pthread_cond_signal(&workers->workers[i].cond);
        jobp += job_size;
    }
}

void send_job_heterogenous(worker_batch* workers, int (* job_fn)(job_spec*), job_spec* jobs, size_t job_size) {
    char* jobp = (char*) jobs;
    for (size_t i = 0; i < workers->n_workers; ++i) {
        workers->workers[i].job_fn = job_fn+i;
        workers->workers[i].job_data = (job_spec*) jobp;
        workers->workers[i].job_status = 1;
        pthread_cond_signal(&workers->workers[i].cond);
        jobp += job_size;
    }
}

void batch_wait(worker_batch* workers) {
    while (true) {
        pthread_cond_wait(&workers->ret_cond, &workers->ret_lock);
        if (workers->ret_counter >= workers->n_workers) {
            workers->ret_counter = 0;
            pthread_mutex_unlock(&workers->ret_lock);
            break;
        }
    }
}

void destroy_worker_batch(worker_batch* workers) {
    // Shutdown threads.
    for (int i = 0; i < workers->n_workers; ++i) {
        workers->workers[i].job_status = -1;
        pthread_cond_signal(&workers->workers[i].cond);
    }
    for (int i = 0; i < workers->n_workers; ++i) {
        void* ret;
        pthread_join(workers->workers[i].thread, &ret);
    }
    free(workers->workers);
    workers->workers = NULL;
}
