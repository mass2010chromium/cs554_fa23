#pragma once

#include <pthread.h>

typedef struct job_spec {
    int ret;
    char data[0];
} job_spec;

struct worker_batch;

typedef struct worker_spec {
    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int job_status;     // 0: no job. 1: job ready. -1: exit thread.
    int (* job_fn)(job_spec*);
    job_spec* job_data;
    struct worker_batch* owner;
} worker_spec;

struct worker_batch {
    size_t n_workers;
    size_t ret_counter;
    pthread_mutex_t ret_lock;
    pthread_cond_t ret_cond;
    worker_spec* workers;
};
typedef struct worker_batch worker_batch;

worker_batch* make_worker_batch(size_t n_workers);

void send_job_homogenous(worker_batch* workers, int (* job_fn)(job_spec*), job_spec* jobs, size_t job_size);

void send_job_heterogenous(worker_batch* workers, int (* job_fns)(job_spec*), job_spec* jobs, size_t job_size);

void batch_wait(worker_batch* workers);

void destroy_worker_batch(worker_batch* workers);

