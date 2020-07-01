#include <zmq.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define NUM_THREADS 8

struct worker_data {
    void *context;
    int mov_height;
    int mov_width;
    int mov_frames;
    int bheight;
    int bwidth;
    int num_frames;
};

void sim_block_coords(int *row_start, int *row_end, int *col_start,
                      int *col_end, int *frame_start, int *frame_end,
                      int mov_height, int mov_width, int mov_frames,
                      int bheight, int bwidth, int num_frames) {

    int _row_start = (rand() % (mov_height - 0 + 1)) + 0;
    *row_start = _row_start - (_row_start % bheight);
    *row_end = (*row_start) + bheight;

    int _col_start = (rand() % (mov_width - 0 + 1)) + 0;
    *col_start = _col_start - (_col_start % bwidth);
    *col_end = (*col_start) + bwidth;

    int _frame_start = (rand() % (mov_frames - 0 + 1)) + 0;
    *frame_start = _frame_start - (_frame_start % num_frames);
    *frame_end = (*frame_start) + num_frames;
}

int get_block(int row_start, int row_end, int col_start, int col_end,
              int frame_start, int frame_end, void *requester, double *buffer,
              size_t n) {

    int res;
    int i = 0;
    int positions[6];
    positions[i++] = row_start;
    positions[i++] = row_end;
    positions[i++] = col_start;
    positions[i++] = col_end;
    positions[i++] = frame_start;
    positions[i] = frame_end;

    res = zmq_send(requester, positions, sizeof(positions), 0);
    if (res != sizeof(positions)) {
        perror("zmq_send() failed");
        exit(1);
    }
    res = zmq_recv(requester, buffer, n * sizeof(double), 0);
    if (res == -1) {
        perror("zmq_recv() failed");
        exit(1);
    }

    return 1;
}

void *worker(void *worker_data) {
    int mov_height = ((struct worker_data *)worker_data)->mov_height;
    int mov_width = ((struct worker_data *)worker_data)->mov_width;
    int mov_frames = ((struct worker_data *)worker_data)->mov_frames;
    int bheight = ((struct worker_data *)worker_data)->bheight;
    int bwidth = ((struct worker_data *)worker_data)->bwidth;
    int num_frames = ((struct worker_data *)worker_data)->num_frames;
    void *context = ((struct worker_data *)worker_data)->context;

    void *requester = zmq_socket(context, ZMQ_REQ);
    int rc = zmq_connect(requester, "ipc://trefide.ipc");
    if (rc != 0) {
        perror("zmq_connect() failed");
        exit(1);
    }

    int iters = (mov_height / bheight) * (mov_width / bwidth) / 8;
    int row_start;
    int row_end;
    int col_start;
    int col_end;
    int frame_start;
    int frame_end;

    int block_size = bheight * bwidth * num_frames;
    double *buffer = (double *)malloc(block_size * sizeof(double));
    if (buffer == NULL) {
        perror("malloc() failed");
        exit(1);
    }

    while (iters--) {
        sim_block_coords(&row_start, &row_end, &col_start, &col_end,
                         &frame_start, &frame_end, mov_height, mov_width,
                         mov_frames, bheight, bwidth, num_frames);

        get_block(row_start, row_end, col_start, col_end, frame_start,
                  frame_end, requester, buffer, block_size);

        sleep(1);

        /*
        printf("new block: %f %f %f ... %f %f %f\n", buffer[0], buffer[1],
               buffer[2], buffer[block_size - 3], buffer[block_size - 2],
               buffer[block_size - 1]);
        */
    }

    free(buffer);
    zmq_close(requester);
    return NULL;
}

int main(void) {
    void *context = zmq_ctx_new();

    /*
    int io_threads = NUM_THREADS;
    zmq_ctx_set(context, ZMQ_IO_THREADS, io_threads);
    if (zmq_ctx_get(context, ZMQ_IO_THREADS) != io_threads) {
        fprintf(stderr, "io_threads not set properly");
        exit(1);
    }
    */

    // void *requester = zmq_socket(context, ZMQ_REQ);
    int mov_height = 800;
    int mov_width = 800;
    int mov_frames = 10000;
    int bheight = 40;
    int bwidth = 40;
    int num_frames = 10000;

    srand(time(NULL));

    struct worker_data wd;
    wd.context = context;
    wd.mov_height = mov_height;
    wd.mov_width = mov_width;
    wd.mov_frames = mov_frames;
    wd.bheight = bheight;
    wd.bwidth = bwidth;
    wd.num_frames = num_frames;

    // Launch pool of worker threads
    pthread_t thread_pool[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&thread_pool[i], NULL, worker, &wd);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(thread_pool[i], NULL);
    }

    zmq_ctx_destroy(context);
    return EXIT_SUCCESS;
}
