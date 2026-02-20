#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 4000
#define HEIGHT 4000
#define MAX_ITER 50000
#define TERMINATE_TAG 999
#define WORK_TAG 1

int compute_pixel(double x, double y) {
    double zr = 0, zi = 0, zr2 = 0, zi2 = 0;
    int iter = 0;
    while (zr2 + zi2 <= 4.0 && iter < MAX_ITER) {
        zi = 2.0 * zr * zi + y;
        zr = zr2 - zi2 + x;
        zr2 = zr * zr;
        zi2 = zi * zi;
        iter++;
    }
    return iter;
}

void master(int num_workers) {
    int next_row = 0;
    int workers_alive = num_workers;
    int row_data[WIDTH];
    MPI_Status status;

    // 1. Initial assignment: Send one row to every worker
    for (int i = 1; i <= num_workers; i++) {
        if (next_row < HEIGHT) {
            MPI_Send(&next_row, 1, MPI_INT, i, WORK_TAG, MPI_COMM_WORLD);
            next_row++;
        }
    }

    // 2. Dynamic loop: Wait for anyone to finish and give them the next row
    while (workers_alive > 0) {
        // Receive result from ANY worker
        MPI_Recv(row_data, WIDTH, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int worker_rank = status.MPI_SOURCE;

        if (next_row < HEIGHT) {
            // Send next available row to the worker that just finished
            MPI_Send(&next_row, 1, MPI_INT, worker_rank, WORK_TAG, MPI_COMM_WORLD);
            next_row++;
        } else {
            // No more work; tell this worker to shut down
            MPI_Send(NULL, 0, MPI_INT, worker_rank, TERMINATE_TAG, MPI_COMM_WORLD);
            workers_alive--;
        }
    }
}

void worker(int rank) {
    int row_to_compute;
    int result_row[WIDTH];
    MPI_Status status;

    while (1) {
        // Wait for a command from Master
        MPI_Recv(&row_to_compute, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TERMINATE_TAG) break; // Exit loop

        // Do the math for the assigned row
        for (int x = 0; x < WIDTH; x++) {
            double real = -2.0 + 3.0 * x / WIDTH;
            double imag = -1.5 + 3.0 * row_to_compute / HEIGHT;
            result_row[x] = compute_pixel(real, imag);
        }

        // Send completed row back and ask for more
        MPI_Send(result_row, WIDTH, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) master(size - 1);
    else worker(rank);

    MPI_Finalize();
    return 0;
}