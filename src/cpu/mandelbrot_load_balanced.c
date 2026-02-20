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
    
    unsigned long long total_checksum = 0;
    double start_time = MPI_Wtime(); // Start Timer

    // 1. Initial assignment
    for (int i = 1; i <= num_workers; i++) {
        if (next_row < HEIGHT) {
            MPI_Send(&next_row, 1, MPI_INT, i, WORK_TAG, MPI_COMM_WORLD);
            next_row++;
        }
    }

    // 2. Dynamic loop
    while (workers_alive > 0) {
        MPI_Recv(row_data, WIDTH, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int worker_rank = status.MPI_SOURCE;

        // Accumulate checksum from the received row
        for (int i = 0; i < WIDTH; i++) {
            total_checksum += row_data[i];
        }

        if (next_row < HEIGHT) {
            MPI_Send(&next_row, 1, MPI_INT, worker_rank, WORK_TAG, MPI_COMM_WORLD);
            next_row++;
        } else {
            MPI_Send(NULL, 0, MPI_INT, worker_rank, TERMINATE_TAG, MPI_COMM_WORLD);
            workers_alive--;
        }
    }

    double end_time = MPI_Wtime(); // End Timer

    printf("\n--- HAL9000v2 Performance Report ---\n");
    printf("Nodes/Ranks: %d\n", num_workers + 1);
    printf("Total Time:  %.4f seconds\n", end_time - start_time);
    printf("Checksum:    %llu\n", total_checksum);
    printf("------------------------------------\n");
}

void worker(int rank) {
    int row_to_compute;
    int result_row[WIDTH];
    MPI_Status status;

    while (1) {
        MPI_Recv(&row_to_compute, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        if (status.MPI_TAG == TERMINATE_TAG) break;

        for (int x = 0; x < WIDTH; x++) {
            double real = -2.0 + 3.0 * (double)x / WIDTH;
            double imag = -1.5 + 3.0 * (double)row_to_compute / HEIGHT;
            result_row[x] = compute_pixel(real, imag);
        }
        MPI_Send(result_row, WIDTH, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) printf("Error: This Master-Worker code requires at least 2 ranks.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) master(size - 1);
    else worker(rank);

    MPI_Finalize();
    return 0;
}