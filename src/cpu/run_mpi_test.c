#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    // 1. Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // 2. Get the total number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 3. Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // 4. Get the name of the processor (Node Name)
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // 5. Print the status message
    // Rank 0 is the Master, others are Workers
    if (world_rank == 0) {
        printf(">>> MASTER: Rank %d running on [%s]. Total World Size: %d\n", 
                world_rank, processor_name, world_size);
    } else {
        printf("WORKER: Rank %d is alive on [%s]\n", 
                world_rank, processor_name);
    }

    // 6. Finalize the MPI environment
    MPI_Finalize();
    return 0;
}