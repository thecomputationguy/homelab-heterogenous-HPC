#!/bin/bash
#SBATCH --job-name=HAL_MPI_test
#SBATCH --output=comm_check_%j.txt
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --overcommit

# 1. DEFINE PATHS TO NATIVE BINARIES
X86_BIN="/clusterfs/HPC_development/homelab-heterogenous-HPC/bin/x86_64/cpu/run_mpi_test_x86"
ARM64_BIN="HPC_development/homelab-heterogenous-HPC/bin/arm/bin_nano/cpu/run_mpi_test_arm"

# 2. LAUNCH ONE RANK PER PHYSICAL NODE
# This proves every node can "talk" back to the Master on the NUC
echo "Starting HAL9000v2 Cluster Communication Handshake..."

mpirun --oversubscribe --tag-output \
    --mca btl_tcp_if_include 10.0.0.0/24 \
    --host 10.0.0.1 -np 1 $X86_BIN : \
    --host 10.0.0.2 -np 1 $ARM64_BIN : \
    --host 10.0.0.3 -np 1 $ARM64_BIN : \
    --host 10.0.0.4 -np 1 $ARM64_BIN