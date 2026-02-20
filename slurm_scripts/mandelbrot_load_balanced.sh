#!/bin/bash
#SBATCH --job-name=HAL_Mandelbrot_Load_Balanced
#SBATCH --output=hal_overload_%j.txt
#SBATCH --nodes=4
#SBATCH --ntasks=73
#SBATCH --overcommit

# 1. DEFINE PATHS
X86_BIN="/clusterfs/HPC_development/homelab-heterogenous-HPC/bin/x86_64/cpu/run_mpi_test_x86"
ARM_BIN="clusterfs/HPC_development/homelab-heterogenous-HPC/bin/arm/bin_nano/cpu/mandelbrot_load_balanced"

# 2. DEFINE SLOTS (High-density allocation)
# We tell MPI each node has many 'slots' to allow high -np counts
NUC_HOST="10.0.0.1:33"   # Master + 31 workers
ORIN_HOST="10.0.0.2:16"  # 16 workers
NANO_HOST="10.0.0.3:12"  # 12 workers
PI_HOST="10.0.0.4:12"    # 12 workers

# 3. FLAGS 
# --oversubscribe is mandatory here to bypass physical core limits
FLAGS="--tag-output --oversubscribe --use-hwthread-cpus \
       --mca btl_tcp_if_include 10.0.0.0/24 \
       --mca pml ob1 --mca btl self,tcp"

# 4. EXECUTION
# Total Ranks = 32 (NUC) + 16 (Orin) + 12 (Nano) + 12 (Pi) + 1 (Master is Rank 0 on NUC)
# Total = 73 Ranks
echo "Launching 73-rank Cluster Overload (4000x4000 @ 50k iters)..."

mpirun $FLAGS \
    --host $NUC_HOST -np 32 $X86_BIN : \
    --host $ORIN_HOST,$NANO_HOST,$PI_HOST -np 41 $ARM_BIN