#!/bin/bash
#SBATCH --job-name=mandel_multi
#SBATCH --output=mandel_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2

# Define nodes based on Array ID
NODES=("intel-head" "jetson-nano" "jetson-orin")
TARGET_NODE=${NODES[$SLURM_ARRAY_TASK_ID]}

# If we aren't on the target node yet, re-submit ourselves to that specific node
if [ "$HOSTNAME" != "$TARGET_NODE" ]; then
    srun --nodelist=$TARGET_NODE --gres=gpu:1 $0
    exit 0
fi

echo "Running on $HOSTNAME"
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

case $HOSTNAME in
  "intel-head")
    BINARY="/clusterfs/homelab-heterogenous-HPC/bin/x86_64/gpu/mandelbrot_load_balanced_sycl"
    ;;
  "jetson-orin")
    BINARY="/clusterfs/homelab-heterogenous-HPC/bin/arm/bin_orin/gpu/mandelbrot_load_balanced_cuda_orin"
    ;;
  "jetson-nano")
    BINARY="/clusterfs/homelab-heterogenous-HPC/bin/arm/bin_nano/gpu/mandelbrot_load_balanced_cuda_nano"
    # The Nano needs to run WITHOUT srun to avoid the internal Slurm timer
    $BINARY
    exit 0
    ;;
esac

# Standard execution for Orin and Intel
srun $BINARY