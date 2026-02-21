#!/bin/bash
#SBATCH --job-name=mandel_final
#SBATCH --output=mandel_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --gres=gpu:1

# 1. IDENTIFY THE TARGET NODE BASED ON ARRAY ID
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    TARGET="intel-head"
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    TARGET="jetson-nano"
else
    TARGET="jetson-orin"
fi

# 2. ENFORCE THE NODE MAPPING
# If the current node isn't the target, re-run via srun to the correct node
if [ "$HOSTNAME" != "$TARGET" ]; then
    echo "Redirecting Task $SLURM_ARRAY_TASK_ID from $HOSTNAME to $TARGET..."
    srun --nodelist=$TARGET --gres=gpu:1 $0
    exit 0
fi

# 3. EXECUTION BLOCK
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# Scrub Slurm vars that break Jetson Unified Memory
unset CUDA_VISIBLE_DEVICES

echo "Final Run on $HOSTNAME"

case $HOSTNAME in
  "intel-head")
    BINARY="/clusterfs/homelab-heterogenous-HPC/bin/x86_64/gpu/mandelbrot_load_balanced_sycl"
    srun $BINARY
    ;;
  "jetson-orin")
    BINARY="/clusterfs/homelab-heterogenous-HPC/bin/arm/bin_orin/gpu/mandelbrot_load_balanced_cuda_orin"
    # Break away from the Slurm cgroup session
    /usr/bin/setsid $BINARY > orin_standalone.out 2>&1
    cat orin_standalone.out
    ;;
  "jetson-nano")
    BINARY="/clusterfs/homelab-heterogenous-HPC/bin/arm/bin_nano/gpu/mandelbrot_load_balanced_cuda_nano"
    $BINARY
    ;;
esac