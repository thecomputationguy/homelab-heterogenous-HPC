#!/bin/bash
#SBATCH --job-name=mandelbrot_all_gpus
#SBATCH --output=mandelbrot_%A_%a.out  # %A is JobID, %a is ArrayID
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-2                    # Create 3 sub-jobs (0, 1, 2)

# --- SAFETY CHECK ---
if [ ! -d "/clusterfs/homelab-heterogenous-HPC" ]; then
    echo "ERROR: /clusterfs not found on $HOSTNAME!"
    exit 1
fi

# Get node metadata
NODE_TYPE=$(scontrol show node $(hostname) | grep -oP 'Gres=gpu:\K[a-z0-9]+' | head -n 1)
echo "Array Task ID: $SLURM_ARRAY_TASK_ID running on $HOSTNAME ($NODE_TYPE)"

case $NODE_TYPE in
  "iris")
    source /opt/intel/oneapi/setvars.sh
    BINARY="/clusterfs/homelab-heterogenous-HPC/bin/x86_64/gpu/mandelbrot_load_balanced_sycl"
    ;;
  "orin")
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    BINARY="/clusterfs/homelab-heterogenous-HPC/bin/arm/bin_orin/gpu/mandelbrot_load_balanced_cuda"
    ;;
  "nano")
    BINARY="/clusterfs/homelab-heterogenous-HPC/bin/arm/bin_nano/gpu/mandelbrot_load_balanced_cuda"
    ;;
  *)
    echo "No compatible GPU type found on $HOSTNAME."
    exit 1
    ;;
esac

if [ -f "$BINARY" ]; then
    chmod +x "$BINARY"
    $BINARY
else
    echo "ERROR: Binary not found at $BINARY"
    exit 1
fi