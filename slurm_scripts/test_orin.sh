#!/bin/bash
#SBATCH --job-name=orin_clean_launch
#SBATCH --nodelist=jetson-orin
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=orin_clean_%j.out

echo "Launching on $HOSTNAME with a stripped environment..."

# 'env -i' ignores all inherited Slurm variables (like CUDA_VISIBLE_DEVICES)
# We manually provide only the PATH and LD_LIBRARY_PATH
/usr/bin/env -i \
    PATH=/usr/local/cuda/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    /clusterfs/homelab-heterogenous-HPC/bin/arm/bin_orin/gpu/mandelbrot_load_balanced_cuda_orin