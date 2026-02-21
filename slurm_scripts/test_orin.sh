#!/bin/bash
#SBATCH --job-name=test_orin
#SBATCH --nodelist=jetson-orin
#SBATCH --gres=gpu:1

# Total environment scrub
unset CUDA_VISIBLE_DEVICES
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Testing Orin Hardware Access..."
/clusterfs/homelab-heterogenous-HPC/bin/arm/bin_orin/gpu/mandelbrot_load_balanced_cuda_orin