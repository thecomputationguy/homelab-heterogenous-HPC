#!/bin/bash
#SBATCH --job-name=mandel_clean
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# The Magic Trick: Clear the environment variables Slurm sets for GPUs
# This forces the CUDA driver to re-scan the hardware like a fresh login.
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

# Direct launch without 'srun' to bypass the Cgroup sandbox
/clusterfs/homelab-heterogenous-HPC/bin/arm/bin_orin/gpu/mandelbrot_load_balanced_cuda_orin