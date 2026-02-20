This repository is a collection of code snippets and scripts designed for a home High-Performance Computing (HPC) cluster. It specifically focuses on managing workloads across heterogeneous hardware, utilizing both x86_64 and ARM architectures.

The project demonstrates parallel computing techniques by calculating the Mandelbrot set using several frameworks, including MPI for distributed CPU tasks, CUDA for NVIDIA GPUs, and SYCL for cross-platform acceleration. It also includes configuration for two different cluster orchestrators, Slurm and Nomad, to manage job scheduling and execution.

### Project Overview

* **Heterogeneous Support**: Contains source code and binaries optimized for various hardware, including Intel NUC (x86_64), NVIDIA Jetson Orin/Nano (ARM), and Raspberry Pi (ARM).
* **Parallel Computing**: Implements a load-balanced master-worker pattern using MPI to distribute intensive mathematical calculations across multiple network nodes.
* **GPU Acceleration**: Features GPU-specific implementations using CUDA (for NVIDIA devices) and SYCL (for general hardware acceleration).
* **Workload Management**: Provides Slurm shell scripts and Nomad job files for automated deployment and communication testing across a cluster.

### Directory Structure

```text
homelab-heterogenous-HPC/
├── bin/                          # Compiled binaries for different architectures
│   ├── arm/                      # Binaries for ARM-based nodes (Jetson, Pi)
│   │   ├── bin_nano/
│   │   └── bin_orin/
│   └── x86_64/                   # Binaries for x86-based nodes (NUC, Desktop)
├── nomad_scripts/                # Job files for HashiCorp Nomad orchestrator
├── slurm_scripts/                # Batch scripts for Slurm workload manager
│   ├── mandelbrot_load_balanced.sh
│   └── run_mpi_test.sh
├── src/                          # Source code for all implementations
│   ├── cpu/                      # C source code for MPI-based CPU tasks
│   │   ├── mandelbrot_load_balanced.c
│   │   └── run_mpi_test.c
│   └── gpu/                      # CUDA and SYCL source code for GPU acceleration
│       ├── mandelbrot_load_balanced_cuda.cu
│       └── mandelbrot_load_balanced_sycl.cpp
└── README.md                     # Project documentation

```