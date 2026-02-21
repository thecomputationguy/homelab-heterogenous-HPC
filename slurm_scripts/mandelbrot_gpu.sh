#!/bin/bash
#SBATCH --job-name=mandelbrot_heterogeneous
#SBATCH --output=mandelbrot_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# --- SAFETY CHECK: Ensure the filesystem is actually there ---
if [ ! -d "/clusterfs/HPC_development" ]; then
    echo "ERROR: /clusterfs is empty or not mounted on $HOSTNAME!"
    exit 1
fi

# Determine node type from GRES (matches your iris/orin/nano types)
# Improved grep to handle cases where multiple GRES might be listed
NODE_TYPE=$(scontrol show node $(hostname) | grep -oP 'Gres=gpu:\K[a-z0-9]+' | head -n 1)

echo "Running on $HOSTNAME with GPU type: $NODE_TYPE"

case $NODE_TYPE in
  "iris")
    # Intel NUC path
    source /opt/intel/oneapi/setvars.sh
    BINARY="/clusterfs/homelab-heterogenous-HPC/bin/x86_64/gpu/mandelbrot_load_balanced_sycl"
    ;;
  "orin")
    # Jetson Orin path
    BINARY="/clusterfs/homelab-heterogenous-HPC/bin/arm/bin_orin/gpu/mandelbrot_load_balanced_cuda"
    ;;
  "nano")
    # Jetson Nano path
    BINARY="/clusterfs/homelab-heterogenous-HPC/bin/arm/bin_nano/gpu/mandelbrot_load_balanced_cuda"
    ;;
  *)
    echo "No compatible GPU type ($NODE_TYPE) found on this node."
    exit 1
    ;;
esac

# Final verification before execution
if [ -f "$BINARY" ]; then
    chmod +x "$BINARY"
    echo "Executing: $BINARY"
    $BINARY
else
    echo "ERROR: Binary not found at $BINARY"
    exit 1
fi