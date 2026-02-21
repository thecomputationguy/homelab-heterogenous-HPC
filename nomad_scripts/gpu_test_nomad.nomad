job "homelab-gpu-mandelbrot" {
  datacenters = ["dc1"]
  type        = "batch"

  # --- GROUP 1: JETSON ORIN (JetPack 6) ---
  group "orin-group" {
    constraint {
      attribute = "${meta.jetpack_version}"
      value     = "6.x"
    }

    count = 1

    # DISABLING ALL RETRIES
    restart {
      attempts = 0
      mode     = "fail"
    }
    reschedule {
      attempts  = 0
      unlimited = false
    }

    task "mandelbrot-orin" {
      driver = "docker"
      config {
        image   = "nvcr.io/nvidia/l4t-base:r36.2.0"
        runtime = "nvidia" 
        
        mount {
          type = "bind"
          source = "/clusterfs"
          target = "/app"
          readonly = false
        }
        command = "/bin/bash"
        args    = ["-c", "chmod +x /app/homelab-heterogenous-HPC/bin/arm/bin_orin/gpu/mandelbrot_load_balanced_cuda_orin && /app/homelab-heterogenous-HPC/bin/arm/bin_orin/gpu/mandelbrot_load_balanced_cuda_orin"]
      }
    }
  }

  # --- GROUP 2: JETSON NANO (JetPack 4) ---
  group "nano-group" {
    constraint {
      attribute = "${meta.jetpack_version}"
      value     = "4.x"
    }

    count = 1

    # DISABLING ALL RETRIES
    restart {
      attempts = 0
      mode     = "fail"
    }
    reschedule {
      attempts  = 0
      unlimited = false
    }

    task "mandelbrot-nano" {
      driver = "docker"
      config {
        image   = "nvcr.io/nvidia/l4t-base:r32.7.1"
        runtime = "nvidia"
        
        mount {
          type = "bind"
          source = "/clusterfs"
          target = "/app"
          readonly = false
        }
        command = "/bin/bash"
        args    = ["-c", "chmod +x /app/homelab-heterogenous-HPC/bin/arm/bin_nano/gpu/mandelbrot_load_balanced_cuda_nano && /app/homelab-heterogenous-HPC/bin/arm/bin_nano/gpu/mandelbrot_load_balanced_cuda_nano"]
      }
    }
  }

  # --- GROUP 3: INTEL NUC (Pocket AI RTX A500) ---
  group "nuc-a500-group" {
    constraint {
      attribute = "${attr.unique.hostname}"
      value     = "intel-head"
    }

    count = 1

    # DISABLING ALL RETRIES
    restart {
      attempts = 0
      mode     = "fail"
    }
    reschedule {
      attempts  = 0
      unlimited = false
    }

    task "mandelbrot-a500" {
      driver = "docker"
      config {
        image   = "nvidia/cuda:12.6.3-base-ubuntu24.04"
        runtime = "nvidia"
        
        mount {
          type = "bind"
          source = "/clusterfs"
          target = "/app"
          readonly = false
        }
        command = "/bin/bash"
        args    = ["-c", "chmod +x /app/homelab-heterogenous-HPC/bin/x86_64/gpu/mandelbrot_load_balanced_cuda && /app/homelab-heterogenous-HPC/bin/x86_64/gpu/mandelbrot_load_balanced_cuda"]
      }
      resources {
        device "nvidia/gpu" {
          count = 1
          constraint {
            attribute = "${device.model}"
            value     = "NVIDIA RTX A500 Embedded GPU"
          }
        }
      }
    }
  }

  # --- GROUP 4: INTEL NUC (Iris iGPU - Commented Out) ---
  /*
  group "nuc-iris-group" {
    constraint {
      attribute = "${meta.gpu_type}"
      value     = "intel_iris"
    }

    task "mandelbrot-intel" {
      driver = "docker"
      config {
        image = "ubuntu:22.04"
        
        # Pass Intel GPU devices to the container
        devices = [
          {
            host_path      = "/dev/dri"
            container_path = "/dev/dri"
          }
        ]
        
        mount {
          type = "bind"
          source = "/clusterfs"
          target = "/app"
          readonly = false
        }
        command = "/bin/bash"
        args    = ["-c", "chmod +x /app/homelab-heterogenous-HPC/bin/x86_64/gpu/mandelbrot_load_balanced_sycl && /app/homelab-heterogenous-HPC/bin/x86_64/gpu/mandelbrot_load_balanced_sycl"]
      }
    }
  }
  */
}