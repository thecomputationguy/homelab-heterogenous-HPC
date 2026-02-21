job "HAL-Mandelbrot-Load-Balanced" {
  datacenters = ["dc1"]
  type        = "batch"

  group "hpc-cluster" {
    constraint {
      attribute = "${attr.unique.hostname}"
      value     = "intel-head"
    }

    task "mandelbrot-launcher" {
      driver = "raw_exec"

      config {
        command = "mpirun"
        args = [
          "--tag-output",
          "--allow-run-as-root",
          "--oversubscribe",
          "--use-hwthread-cpus",
          "--mca", "plm_rsh_args", "-l rahul -i /home/rahul/.ssh/id_rsa -o StrictHostKeyChecking=no",
          "--mca", "btl_tcp_if_include", "10.0.0.0/24",
          "--mca", "oob_tcp_if_include", "10.0.0.0/24",
          "--mca", "pml", "ob1",
          "--mca", "btl", "self,tcp",
          "--host", "10.0.0.1:32", 
          "-np", "32", "/clusterfs/HPC_development/homelab-heterogenous-HPC/bin/x86_64/cpu/mandelbrot_load_balanced",
          ":",
          "--host", "10.0.0.2:16,10.0.0.3:12,10.0.0.4:13", 
          "-np", "41", "/clusterfs/HPC_development/homelab-heterogenous-HPC/bin/arm/bin_nano/cpu/mandelbrot_load_balanced"
        ] # Closing the args list
      } # Closing config

      resources {
        cpu    = 500
        memory = 256
      }
    }
  }
}