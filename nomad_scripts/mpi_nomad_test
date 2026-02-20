job "HAL-MPI-Nomad-Final" {
  datacenters = ["dc1"]
  type        = "batch" 

  group "hpc-cluster" {
    constraint {
      attribute = "${attr.unique.hostname}"
      value     = "intel-head"
    }

    task "mpi-launcher" {
      driver = "raw_exec"

      config {
        command = "mpirun"
        args = [
          "--tag-output",
          "--allow-run-as-root",
          # Explicitly force the user and the private key path
          "--mca", "plm_rsh_args", "-l rahul -i /home/rahul/.ssh/id_rsa -o StrictHostKeyChecking=no", 
          "--mca", "btl_tcp_if_include", "10.0.0.0/24",
          "--mca", "oob_tcp_if_include", "10.0.0.0/24",
          "--mca", "routed", "direct", 
          "--mca", "plm_rsh_no_tree_spawn", "1",
          "--mca", "pml", "ob1",
          "--mca", "btl", "self,tcp",
          "--host", "10.0.0.1", "-np", "1", "/clusterfs/HPC_development/homelab-heterogenous-HPC/bin/x86_64/cpu/run_mpi_test_x86",
          ":",
          "--host", "10.0.0.2,10.0.0.3,10.0.0.4", "-np", "3", "/clusterfs/bin/arm/bin_nano/mpi_test_arm"
        ]
      }
    }
  }
}