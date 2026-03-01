A collection of frequently used slurm commands.

1. sudo systemctl start/stop/status slurmctld - for the slurm controller module.
2. sudo systemctl start/stop/status slurmd - for the slurm worker module.
3. sudo scontrol update NodeName=NODE-NAME State=RESUME - to change slurm node status from down -> idle, useful after a restart.