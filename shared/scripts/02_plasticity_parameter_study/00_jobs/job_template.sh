#!/bin/bash
#SBATCH -J {JOB_NAME}
#SBATCH -A project02338
#SBATCH -t 1200  # "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#SBATCH --mem-per-cpu=5000
#SBATCH -n 16
#SBATCH -e /home/as12vapa/dolfiny_alex/shared/scripts/02_plasticity_parameter_study/{FOLDER_NAME}/%x.err.%j
#SBATCH --mail-type=End
#SBATCH -C i01

# # cd $HPC_SCRATCH
cd /home/as12vapa/dolfiny_alex

# Parameters for simulation_script.py (passed as command-line arguments)
srun -n 16 apptainer exec --bind ./shared:/home alex-dolfiny.sif python3 /home/scripts/02_plasticity_parameter_study/{FOLDER_NAME}/script.py \
    {TOTAL_COMPUTATIONS} \
    {CURRENT_COMPUTATION} 

EXITCODE=$?

# JobScript mit dem Status des wiss. Programms beenden
exit $EXITCODE