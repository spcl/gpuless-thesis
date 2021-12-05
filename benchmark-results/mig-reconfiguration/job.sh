#!/bin/bash -l
#SBATCH --job-name="gpuless"
#SBATCH --account="g34"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lutobler@student.ethz.ch
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=amda100
#SBATCH --oversubscribe

module load nvhpc
module load cuda
module load python

srun python ./mig-reconfig-bench.py
