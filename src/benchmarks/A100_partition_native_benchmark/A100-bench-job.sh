#!/bin/bash -l
#SBATCH --job-name="gpuless"
#SBATCH --account="g34"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mingjli@student.ethz.ch
#SBATCH --time=01:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=amda100
#SBATCH --oversubscribe

module load cuda
module load python
source ~/testenv/bin/activate
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.8.0-fjdnxm6yggxxp75sb62xrxxmeg4s24ml/lib64 \
srun python ./A100-bench.py
