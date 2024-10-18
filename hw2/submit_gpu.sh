#!/bin/sh
#SBATCH --time=6:00:00          # Maximum run time in hh:mm:ss
#SBATCH --mem=16000             # Maximum memory required (in megabytes)
#SBATCH --job-name=default_479  # Job name (to track progress)
#SBATCH --partition=cse479      # Partition on which to run job
#SBATCH --gres=gpu:1            # Don't change this, it requests a GPU
#SBATCH --constraint=gpu_16gb   # will request a GPU with 16GB of RAM, independent of the type of card
#SBATCH --licenses=common

module load mamba
conda activate /common/cse479/shared/envs/tensorflow-env
$@