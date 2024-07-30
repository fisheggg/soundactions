#!/bin/bash

#SBATCH --job-name="soundactions"
#SBATCH --time=24:00:00     # walltime
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=1

module load Miniconda3/22.11.1-1
conda activate soundactions
cd /projects/ec12/jinyueg/SoundActions/soundactions/soundactions
python finetune.py