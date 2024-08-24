#!/bin/bash

#SBATCH --job-name="soundactions"
#SBATCH --time=24:00:00     # walltime
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=rtx30:1

module load Miniconda3/22.11.1-1
conda activate /projects/ec12/jinyueg/conda/envs/soundactions

echo "=> Python path:"
which python

echo "=> Available GPU(s):"
nvidia-smi -L
echo 


cd /projects/ec12/jinyueg/SoundActions/soundactions/soundactions
python finetune_cls.py