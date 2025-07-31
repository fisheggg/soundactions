#!/bin/bash

#SBATCH --job-name="sa-test"
#SBATCH --time=24:00:00     # walltime
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=1

echo "=> Python path:"
which python

echo "=> Available GPU(s):"
nvidia-smi -L


cd /projects/ec12/jinyueg/SoundActions/soundactions
python test.py