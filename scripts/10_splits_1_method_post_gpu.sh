#!/bin/bash
#SBATCH --job-name=predmu-svr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=118G
#SBATCH --time=32:00:00
#SBATCH --tmp=20G
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ihounie@fing.edu.uy
# First  argument is method (lasso, gbm, etc)
# Second argument is Config file

## #SBATCH --gres=gpu:1

USR=$USER
mkdir -p /scratch/${USR}/dnai
cp -r  ~/dnai /scratch/${USR}/
# Change to run less splits
singularity run --nv --cleanenv -B /scratch/${USR}/dnai:/dnai ~/dnai/scripts/dnai-classic.simg ~/dnai/scripts/run_all_post.sh $*







