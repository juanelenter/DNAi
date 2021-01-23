#!/bin/bash
#SBATCH --job-name=predmu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=110G
#SBATCH --time=2-24:00:00
#SBATCH --tmp=9G
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ihounie@gmail.com
# First  argument is method (lasso, gbm, etc)
# Second argument is Config file
 
USR=$USER
#mkdir -p /scratch/${USR}/$1
#cp -v -r -u  ~/dnai /scratch/${USR}/$1
# Change to run less splits
singularity run --cleanenv --nv ~/ggnn-latest.simg ~/dnai/GNN/run_all_cluster.sh
