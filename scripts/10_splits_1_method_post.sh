#!/bin/bash
#SBATCH --job-name=predmu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=110G
#SBATCH --time=1:00:00
#SBATCH --tmp=9G
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ihounie@fing.edu.uy
# First  argument is method (lasso, gbm, etc)
# Second argument is Config file
 
USR=$USER
mkdir -p /scratch/${USR}/dnai
cp -r  ~/dnai /scratch/${USR}/
# Change to run less splits
singularity run --cleanenv -B /scratch/${USR}/dnai:/dnai ~/dnai/scripts/dnai-r.simg ~/dnai/scripts/run_all_post.sh $*







