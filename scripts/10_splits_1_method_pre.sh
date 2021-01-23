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
 
USR="ihounie"
#mkdir -p /scratch/${USR}
#cp -v -r -u  ~/dnai /scratch/${USR}
# Change to run less splits
singularity run --cleanenv ~/dnai/scripts/dnai-r.simg ~/dnai/scripts/pre.sh $*
