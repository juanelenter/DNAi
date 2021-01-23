#!/bin/bash
#SBATCH --job-name=predmu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=110G
#SBATCH --time=2-24:00:00
#SBATCH --tmp=9G
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ihounie@gmail.com
# First  argument is method (lasso, gbm, etc)
# Second argument is Config file
 
USR=$USER
#mkdir -p /scratch/${USR}/$1
#cp -v -r -u  ~/dnai /scratch/${USR}/$1
# Change to run less splits
echo "${*}"
echo "singularity run --cleanenv ~/dnai/scripts/dnai-classic.simg ~/dnai/scripts/noise_all.sh $2 $1 $i ${*:3}"
singularity run --cleanenv ~/dnai/scripts/dnai-classic.simg ~/dnai/scripts/noise_all.sh $2 $1 $i ${*:3}
