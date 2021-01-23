#!/bin/bash
#SBATCH --job-name=jersey
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=12000
#SBATCH --time=24:00:00
#SBATCH --tmp=12000
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ihounie@fing.edu.uy
USR="ihounie"
#Asumo que el pre-proc ya se hizo
cd /scratch/${USR}/dnai/predictors
CFG="${2}.json"
python train_test.py $1 $2 -n_iter 150 -n_jobs 15 -log_exp comet -s $3 -cg $CFG

