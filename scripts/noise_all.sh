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
#SBATCH --mail-user=ihounie@gmail.com
USR=$USER
#Asumo que el pre-proc ya se hizo
#cd /scratch/${USR}/dnai/predictors
cd ../predictors
CFG="${2}.json"
echo "${*}"
echo "python -W ignore noise_study.py $1 $2 -n_iter 75 -n_jobs 15 -log_exp comet -log_proj crossa_noise -cg $CFG -nt $3 ${*:4}"
python -W ignore noise_study.py $1 $2 -n_iter 75 -n_jobs 15 -cg $CFG -nt $3 ${*:4}

