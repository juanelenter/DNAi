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
USR=$USR
#Asumo que el pre-proc ya se hizo
CFG="${4}.json"
#echo "$CFG"
USE_CFG=0
if [ $2 == "None" ]
then
	PROJ="${1}"
else 
	PROJ="${1}"
fi
n_iter=1
n_jobs=1
log_exp=1
log_proj=1
shft=5
while getopts ":ht" opt; do
	case ${opt} in
		cg ) USE_CFG=1
		;;
		log_proj ) log_proj=0
	esac
done
#echo "${@:5}"
#shift $((OPTIND))
echo "$*"
echo "$shft"
#shft=5+$OPTIND
cargs="${@:$shft}"
#echo "${@:$shft}"
# load default arguments
if [ "$USE_CFG"="1" ]
then
cargs="${@:6}"
cargs+=" -cg ${CFG}"
fi
if [ $log_proj ]
then
cargs+=" -log_proj ${PROJ}"
fi


echo "$3 $4 ${cargs}"
cd /dnai/predictors
python -W ignore train_test_new_v3.py $3 $4 ${cargs}

