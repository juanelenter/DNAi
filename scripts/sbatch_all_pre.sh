while getopts ":ht" opt; do
  case ${opt} in
    h ) echo "USAGE: <number of fenos/envs> <base name> -f <format> -e <encoding> -i <imputation> -nan_flag <imputation flag> -ns  <nsplits>"
	 exit 1
      ;;
  esac
done
echo "USAGE: <number of fenos/envs> <base name> -f <format> -e <encoding> -i <imputation> -nan_flag <imputation flag> -ns  <nsplits>"
j=0
if [ $1 -gt 1 ]
then
	while [ $j -lt $1 ]
	do
		sbatch 10_splits_1_method_pre.sh ${*:2} -nf $j
		j=$[$j+1]
	done
else
	sbatch 10_splits_1_method_pre.sh ${*:2}
fi
