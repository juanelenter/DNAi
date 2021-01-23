echo "USAGE: <n fenos/env> <config_name sin el .json> <noise type (e.g: drop_markers)> <opt args>"
for m in "gbm" "rf" "ridge" "svm"
do
	if [ $1 -gt 0 ]
	then
		j=0
		while [ $j -lt $1 ]
		do
			if [ $m == "svm" ]
			then
        			sbatch noise_gpu.sh ${m} ${2}_${j}.json ${*:3}
    				echo "sbatch noise_gpu.sh ${m} ${2}_${j}.json ${*:3}"
			else
				sbatch noise.sh ${m} ${2}_${j}.json ${*:3}
				echo "sbatch noise.sh ${m} ${2}_${j}.json ${*:3}"
			fi
			j=$[$j+1]
		done
	elif [ $m == "svm" ]
	then
		echo "batch noise_gpu.sh ${m} ${2}.json ${*:3}"
		sbatch noise_gpu.sh ${m} ${2}.json ${*:3}
	else
		echo "sbatch noise.sh ${m} ${2}.json ${*:3}"
                sbatch noise.sh ${m} ${2}.json ${*:3}
	fi
done
