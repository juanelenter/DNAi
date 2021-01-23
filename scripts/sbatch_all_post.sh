while getopts ":ht" opt; do
  case ${opt} in
    h ) echo "USAGE: <n fenos/env> <nombre base> <model> <config_name sin el .json>  <-g> <-f> <-cg>"
	 exit 1
      ;;
  esac
done
echo "USAGE: <n fenos/env> <nombre base> <model (all para correr todos)> <config_name sin el .json>  <-g> <-f> <-cg>"
j=0
#sleep 3h
if [ $1 -gt 1 ]
then
        while [ $j -lt $1 ]
	do
		if [ $3 == "all" ]
		then
			for i in "ridge" "gbm" "rf" "RFECV9" "SGD_regressor" "bayes_skl"
			do
				echo "$2 ${j} ${3}_${j}.json ${i} ${*:5}"
				sbatch 10_splits_1_method_post.sh $2 ${j} ${4}_${j}.json ${i} ${*:5}
			done
        	elif [ "$3" != "svm" ]
        	then
			echo "sbatch 10_splits_1_method_post.sh $2 ${j} ${4}_${j}.json $3 ${*:5}"
			sbatch 10_splits_1_method_post.sh $2 ${j} ${4}_${j}.json $3 ${*:5}
		else
			echo "sbatch 10_splits_1_method_post_gpu.sh $2 ${j} ${4}_${j}.json $3 ${*:5}"
                        sbatch 10_splits_1_method_post_gpu.sh $2 ${j} ${4}_${j}.json $3 ${*:5}

		fi
	j=$[$j+1]
	done
else
	if [ "$3" == "all" ]
        then
		for i in "bayes_skl" "SGD_regressor" "ridge" "gbm" "rf" "RFECV9"
                do
                        echo "sbatch 10_splits_1_method_post.sh $2 None ${4}.json ${i} ${*:5}"
			sbatch 10_splits_1_method_post.sh $2 0 ${4}.json ${i} ${*:5}
                        j=$[$j+1]
                done
	elif [ "$3" != "svm" ]
	then
		echo "sbatch 10_splits_1_method_post.sh $2 None ${4}.json $3 ${*:5}"
                sbatch 10_splits_1_method_post.sh $2 0 ${4}.json $3 ${*:5}
	else
		echo "sbatch 10_splits_1_method_post_gpu.sh $2 None ${4}.json $3 ${*:5}"
                sbatch 10_splits_1_method_post_gpu.sh $2 0 ${4}.json $3 ${*:5}
	fi
fi
