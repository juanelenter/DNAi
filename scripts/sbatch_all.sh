for i in "ridge" "svm" "gbm" "rf" "RFECV11"
do
sbatch 10_splits_1_method.sh ${i} $1
done
