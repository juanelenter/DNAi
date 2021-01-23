for m in "gbm" "rf" "ridge"
do
    for i in {0..10}
    do
        sbatch noise.sh ${m} "yeast_no_codif_del_rows_2020-08-11_${i}.json"
    done
done|
