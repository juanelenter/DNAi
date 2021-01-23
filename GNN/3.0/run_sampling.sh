NUM_RUNS=1
for dr in 0.97 0.9 0.8 0.99
do
    for w in "" "-pres"
    do
        for sr in 0.1 0.05 0.3 0.01
        do
            for cr in 0.9 0.5 0.98
            do
                pushd ../
                python geno_to_gso.py  data/yeast/geno_yeast_congored.npy data/yeast/pheno_yeast_congored.npy -rs -sr ${sr} -dr ${dr} ${w} -c -cr ${cr}
                popd
                    j=0
                    while [ $j -lt $NUM_RUNS ]
                        do
                        for model in "gnn" "fcn" "mfgf" "linear" "gnn2"
                            do
                            echo "python3 main3.py ${model} -lr 0.000935 -b1 0.9 -b2 0.9998 -lrd 0.96 -e 15 -es 20 -bs 4 -l -n gnn-yeast-v2"
                            python3 main3.py ${model} -lr 0.000935 -b1 0.9 -b2 0.9998 -lrd 0.96 -e 100 -es 20 -bs 32 -l -n gnn-yeast-v2
                            done
                        j=$[$j+1]
                    done
            done
        done
    done
done