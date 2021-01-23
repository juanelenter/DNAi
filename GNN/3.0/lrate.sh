for i in 0.001 0.0009 0.00001 0.01
do
    python3 main3.py gnn-pool-2 -lr ${i} -b1 0.9 -b2 0.9998 -lrd 0.96 -e 30 -es 10 -bs 64 -l
done