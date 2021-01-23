#python main.py gnn -l -bs 4 -lr 0.001 -e=200 -vi 20
#python main.py gnn2 -l -bs 16 -lr 0.001 -e=200 -vi 20
python main3.py linear -l -bs 64 -lr 0.001 -e=200 -vi 50 -lrd 0.98
python main3.py mfgf -l -bs 64 -lr 0.001 -e=200 -vi 50 -lrd 0.9 -es 50
python main3.py fcn -l -bs 64 -lr 0.000935 -e=500 -es 30 -lrd 0.9 -vi 50
#python main.py linear -l -bs 64 -lr 0.001 -e=200 -vi 5

