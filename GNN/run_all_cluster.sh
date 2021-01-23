cd ~/dnai/GNN
export PATH=/opt/conda/envs/benchmark_gnn/bin:$PATH
source activate benchmark_gnn
python main.py linear -l -bs 64 -lr 0.001 -e=1000 -vi 5
python main.py fcn -l -bs 4 -lr 0.000935 -e=500 -vi 5
python main.py gnn -l -bs 4 -lr 0.000935 -e=100 -vi 20
python main.py gnn2 -l -bs 4 -lr 0.000935 -e=100 -vi 20
python main.py mfgf -l -bs 4 -lr 0.000935 -e=100 -vi 5
