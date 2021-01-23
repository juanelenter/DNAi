export PATH=/opt/conda/envs/benchmark_gnn/bin:$PATH
source activate benchmark_gnn
cd ~/dnai/GNN
python hyper_optimize.py gnn 
