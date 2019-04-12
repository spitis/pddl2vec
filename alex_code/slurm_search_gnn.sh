#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:1

cd ..
source add_python_path.sh
cd alex_code

source /h/alexadam/anaconda3/bin/activate gnn

graph_file="logistics/43/problogistics-6-1.edgelist"

python -u gnn/train.py

