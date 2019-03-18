#! /bin/bash

cd ..
source add_python_path.sh
cd scripts

graph_file="logistics/43/problogistics-6-1.edgelist"
dimensions=(20 35 50 65 80 95 110 128)
lengths=(20 40 60 80)
num_walks=(2 5 8 10)
context_sizes=(3 5 7 10)

for d in ${dimensions[@]}
do
    for l in ${lengths[@]}
    do
        for r in ${num_walks[@]}
        do
            for k in ${context_sizes[@]}
            do
                python run_node2vec.py --graph-file=$graph_file --d=$d --l=$l --r=$r --k=$k
            done
        done
    done
done
