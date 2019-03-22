#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=15
#SBATCH --mem-per-cpu=1000

cd ..
source add_python_path.sh
cd alex_code

source /h/alexadam/anaconda3/bin/activate gnn

graph_file="logistics/43/problogistics-6-1.edgelist"
dimensions=(20 35 50 65 80 95 110 128)
lengths=(20 40 60 80)
num_walks=(2 5 8 10)
context_sizes=(3 5 7 10)
directions=(dr u)

#dimensions=(160 200)
#lengths=(100 120)
#num_walks=(10 15 20)
#context_sizes=(12 15 20)

count=0

for d in ${dimensions[@]}
do
    for l in ${lengths[@]}
    do
        for r in ${num_walks[@]}
        do
            for k in ${context_sizes[@]}
            do
                for directed in ${directions[@]}
                do
                    python -u run_node2vec.py --graph-file=$graph_file --d=$d --l=$l --r=$r --k=$k --directed=$directed &
                    count=$(( count + 1 ))

                    if [ ${count} -gt 14 ]
                    then
                        count=0
                        wait
                    fi
                done
            done
        done
    done
done
