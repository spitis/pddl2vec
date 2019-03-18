#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=15
#SBATCH --mem-per-cpu=1000

cd ..
source add_python_path.sh
cd alex_code

source /h/alexadam/anaconda3/bin/activate gnn

problem_path="logistics/43/problogistics-6-1.pddl"
dimensions=(20 35 50 65 80 95 110 128)
lengths=(20 40 60 80)
num_walks=(2 5 8 10)
context_sizes=(3 5 7 10)

count=0

for d in ${dimensions[@]}
do
    for l in ${lengths[@]}
    do
        for r in ${num_walks[@]}
        do
            for k in ${context_sizes[@]}
            do
                python -u evaluate_embeddings.py --problem-path=$problem_path --d=$d --l=$l --r=$r --k=$k &
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
