#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=3000
#SBATCH --gres=gpu:1

cd ..
source add_python_path.sh
cd alex_code

source /h/alexadam/anaconda3/bin/activate gnn

problem_path="logistics/43/problogistics-6-1.pddl"
epochs=(100 200 500)
batch_sizes=(100 1000 10000)
lrs=(1.0 0.1 0.01 0.001)
normalizations=(none normalize)

count=0

for epoch in ${epochs[@]}
do
    for batch_size in ${batch_sizes[@]}
    do
        for lr in ${lrs[@]}
        do
            for normalization in ${normalizations[@]}
            do
                python -u gnn/train.py --epochs=$epochs --batch-size=$batch_size --lr=$lr --normalization=$normalization &
                count=$(( count + 1 ))

                sleep 2

                if [ ${count} -gt 3 ]
                then
                    count=0
                    wait
                fi
            done
        done
    done
done