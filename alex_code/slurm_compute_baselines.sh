#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=7000



cd ..
source add_python_path.sh
cd alex_code

source /h/alexadam/anaconda3/bin/activate gnn

domain=logistics
competition=38
problems=$(ls ../pddl_files/${domain}/${competition})

count=0

for problem in ${problems[@]}
do
    if [ "$problem" == "domain.pddl" ]
    then
        continue
    fi

    echo $problem

    domain_file="${domain}/${competition}/domain.pddl"
    problem_file="${domain}/${competition}/${problem}"

    python -u compute_baselines.py --domain-file=$domain_file --problem-file=$problem_file &
    count=$(( count + 1 ))

    if [ ${count} -gt 1 ]
    then
        count=0
        wait
    fi
done