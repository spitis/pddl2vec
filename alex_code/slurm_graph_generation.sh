#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=5
#SBATCH --mem-per-cpu=2000

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

    domain_path="${domain}/${competition}/domain.pddl"
    problem_path="${domain}/${competition}/${problem}"

    python -u create_graph_from_pddl.py --domain-path=$domain_path --problem-path=$problem_path &
    count=$(( count + 1 ))

    if [ ${count} -gt 5 ]
    then
        count=0
        wait
    fi
done