#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=2000



cd ..
source add_python_path.sh
cd alex_code

# Replace with your conda path and name of your conda env
source /h/alexadam/anaconda3/bin/activate gnn

domain=logistics
competition=38
problems=$(ls ../pddl_files/${domain}/${competition})

count=0
heuristics=(hadd lmcut blind)
time_limits=(30  60 90)

for time_limit in ${time_limits[@]}
do
    for heuristic in ${heuristics[@]}
    do
        for problem in ${problems[@]}
        do
            if [ "$problem" == "domain.pddl" ]
            then
                continue
            fi

            echo $problem

            domain_file="${domain}/${competition}/domain.pddl"
            problem_file="${domain}/${competition}/${problem}"

            python -u compute_baselines.py --domain-file=$domain_file --problem-file=$problem_file --heuristic=$heuristic --time-limit=$time_limit &
            count=$(( count + 1 ))

            if [ ${count} -gt 3 ]
            then
                count=0
                wait
            fi
        done
    done
done

