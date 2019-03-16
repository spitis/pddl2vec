from pyperplan import _parse, _ground, get_heuristics, _get_heuristic_name, search
from argparse import ArgumentParser
import networkx as nx
import numpy as np
from argparse import ArgumentParser
from dotenv import load_dotenv, find_dotenv

import json
import os
import timeit
from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--domain-file", default="logistics/43/domain.pddl", type=str)
parser.add_argument("--problem-file", default="logistics/43/problogistics-6-1.pddl", type=str)


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    
    return wrapped


def solve_problem(task, h):
    solution = search.astar_search(task, h)
    
    return solution


def main(args):
    load_dotenv(find_dotenv(), override=True)
    pddl_dir = os.environ.get("PDDL_DIR")
    pddl_dir = os.path.join(ROOT_DIR, pddl_dir)
                    
    problem = _parse(domain_file=os.path.join(pddl_dir, args.domain_file),
                     problem_file=os.path.join(pddl_dir,args.problem_file))
    task = _ground(problem)
    
    heuristics = {_get_heuristic_name(heur): heur for heur in get_heuristics()}
    times = {}
    
    for heuristic in ["hadd", "hmax"]:
        wrapped = wrapper(solve_problem, task, heuristics[heuristic](task))
        times[heuristic] = timeit.timeit(wrapped, number=10)
        
    result_path = os.environ.get("BASELINES_DIR")
    result_path = os.path.join(ROOT_DIR, result_path)
    result_path = os.path.join(result_path, os.path.dirname(args.problem_file))
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    result_path = os.path.join(result_path, os.path.basename(args.problem_file).split(".")[0] + ".json")
    
    with open(result_path, "w") as fp:
        json.dump(times, fp, indent=4, sort_keys=True)
    
if __name__ == "__main__":
    main(parser.parse_args())