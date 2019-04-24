from argparse import ArgumentParser
import networkx as nx
import numpy as np

from dotenv import find_dotenv, load_dotenv
from scipy.stats import spearmanr, pearsonr

from settings import ROOT_DIR
import os
import json
import pprint

parser = ArgumentParser()
parser.add_argument("--problem-path", default="logistics/43/problogistics-6-1.pddl", type=str)
parser.add_argument("--search", default="gbfs", type=str, choices=["astar", "gbfs"])



def get_best_results(result_dir, problem_name, search):
    # Create graph from edges
    keys = ["expansions", "time"]
    best = {key: {"val": float("inf"), "path": None} for key in keys}
    
    files = os.listdir(result_dir)
    
    for f in files:
        if not "{}~".format(problem_name) in f:
            continue

        with open(os.path.join(result_dir, f), "r") as read_file:
            res = json.load(read_file)
            res = res[search]
        
        for key in keys:
            if np.abs(res[key]) < best[key]["val"]:
                best[key]["val"] = np.abs(res[key])
                best[key]["path"] = os.path.join(result_dir, f)
                
    return best


def main(args):
    load_dotenv(find_dotenv(), override=True)
    
    problem_name = os.path.basename(args.problem_path).split(".")[0]
    
    result_dir = os.environ.get("NODE2VEC_HEURISTIC_RESULT_DIR")
    result_dir = os.path.join(ROOT_DIR, result_dir, os.path.dirname(args.problem_path))
    
    best = get_best_results(result_dir, problem_name, args.search)
    pprint.pprint(best)

    
if __name__ == "__main__":
    main(parser.parse_args())