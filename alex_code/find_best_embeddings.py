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



def get_best_results(result_dir):
    # Create graph from edges
    keys = ["pearson_cor", "pearson_cor_inv_dist", "pearson_cor_inv_dot", "spearman_cor", "spearman_cor_inv_dist", "spearman_cor_inv_dot"]
    best = {key: {"val": 0.0, "path": None} for key in keys}
    
    files = os.listdir(result_dir)
    
    for f in files:
        with open(os.path.join(result_dir, f), "r") as read_file:
            res = json.load(read_file)
        
        for key in keys:
            if np.abs(res[key][0]) > best[key]["val"]:
                best[key]["val"] = np.abs(res[key][0])
                best[key]["path"] = os.path.join(result_dir, f)
                
    return best


def main(args):
    load_dotenv(find_dotenv(), override=True)
    
    problem_name = os.path.basename(args.problem_path).split(".")[0]
    
    result_dir = os.environ.get("EMBEDDING_EVALUATION_DIR")
    result_dir = os.path.join(ROOT_DIR, result_dir, os.path.dirname(args.problem_path))
    
    best = get_best_results(result_dir)
    pprint.pprint(best)

    
if __name__ == "__main__":
    main(parser.parse_args())