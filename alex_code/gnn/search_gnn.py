from pyperplan import _parse, _ground, get_heuristics, _get_heuristic_name, search
from argparse import ArgumentParser
import networkx as nx
import numpy as np

from dotenv import find_dotenv, load_dotenv
from scipy.stats import spearmanr, pearsonr

from settings import ROOT_DIR
import os
import json
import torch
import timeit

from heuristics.gnn import GNNHeuristic

from alex_code.utils.save import read_pickle, read_json
from alex_code.gnn.regression import RegressionGCN
from alex_code.compute_baselines import wrapper, solve_problem

parser = ArgumentParser()
parser.add_argument("--domain-file", default="logistics/43/domain.pddl", type=str)
parser.add_argument("--problem-file", default="logistics/43/problogistics-6-1.pddl", type=str)
parser.add_argument("--graph-path", default="logistics/43/problogistics-6-1.p", type=str)


def main(args):
    load_dotenv(find_dotenv(), override=True)

    problem_name = os.path.basename(args.problem_file).split(".")[0]

    pddl_dir = os.environ.get("PDDL_DIR")
    pddl_dir = os.path.join(ROOT_DIR, pddl_dir)

    domain_file = os.path.join(pddl_dir, args.domain_file)
    problem_file = os.path.join(pddl_dir, args.problem_file)

    problem = _parse(domain_file=domain_file, problem_file=problem_file)
    task = _ground(problem)

    model_dir = os.environ.get("GNN_MODEL_DIR")
    model_dir = os.path.join(ROOT_DIR, model_dir, os.path.dirname(args.graph_path))

    model_file = os.environ.get("GNN_MODEL_FILE")
    model_path = os.path.join(model_dir, model_file.format(problem_name=problem_name))

    stats_file = os.environ.get("GNN_STATS_FILE")
    stats_path = os.path.join(model_dir, stats_file.format(problem_name=problem_name))

    stats = read_json(stats_path)
    state_dict = torch.load(model_path)
    model = RegressionGCN(stats["num_features"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(state_dict)
    model = model.to(device)

    result_dir = os.environ.get("GNN_HEURISTIC_RESULT_DIR")
    result_dir = os.path.join(ROOT_DIR, result_dir, os.path.dirname(args.problem_file))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_file = os.environ.get("GNN_HEURISTIC_RESULT_FILE")
    result_path = os.path.join(result_dir, result_file.format(problem_name=problem_name))

    results = {"node2vec": {"time": None, "expansions": None}}

    heuristic = GNNHeuristic(problem, task, model, device)

    wrapped = wrapper(solve_problem, task, heuristic)
    results["node2vec"]["time"] = timeit.timeit(wrapped, number=1)
    solution, expansions = solve_problem(task, heuristic)

    if solution is None:
        raise Exception("Solution does not exist for this problem")

    results["node2vec"]["expansions"] = expansions

    print(results)

    with open(result_path, "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    main(parser.parse_args())