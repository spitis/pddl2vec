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
from alex_code.gnn.model_loading import create_model
from alex_code.gnn.regression import RegressionGCN
from alex_code.compute_baselines import wrapper

from alex_code.utils.load import find_weight_dict

parser = ArgumentParser()
parser.add_argument("--domain-file", default="logistics/43/domain.pddl", type=str)
parser.add_argument("--problem-file", default="logistics/43/problogistics-6-1.pddl", type=str)
parser.add_argument("--graph-path", default="logistics/43/problogistics-6-1.p", type=str)

parser.add_argument("--epochs", default=200, dest="epochs", type=int)
parser.add_argument("--batch-size", default=1000, dest="batch_size", type=int)
parser.add_argument("--normalization", default="features", dest="normalization", choices=["none", "normalize",
                                                                                           "features", "samples"])
parser.add_argument("--seed", default=219, dest="seed")
parser.add_argument("--lr", default=0.01, dest="lr", type=float)
parser.add_argument("--model", default="deepgcn", dest="model", type=str, choices=["arma", "gcn", "deepgcn"])
parser.add_argument("--directed", default="undirected", type=str, choices=["directed", "undirected"])
parser.add_argument("--activation", default="selu", type=str, choices=["sigmoid", "tanh", "relu", "elu", "selu"])
parser.add_argument("--comparison-heuristic", default="hadd", type=str, choices=["hadd", "hmax"])
parser.add_argument("--comparison-limit", default=200, type=int)
parser.add_argument("--comparison-switch", default=True)


def track_heuristics(task, heuristics, limit):
    solution, expansions, heuristic_values = search.astar_search_comparison(task, heuristics, limit=limit)

    return solution, expansions, heuristic_values


def print_comparison(heuristic_values):
    for i in range(len(heuristic_values["main"])):
        print("{} | {}".format(heuristic_values["main"][i], heuristic_values["comp"][i]))


def main(args):
    load_dotenv(find_dotenv(), override=True)

    problem_name = os.path.basename(args.problem_file).split(".")[0]

    pddl_dir = os.environ.get("PDDL_DIR")
    pddl_dir = os.path.join(ROOT_DIR, pddl_dir)

    domain_file = os.path.join(pddl_dir, args.domain_file)
    problem_file = os.path.join(pddl_dir, args.problem_file)

    problem = _parse(domain_file=domain_file, problem_file=problem_file)
    task = _ground(problem)

    models_dir = os.environ.get("GNN_MODEL_DIR")
    models_dir = os.path.join(ROOT_DIR, models_dir, os.path.dirname(args.graph_path))

    model_path, stats_path, model_id = find_weight_dict(models_dir, problem_name, args)

    stats = read_json(stats_path)
    state_dict = torch.load(model_path)

    model = create_model(args.model, args.activation, stats["num_features"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(state_dict)
    model = model.to(device)

    result_dir = os.environ.get("GNN_HEURISTIC_SUBDIR")
    result_dir = os.path.join(ROOT_DIR, result_dir.format(problem_name=os.path.dirname(args.problem_file),
                                                          model_id=model_id))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_file = os.environ.get("GNN_HEURISTIC_RESULT_FILE")
    result_path = os.path.join(result_dir, result_file)

    result_stats_file = os.environ.get("GNN_STATS_FILE")
    result_stats_path = os.path.join(result_dir, result_stats_file)

    results = {"node2vec": {"time": None, "expansions": None}}

    all_heuristics = {_get_heuristic_name(heur): heur for heur in get_heuristics()}

    gnn_heuristic = GNNHeuristic(problem, task, model, args.directed, args.normalization, device)

    if args.comparison_switch:
        heuristics = {"comp": gnn_heuristic, "main": all_heuristics[args.comparison_heuristic](task)}
    else:
        heuristics = {"main": gnn_heuristic, "comp": all_heuristics[args.comparison_heuristic](task)}

    solution, expansions, heuristic_values = track_heuristics(task, heuristics, args.comparison_limit)

    print_comparison(heuristic_values)

    print("pearson correlation between heuristics: {}".format(pearsonr(heuristic_values["main"], heuristic_values[
        "comp"])))
    print("spearman correlation between heuristics: {}".format(spearmanr(heuristic_values["main"], heuristic_values[
        "comp"])))

    # if solution is None:
    #     raise Exception("Solution does not exist for this problem")
    #
    # results["node2vec"]["expansions"] = expansions
    #
    # print(results)
    #
    # with open(result_path, "w") as fp:
    #     json.dump(results, fp, indent=4, sort_keys=True)
    #
    # with open(result_stats_path, "w") as fp:
    #     json.dump(args.__dict__, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    main(parser.parse_args())