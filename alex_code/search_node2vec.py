from pyperplan import _parse, _ground, get_heuristics, _get_heuristic_name, search
from argparse import ArgumentParser
import networkx as nx
import numpy as np

from dotenv import find_dotenv, load_dotenv
from scipy.stats import spearmanr, pearsonr

from settings import ROOT_DIR
import os
import json
import timeit

from evaluate_embeddings import load_embeddings
from heuristics.node2vec import Node2VecHeuristic
from compute_baselines import wrapper, solve_problem

from alex_code.utils.save import read_pickle

parser = ArgumentParser()
parser.add_argument("--domain-file", default="logistics/43/domain.pddl", type=str)
parser.add_argument("--problem-file", default="logistics/43/problogistics-6-1.pddl", type=str)
parser.add_argument("--d", default=20, type=int)
parser.add_argument("--l", default=20, type=int)
parser.add_argument("--r", default=2, type=int)
parser.add_argument("--k", default=3, type=int)
parser.add_argument("--e", default=1, type=int)
parser.add_argument("--p", default=1, type=float)
parser.add_argument("--q", default=1, type=float)
parser.add_argument("--directed", default="dr", type=str,
                   choices=["dr", "u"])


def main(args):
    load_dotenv(find_dotenv(), override=True)
    
    problem_name = os.path.basename(args.problem_file).split(".")[0]
    
    pddl_dir = os.environ.get("PDDL_DIR")
    pddl_dir = os.path.join(ROOT_DIR, pddl_dir)
    
    domain_file = os.path.join(pddl_dir, args.domain_file)
    problem_file = os.path.join(pddl_dir, args.problem_file)
    
    problem = _parse(domain_file=domain_file, problem_file=problem_file)
    task = _ground(problem)
    
    graph_dir = os.environ.get("NODE2VEC_GRAPH_DIR")
    graph_dir = os.path.join(ROOT_DIR, graph_dir, os.path.dirname(args.problem_file))
    
    node_mapping_file = os.environ.get("NODE_MAPPING_FILE")
    node_mapping_file = os.path.join(graph_dir, node_mapping_file.format(problem_name=problem_name))
    
    print(node_mapping_file)

    goal_file = os.environ.get("GOAL_FILE")
    goal_file = os.path.join(graph_dir, goal_file.format(problem_name=problem_name))
    
    embedding_dir = os.environ.get("EMBEDDINGS_DIR")
    embedding_dir = os.path.join(ROOT_DIR, embedding_dir, os.path.dirname(args.problem_file))
    embedding_file = os.environ.get("EMBEDDING_FILE")
    
    embedding_path = os.path.join(embedding_dir, embedding_file.format(problem_name=problem_name, d=args.d,  l=args.l, r=args.r, k=args.k,
                                                                      e=args.e, p=args.p, q=args.q, directed=args.directed))
    
    result_dir = os.environ.get("NODE2VEC_HEURISTIC_RESULT_DIR")
    result_dir = os.path.join(ROOT_DIR, result_dir, os.path.dirname(args.problem_file))
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    result_file = os.environ.get("NODE2VEC_HEURISTIC_RESULT_FILE")
    result_path = os.path.join(result_dir, result_file.format(problem_name=problem_name, d=args.d, l=args.l, r=args.r, k=args.k,
                                                             e=args.e, p=args.p, q=args.q, directed=args.directed))
    
    results = {"node2vec": {"time": None, "expansions": None}}
    
    embeddings = load_embeddings(embedding_path)
    node_mapping = read_pickle(node_mapping_file)
    goal = read_pickle(goal_file)["idx"]
    
    heuristic = Node2VecHeuristic(task, embeddings, node_mapping, goal)
    
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