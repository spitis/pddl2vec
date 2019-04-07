from pyperplan import _parse, _ground
import numpy as np
import networkx as nx
import os

from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv

from settings import ROOT_DIR
from alex_code.utils.graph import expand_state_space_node2vec, gen_primes, hash_state
from alex_code.utils.save import write_pickle

parser = ArgumentParser()
parser.add_argument("--domain-path", default="logistics/43/domain.pddl", type=str)
parser.add_argument("--problem-path", default="logistics/43/problogistics-6-1.pddl", type=str)


def generate_token_mapping(problem):
    objects = sorted(list(problem.objects.keys()))
    predicates = sorted(list(problem.domain.predicates.keys()))
    actions = sorted(list(problem.domain.actions.keys()))
    
    tokens = actions + objects + predicates
    token_mapping = {token: p for token, p in zip(tokens, gen_primes())}
    
    return token_mapping


def write_edges(G, graph_file):
    with open(graph_file, "w") as f:
        for edge in list(G.edges):
            f.write("{} {}\n".format(edge[0], edge[1]))


def main(args):
    load_dotenv(find_dotenv(), override=True)
    
    pddl_dir = os.environ.get("PDDL_DIR")
    pddl_dir = os.path.join(ROOT_DIR, pddl_dir)
    
    domain_file = os.path.join(pddl_dir, args.domain_path)
    problem_file = os.path.join(pddl_dir, args.problem_path)
    
    problem = _parse(domain_file=domain_file, problem_file=problem_file)
    task = _ground(problem)

    print("Generating graph for: {}".format(args.problem_path))

    G, goal_node = expand_state_space_node2vec(task)
    node_mapping = {n: i for i, n in enumerate(list(G.nodes))}

    final_G = nx.Graph()

    for edge in list(G.edges):
        final_G.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
        
    graph_dir = os.environ.get("NODE2VEC_GRAPH_DIR")
    graph_dir = os.path.join(ROOT_DIR, graph_dir, os.path.dirname(args.problem_path))
    
    problem_name = os.path.basename(args.problem_path).split(".")[0]
    
    graph_file = os.environ.get("NODE2VEC_GRAPH_FILE")
    graph_file = os.path.join(graph_dir, graph_file.format(problem_name=problem_name))

    node_mapping_file = os.environ.get("NODE_MAPPING_FILE")
    node_mapping_file = os.path.join(graph_dir, node_mapping_file.format(problem_name=problem_name))

    goal_file = os.environ.get("GOAL_FILE")
    goal_file = os.path.join(graph_dir, goal_file.format(problem_name=problem_name))

    goal_number = node_mapping[hash_state(goal_node.state)]
    goal_number = {"idx": goal_number}

    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    write_edges(final_G, graph_file)
    write_pickle(node_mapping, node_mapping_file)
    write_pickle(goal_number, goal_file)
    

if __name__ == "__main__":
    main(parser.parse_args())