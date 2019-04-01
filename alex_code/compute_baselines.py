from pyperplan import _parse, _ground, get_heuristics, _get_heuristic_name, search
from argparse import ArgumentParser
import networkx as nx
import numpy as np
from argparse import ArgumentParser
from dotenv import load_dotenv, find_dotenv
from contextlib import contextmanager

import json
import os
import signal
import timeit
from settings import ROOT_DIR

from alex_code.utils.save import read_json

parser = ArgumentParser()
parser.add_argument("--domain-file", default="logistics/38/domain.pddl", type=str)
parser.add_argument("--problem-file", default="logistics/38/prob02.pddl", type=str)
parser.add_argument("--heuristic", default="hadd", type=str,
                    choices=["blind", "hadd", "hff", "landmark", "lmcut"])
parser.add_argument("--time-limit", default=10, type=int)


class TimeoutException(Exception): pass


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    
    return wrapped


def solve_problem(task, h):
    solution, expansions = search.astar_search(task, h)
    
    return solution, expansions


def run_baseline(results, task, h, identifier):
    wrapped = wrapper(solve_problem, task, h)
    results[identifier]["time"] = timeit.timeit(wrapped, number=1)
    solution, expansions = solve_problem(task, h)

    if solution is None:
        raise Exception("Solution does not exist for this problem")

    results[identifier]["expansions"] = expansions


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def main(args):
    load_dotenv(find_dotenv(), override=True)
    pddl_dir = os.environ.get("PDDL_DIR")
    pddl_dir = os.path.join(ROOT_DIR, pddl_dir)
                    
    problem = _parse(domain_file=os.path.join(pddl_dir, args.domain_file),
                     problem_file=os.path.join(pddl_dir,args.problem_file))
    task = _ground(problem)
    
    heuristics = {_get_heuristic_name(heur): heur for heur in get_heuristics()}

    result_path = os.environ.get("BASELINES_DIR")
    result_path = os.path.join(ROOT_DIR, result_path)
    result_path = os.path.join(result_path, os.path.dirname(args.problem_file))
    result_path = os.path.join(result_path, os.path.basename(args.problem_file).split(".")[0] + ".json")

    identifier = "{heuristic}_{time_limit}s".format(heuristic=args.heuristic, time_limit=args.time_limit)

    if os.path.exists(result_path):
        results = read_json(result_path)
        results[identifier] = {"time": None, "expansions": None}
    else:
        results = {identifier: {"time": None, "expansions": None}}

        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))

    try:
        with time_limit(args.time_limit):
            run_baseline(results, task, heuristics[args.heuristic](task), identifier)
            results[identifier]["timed_out"] = False
    except TimeoutException as e:
        print("Timed out")
        results[identifier]["timed_out"] = True

    with open(result_path, "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    main(parser.parse_args())