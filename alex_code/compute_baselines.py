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
parser.add_argument("--domain-file", default="logistics/43/domain.pddl", type=str)
parser.add_argument("--problem-file", default="logistics/43/problogistics-6-1.pddl", type=str)
parser.add_argument("--heuristic", default="lmcut", type=str,
                    choices=["blind", "hadd", "hff", "landmark", "lmcut"])
parser.add_argument("--time-limit", default=10, type=int)


class TimeoutException(Exception): pass


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    
    return wrapped


def search_astar(task, h):
    solution, expansions = search.astar_search(task, h)
    
    return solution, expansions


def run_baseline_astar(task, h):
    solution, expansions = search_astar(task, h)

    if solution is None:
        raise Exception("Solution does not exist for this problem")

    return expansions


def search_gbfs(task, h):
    solution, expansions = search.greedy_best_first_search(task, h)

    return solution, expansions


def run_baseline_gbfs(task, h):
    solution, expansions = search_gbfs(task, h)

    if solution is None:
        raise Exception("Solution does not exist for this problem")

    return expansions


def time_baseline(task, h, search_fn):
    wrapped = wrapper(search_fn, task, h)
    time = timeit.timeit(wrapped, number=1)

    return time


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
        results["astar"][identifier] = {"time": None, "expansions": None}
        results["gbfs"][identifier] = {"time": None, "expansions": None}
    else:
        results = {"astar": {identifier: {"time": None, "expansions": None}},
                   "gbfs": {identifier: {"time": None, "expansions": None}}}

        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))

    try:
        with time_limit(args.time_limit):
            results["astar"][identifier]["expansions"] = run_baseline_astar(task, heuristics[args.heuristic](task))
            results["astar"][identifier]["timed_out"] = False
        with time_limit(args.time_limit):
            results["gbfs"][identifier]["expansions"] = run_baseline_gbfs(task, heuristics[args.heuristic](task))
            results["gbfs"][identifier]["timed_out"] = False
        with time_limit(args.time_limit):
            results["astar"][identifier]["time"] = time_baseline(task, heuristics[args.heuristic](task), search_astar)
        with time_limit(args.time_limit):
            results["gbfs"][identifier]["time"] = time_baseline(task, heuristics[args.heuristic](task), search_gbfs)
    except TimeoutException as e:
        print("Timed out")
        results[identifier]["timed_out"] = True

    with open(result_path, "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    main(parser.parse_args())