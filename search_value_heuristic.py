import os
import json
import timeit
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from pyperplan import search, get_heuristics, _get_heuristic_name

from pddl2env import PddlBasicEnv
from value_iteration_sp import SimpleValueIteration
from heuristics.value_iteration_heuristic import ValueIterationHeuristic

parser = ArgumentParser()
parser.add_argument(
    "--domain-file",
    default="pddl_files/modded_transport/domain.pddl",
    type=str)
parser.add_argument(
    "--problem-file",
    default="pddl_files/modded_transport/ptest.pddl",
    type=str)
parser.add_argument(
    "--restore-dir",
    default="/scratch/gobi1/eleni/csc2542/value_iteration_1",
    type=str)


def wrapper(func, *args, **kwargs):
    """Returns a callable that calls the given func with the given arguments."""

    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def perform_search(task, heuristic):
    """Runs A-star search in the given task using the given heuristic."""
    do_search = wrapper(search.astar_search, task, heuristic)
    solution, expansions = do_search()
    time_taken = timeit.timeit(do_search, number=1)

    if solution is None:
        raise Exception("Solution does not exist for this problem")

    return {
        "time": time_taken,
        "expansions": expansions,
        "solution length": len(solution)
    }


def evaluate_vi_model(vi_model):
    """Performs search using the given value-iteration model."""
    # Make sure the current state is the initial state.
    vi_model.env.reset()

    # The heuristic estimate of each given state based on the model's values.
    heuristic = ValueIterationHeuristic(vi_model._obs_ph, vi_model.values,
                                        vi_model.sess, vi_model.env)
    return perform_search(vi_model.env.task, heuristic)


def evaluate_standard(env, heuristic):
    """Performs search using the given standard heuristic."""
    # Make sure the current state is the initial state.
    env.reset()
    return perform_search(env.task, heuristic)


def main(args):
    problem_name = os.path.basename(args.problem_file).split(".")[0]

    # Get an environment for the specified domain and problem.
    env = PddlBasicEnv(domain=args.domain_file, instance=args.problem_file)

    # Load a pre-trained Value-Iteration model.
    vi_model = SimpleValueIteration(
        env=env, ckpt_dir=args.restore_dir, restore_dir=args.restore_dir)

    # Get a Value-Iteration model with random weights.
    vi_model_random = SimpleValueIteration(env=env)

    results = {}
    results["vi_heuristic"] = evaluate_vi_model(vi_model)
    results["vi_heuristic_random_weights"] = evaluate_vi_model(vi_model_random)

    # Also try some standard heuristics on the same problem.
    heuristics = {_get_heuristic_name(h): h for h in get_heuristics()}
    for h_name, h in heuristics.items():
        if h_name in ['valueiteration', 'node2vec']:
          continue
        results[h_name] = evaluate_standard(env, h(env.task))

    # Print the results of all heuristics.
    for model, model_results in results.items():
        print("model {}: {}".format(model, model_results))


if __name__ == "__main__":
    main(parser.parse_args())
