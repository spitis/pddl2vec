import os
import torch

from alex_code.utils.save import read_json


def find_weight_dict(models_dir, problem_name, args):
    dirs = os.listdir(models_dir)

    for d in dirs:
        if not problem_name in d:
            continue

        temp = os.path.join(models_dir, d)

        stats_path = os.path.join(temp, os.environ.get("GNN_STATS_FILE"))
        if not os.path.exists(stats_path):
            continue

        stats = read_json(stats_path)
        subset = dictionary_subset(args.__dict__, stats)

        if not subset:
            continue
        else:
            model_path = os.path.join(temp, os.environ.get("GNN_MODEL_FILE"))

            return model_path, stats_path, d

        return None


def find_best_model(models_dir, restrictions={}):
    best_stats = None
    best_loss = float("inf")
    dirs = os.listdir(models_dir)

    for d in dirs:
        temp = os.path.join(models_dir, d)

        stats_path = os.path.join(temp, os.environ.get("GNN_STATS_FILE"))

        if not os.path.exists(stats_path):
            continue

        stats = read_json(stats_path)
        subset = dictionary_subset(restrictions, stats)

        if subset and stats["best_loss"] < best_loss:
            best_loss = stats["best_loss"]
            best_stats = stats
            best_dir = d

    return best_stats, best_dir


def dictionary_subset(d1, d2):
    for k1, v1 in d1.items():
        if "file" in k1 or "path" in k1:
            continue

        if k1 not in d2.keys():
            return False

        if v1 != d2[k1]:
            return False

    return True