from argparse import ArgumentParser
from dotenv import find_dotenv, load_dotenv

from settings import ROOT_DIR
import os

from alex_code.utils.load import find_best_model

parser = ArgumentParser()
parser.add_argument("--domain-file", default="logistics/43/domain.pddl", type=str)
parser.add_argument("--problem-file", default="logistics/43/problogistics-6-1.pddl", type=str)
parser.add_argument("--graph-path", default="logistics/43/problogistics-6-1.p", type=str)
parser.add_argument("--restrictions", default={"directed": "undirected"})

def main(args):
    load_dotenv(find_dotenv(), override=True)

    models_dir = os.environ.get("GNN_MODEL_DIR")
    models_dir = os.path.join(ROOT_DIR, models_dir, os.path.dirname(args.graph_path))

    best_stats, best_dir = find_best_model(models_dir, args.restrictions)

    print(best_stats)
    print(best_dir)

if __name__ == "__main__":
    main(parser.parse_args())