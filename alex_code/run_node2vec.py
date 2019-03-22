from argparse import ArgumentParser
from dotenv import load_dotenv, find_dotenv
import os
import subprocess
import logging

from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--graph-file", default="logistics/43/problogistics-6-1.edgelist", type=str)
parser.add_argument("--d", default=128, type=int)
parser.add_argument("--l", default=80, type=int)
parser.add_argument("--r", default=10, type=int)
parser.add_argument("--k", default=10, type=int)
parser.add_argument("--e", default=1, type=int)
parser.add_argument("--p", default=1, type=float)
parser.add_argument("--q", default=1, type=float)
parser.add_argument("--directed", default="dr", type=str,
                   choices=["dr", "u"])


def main(args):
    load_dotenv(find_dotenv(), override=True)
    graph_dir = os.environ.get("GRAPHS_DIR")
    graph_dir = os.path.join(ROOT_DIR, graph_dir)
    
    embedding_dir = os.environ.get("EMBEDDINGS_DIR")
    embedding_dir = os.path.join(ROOT_DIR, embedding_dir)
                    
    graph_path = os.path.join(graph_dir, args.graph_file)
    embedding_path = os.path.join(embedding_dir, os.path.dirname(args.graph_file))
    
    if not os.path.exists(embedding_path):
        os.makedirs(embedding_path)
    
    problem_name = os.path.basename(args.graph_file).split(".")[0]
    embedding_file = os.environ.get("EMBEDDING_FILE")
    embedding_path = os.path.join(embedding_path,  embedding_file.format(problem_name=problem_name, d=args.d, l=args.l, r=args.r, k=args.k, e=args.e,
                                                                        p=args.p, q=args.q, directed=args.directed))
    
    command_path = os.environ.get("NODE2VEC_COMMAND")
    cmd = [os.path.join(ROOT_DIR, command_path), "-i:{}".format(graph_path), "-o:{}".format(embedding_path), "-d:{}".format(args.d),
           "l:{}".format(args.l), "-r:{}".format(args.r), "-k:{}".format(args.k), "-e:{}".format(args.e), "-p:{}".format(args.p), "-q:{}".format(args.q), "-v"]
    
    if args.directed == "dr":
        cmd.append("dr")
    
    exitcode = subprocess.call(cmd, stdout=subprocess.PIPE)

    if exitcode == 0:
        print('Node2Vec Complete')
    else:
        print('Node2Vec Failed')

    
if __name__ == "__main__":
    main(parser.parse_args())