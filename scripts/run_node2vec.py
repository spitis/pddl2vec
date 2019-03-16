from argparse import ArgumentParser
from dotenv import load_dotenv, find_dotenv
import os
import subprocess
import logging

from settings import ROOT_DIR

parser = ArgumentParser()
parser.add_argument("--graph-file", default="logistics/43/problogistics-6-1.edgelist", type=str)


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
    
    embedding_path = os.path.join(embedding_path, os.path.basename(args.graph_file).split(".")[0] + ".emb")
    
    command_path = os.environ.get("NODE2VEC_COMMAND")
    cmd = [os.path.join(ROOT_DIR, command_path), "-i:{}".format(graph_path), "-o:{}".format(embedding_path), "-dr", "-v"]
    
    exitcode = subprocess.call(cmd, stdout=subprocess.PIPE)

    if exitcode == 0:
        print('Node2Vec Complete')
    else:
        print('Node2Vec Failed')

    
if __name__ == "__main__":
    main(parser.parse_args())