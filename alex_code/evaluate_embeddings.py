from argparse import ArgumentParser
import networkx as nx
import numpy as np

from dotenv import find_dotenv, load_dotenv
from scipy.stats import spearmanr, pearsonr

from settings import ROOT_DIR
import os
import json

from alex_code.utils.similarity import cosine_similarity

parser = ArgumentParser()
parser.add_argument("--problem-path", default="logistics/43/problogistics-6-1.pddl", type=str)
parser.add_argument("--d", default=128, type=int)
parser.add_argument("--l", default=80, type=int)
parser.add_argument("--r", default=10, type=int)
parser.add_argument("--k", default=10, type=int)
parser.add_argument("--e", default=1, type=int)
parser.add_argument("--p", default=1, type=float)
parser.add_argument("--q", default=1, type=float)


def load_embeddings(embedding_path):
    ordering = []
    embeddings = []

    with open(embedding_path, "r") as f:
        content = f.readlines()

    for i in range(1, len(content)):
        split = content[i].split(" ")
        split[-1] = split[-1][:-2]

        ordering.append(split[0])
        embeddings.append(np.array(split[1:]))

    ordering = np.array(ordering).astype(int)
    embeddings = np.array(embeddings).astype(float)
    new_embeddings = np.ones(embeddings.shape)
    new_embeddings[ordering, :] = embeddings[np.arange(len(embeddings)),:]
    
    return new_embeddings


def load_graph(edges_path):
    # Create graph from edges
    G = nx.Graph()

    with open(edges_path, "r") as f:
        content = f.readlines()

    for i in range(0, len(content)):
        split = content[i].split(" ")
        G.add_edge(int(split[0]), int(split[1][:-1]))
        
    return G


def compute_distances(embeddings, G):
    dot_products = []
    distances = []
    cosine_similarities = []
    l2s = []

    for i in range(len(G.nodes)):
        for j in range(len(G.nodes)):
            if nx.has_path(G, i, j):
                dot_products.append(np.dot(embeddings[i], embeddings[j]))
                distances.append(nx.shortest_path_length(G, i, j))
                cosine_similarities.append(cosine_similarity(embeddings[i], embeddings[j]))
                l2s.append(np.linalg.norm(embeddings[i] - embeddings[j], 2))
    
    return np.array(dot_products), np.array(distances), np.array(cosine_similarities), np.array(l2s)


def main(args):
    load_dotenv(find_dotenv(), override=True)
    
    problem_name = os.path.basename(args.problem_path).split(".")[0]
    
    embedding_dir = os.environ.get("EMBEDDINGS_DIR")
    embedding_dir = os.path.join(ROOT_DIR, embedding_dir, os.path.dirname(args.problem_path))
    embedding_file = os.environ.get("EMBEDDING_FILE")
    
    embedding_path = os.path.join(embedding_dir, embedding_file.format(problem_name=problem_name, d=args.d,  l=args.l, r=args.r, k=args.k,
                                                                      e=args.e, p=args.p, q=args.q))
    
    edges_dir = os.environ.get("GRAPHS_DIR")
    edges_dir = os.path.join(ROOT_DIR, edges_dir, os.path.dirname(args.problem_path))
    edges_path = os.path.join(edges_dir, problem_name + ".edgelist")
    
    result_dir = os.environ.get("EMBEDDING_EVALUATION_DIR")
    result_dir = os.path.join(ROOT_DIR, result_dir, os.path.dirname(args.problem_path))
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    result_file = os.environ.get("EMBEDDING_EVAL_FILE")
    result_path = os.path.join(result_dir, result_file.format(problem_name=problem_name, d=args.d, l=args.l, r=args.r, k=args.k,
                                                             e=args.e, p=args.p, q=args.q))
    
    embeddings = load_embeddings(embedding_path)
    G = load_graph(edges_path)
    
    dot_products, distances, cosine_similarities, l2s= compute_distances(embeddings, G)
    
    spearman_cor = spearmanr(dot_products, distances)
    pearson_cor = pearsonr(dot_products, distances)
    
    spearman_cor_inv_dot = spearmanr(1.0 / dot_products, distances)
    pearson_cor_inv_dot = pearsonr(1.0 / dot_products, distances)
    
    spearman_cor_inv_dist = spearmanr(dot_products, 1.0 / distances)
    pearson_cor_inv_dist = pearsonr(dot_products, 1.0 / distances)

    spearman_cosine_sim_cor = spearmanr(cosine_similarities, distances)
    pearson_cosine_sim_cor = pearsonr(cosine_similarities, distances)

    spearman_l2_cor = spearmanr(l2s, distances)
    pearson_l2_cor = pearsonr(l2s, distances)
    
    results = {"spearman_cor": spearman_cor, "pearson_cor": pearson_cor,
               "spearman_cor_inv_dot": spearman_cor_inv_dot, "pearson_cor_inv_dot": pearson_cor_inv_dot,
              "spearman_cor_inv_dist": spearman_cor_inv_dist, "pearson_cor_inv_dist": pearson_cor_inv_dist,
               "spearman_cosine_sim_cor": spearman_cosine_sim_cor, "pearson_cosine_sim_cor": pearson_cosine_sim_cor,
               "spearman_l2_cor": spearman_l2_cor, "pearson_l2_cor": pearson_l2_cor}

    with open(result_path, "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

    
if __name__ == "__main__":
    main(parser.parse_args())