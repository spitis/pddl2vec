from argparse import ArgumentParser
from dotenv import load_dotenv, find_dotenv
import copy
import json
import os
import subprocess
import logging

from settings import ROOT_DIR
from alex_code.gnn.gnn_pair_dataset import get_pairs

from alex_code.gnn.gnn_pair_dataset import GNNPairDatasetDisk
from alex_code.gnn.model_loading import create_model
from alex_code.gnn.regression import RegressionGCN
from alex_code.utils.similarity import euclidean_distance
from alex_code.utils.save import get_time


import torch
import torch.nn.functional as F

import torch_geometric.transforms as T

parser = ArgumentParser()
parser.add_argument("--graph-path", default="logistics/43/problogistics-6-1.p", type=str)
parser.add_argument("--epochs", default=200, dest="epochs", type=int)
parser.add_argument("--batch-size", default=1000, dest="batch_size", type=int)
parser.add_argument("--normalization", default="none", dest="normalization", choices=["none", "normalize"])
parser.add_argument("--seed", default=219, dest="seed")
parser.add_argument("--lr", default=0.01, dest="lr", type=float)
parser.add_argument("--model", default="gcn", dest="model", type=str, choices=["arma", "gcn"])
parser.add_argument("--directed", default="undirected", type=str, choices=["directed", "undirected"])


def train(dataset, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(args.model, dataset.data.num_features).to(device)
    dataset.data = dataset.data.to(device)

    if args.normalization != "none":
        dataset.data = T.NormalizeFeatures()(dataset.data)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)

    model.train()

    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        left, right, distance, edge_index = get_pairs(dataset, device, num_pairs=args.batch_size)
        optimizer.zero_grad()

        out = model(dataset.data.x, edge_index)
        left_features = out[left]
        right_features = out[right]
        euclidean = euclidean_distance(left_features, right_features)

        loss = F.mse_loss(euclidean, distance)
        loss.backward()
        optimizer.step()

        print("Loss: {}".format(loss))

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    left, right, distance, edge_index = get_pairs(dataset, device)
    optimizer.zero_grad()

    out = model(dataset.data.x, edge_index)
    left_features = out[left]
    right_features = out[right]
    euclidean = euclidean_distance(left_features, right_features)
    print("Pred: {}".format(euclidean))
    print("Actual: {}".format(distance))

    return model


def main(args):
    load_dotenv(find_dotenv(), override=True)

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    graph_dir = os.environ.get("GNN_GRAPH_DIR")
    graph_dir = os.path.join(ROOT_DIR, graph_dir, os.path.dirname(args.graph_path))

    problem_name = os.path.basename(args.graph_path).split(".")[0]

    graph_file = os.environ.get("GNN_GRAPH_FILE")
    graph_path = os.path.join(graph_dir, graph_file.format(problem_name=problem_name,
                                                           directed=args.directed))

    goal_file = os.environ.get("GNN_GOAL_FILE")
    goal_path = os.path.join(graph_dir, goal_file.format(problem_name=problem_name,
                                                         directed=args.directed))

    node_mapping_file = os.environ.get("GNN_NODE_MAPPING_FILE")
    node_mapping_path = os.path.join(graph_dir, node_mapping_file.format(problem_name=problem_name,
                                                                         directed=args.directed))

    models_dir = os.environ.get("GNN_MODEL_DIR")
    models_dir = os.path.join(ROOT_DIR, models_dir, os.path.dirname(args.graph_path))

    model_folder = os.environ.get("GNN_MODEL_SUBDIR")
    model_folder = os.path.join(models_dir, model_folder.format(problem_name=problem_name,
                                                                timestamp=get_time()))

    model_file = os.environ.get("GNN_MODEL_FILE")
    model_path = os.path.join(model_folder, model_file)

    gnn_pair_dataset = GNNPairDatasetDisk(graph_path, node_mapping_path, goal_path)

    model = train(gnn_pair_dataset, args)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    stats_file = os.environ.get("GNN_STATS_FILE")
    stats_path = os.path.join(model_folder, stats_file)

    d = vars(args)
    d["num_features"] = gnn_pair_dataset.data.num_features

    with open(stats_path, "w") as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)

    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main(parser.parse_args())