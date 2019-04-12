from argparse import ArgumentParser
from dotenv import load_dotenv, find_dotenv
import json
import os
import subprocess
import logging

from settings import ROOT_DIR
from alex_code.gnn.gnn_pair_dataset import get_pairs

from alex_code.gnn.gnn_pair_dataset import GNNPairDatasetDisk
from alex_code.gnn.regression import RegressionGCN
from alex_code.utils.similarity import euclidean_distance

import torch
import torch.nn.functional as F

parser = ArgumentParser()
parser.add_argument("--graph-path", default="logistics/43/problogistics-6-1.p", type=str)



def train(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RegressionGCN(dataset.data.num_features).to(device)
    dataset.data = dataset.data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-5)

    model.train()

    for epoch in range(500):
        left, right, distance, edge_index = get_pairs(dataset, device, num_pairs=1000)
        optimizer.zero_grad()

        out = model(dataset.data.x, edge_index)
        left_features = out[left]
        right_features = out[right]
        euclidean = euclidean_distance(left_features, right_features)

        loss = F.mse_loss(euclidean, distance)
        loss.backward()
        optimizer.step()

        print("Loss: {}".format(loss))

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
    graph_dir = os.environ.get("GNN_GRAPH_DIR")
    graph_dir = os.path.join(ROOT_DIR, graph_dir, os.path.dirname(args.graph_path))

    problem_name = os.path.basename(args.graph_path).split(".")[0]

    graph_file = os.environ.get("GNN_GRAPH_FILE")
    graph_path = os.path.join(graph_dir, graph_file.format(problem_name=problem_name))

    goal_file = os.environ.get("GOAL_FILE")
    goal_path = os.path.join(graph_dir, goal_file.format(problem_name=problem_name))

    node_mapping_file = os.environ.get("NODE_MAPPING_FILE")
    node_mapping_path = os.path.join(graph_dir, node_mapping_file.format(problem_name=problem_name))

    model_dir = os.environ.get("GNN_MODEL_DIR")
    model_dir = os.path.join(ROOT_DIR, model_dir, os.path.dirname(args.graph_path))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_file = os.environ.get("GNN_MODEL_FILE")
    model_path = os.path.join(model_dir, model_file.format(problem_name=problem_name))

    gnn_pair_dataset = GNNPairDatasetDisk(graph_path, node_mapping_path, goal_path)

    model = train(gnn_pair_dataset)

    stats_file = os.environ.get("GNN_STATS_FILE")
    stats_path = os.path.join(model_dir, stats_file.format(problem_name=problem_name))

    stats = {"num_features": gnn_pair_dataset.data.num_features}

    with open(stats_path, "w") as fp:
        json.dump(stats, fp, indent=4, sort_keys=True)

    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main(parser.parse_args())