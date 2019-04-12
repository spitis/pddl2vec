#
# This file is part of pyperplan.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import networkx as nx
import numpy as np
import torch

from task import Operator, Task
from heuristics.heuristic_base import Heuristic

from alex_code.utils.graph import hash_state, get_neighbours_forward, get_neighbours_regression, get_counts
from alex_code.utils.similarity import cosine_similarity
from alex_code.gnn.gnn_pair_dataset import GNNPairDatasetMemory

from search import searchspace

class GNNHeuristic(Heuristic):
    """
    Implements a simple blind heuristic for convenience.
    It returns 0 if the goal was reached and 1 otherwise.
    """

    def __init__(self, problem, task, gnn, device="gpu"):
        super(GNNHeuristic, self).__init__()
        self.goals = task.goals
        self.gnn = gnn
        self.task = task
        self.problem = problem
        self.device = device

        self.goal_embedding = self.generate_goal_embedding()

    def __call__(self, node, goal=None):
        node_embedding = self.generate_embedding(node)

        return torch.norm(self.goal_embedding - node_embedding, p=2)

    def generate_embedding(self, node):
        G = nx.Graph()
        neighbours = get_neighbours_forward(self.task, node)

        counts = {}
        counts[hash_state(node.state)] = get_counts(self.problem, self.task, node.state)

        for neighbour in neighbours:
            G.add_edge(hash_state(node.state), hash_state(neighbour.state))

            counts[hash_state(neighbour.state)] = get_counts(self.problem, self.task, neighbour.state)

        nx.set_node_attributes(G, counts, "counts")

        node_mapping = {n: i for i, n in enumerate(list(G.nodes))}

        final_G = nx.Graph()

        for edge in list(G.edges):
            final_G.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])

        final_counts = {}

        for key, value in counts.items():
            final_counts[node_mapping[key]] = value

        nx.set_node_attributes(final_G, final_counts, "counts")
        G = final_G

        pair_dataset = GNNPairDatasetMemory(G)
        pair_dataset.data = pair_dataset.data.to(self.device)
        embedding = self.gnn(pair_dataset.data.x, pair_dataset.data.edge_index)

        return embedding[node_mapping[hash_state(node.state)]]

    def generate_goal_embedding(self):
        G = nx.Graph()
        root = searchspace.make_root_node(self.task.goals)
        neighbours = get_neighbours_regression(self.task, root)

        counts = {}
        counts[hash_state(root.state)] = get_counts(self.problem, self.task, root.state)

        for neighbour in neighbours:
            G.add_edge(hash_state(root.state), hash_state(neighbour))

            counts[hash_state(neighbour)] = get_counts(self.problem, self.task, neighbour)

        for neighbour in neighbours:
            temp = searchspace.make_root_node(neighbour)

            second_level = get_neighbours_regression(self.task, temp)

            for sec in second_level:
                G.add_edge(hash_state(neighbour), hash_state(sec))

                counts[hash_state(sec)] = get_counts(self.problem, self.task, sec)

        nx.set_node_attributes(G, counts, "counts")

        node_mapping = {n: i for i, n in enumerate(list(G.nodes))}

        final_G = nx.Graph()

        for edge in list(G.edges):
            final_G.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])

        final_counts = {}

        for key, value in counts.items():
            final_counts[node_mapping[key]] = value

        nx.set_node_attributes(final_G, final_counts, "counts")
        G = final_G

        pair_dataset = GNNPairDatasetMemory(G)
        pair_dataset.data = pair_dataset.data.to(self.device)
        goal_embedding = self.gnn(pair_dataset.data.x, pair_dataset.data.edge_index)

        return goal_embedding[node_mapping[hash_state(root.state)]]

