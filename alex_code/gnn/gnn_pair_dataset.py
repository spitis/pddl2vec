import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data

from alex_code.utils.save import read_pickle

class GNNPairDatasetDisk(InMemoryDataset):
    """

    """

    def __init__(self, graph_path, node_mapping_path, goal_path, transform=None):
        super(GNNPairDatasetDisk, self).__init__(graph_path, transform, None, None)

        G = read_pickle(graph_path)
        node_mapping = read_pickle(node_mapping_path)
        goal = read_pickle(goal_path)

        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        data = Data(edge_index=edge_index)

        x = []

        for key, item in nx.get_node_attributes(G, "counts").items():
            x.append(item)

        x = torch.tensor(x).float()

        data.x = x

        self.node_mapping = node_mapping
        self.goal = goal
        self.data, self.slices = self.collate([data])
        self.G = G

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class GNNPairDatasetMemory(InMemoryDataset):
    """

    """

    def __init__(self, graph, transform=None):
        super(GNNPairDatasetMemory, self).__init__("", transform, None, None)

        G = graph

        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        data = Data(edge_index=edge_index)

        x = []

        for key, item in nx.get_node_attributes(G, "counts").items():
            x.append(item)

        x = torch.tensor(x).float()

        data.x = x

        self.data, self.slices = self.collate([data])
        self.G = G

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def get_pairs_long(dataset, dev, num_pairs=10000):
    num_nodes = dataset.data.num_nodes

    keys = torch.tensor(list(dataset.G.nodes))

    left_pair = torch.randint(0, num_nodes, (num_pairs,))
    left_pair = keys[left_pair]
    right_pair = torch.randint(0, num_nodes, (num_pairs,))
    right_pair = keys[right_pair]

    distances = []

    for i in range(num_pairs):
        distances.append(nx.shortest_path_length(dataset.G, left_pair[i].item(), right_pair[i].item()))

    left_pair = [dataset.node_mapping[key.item()] for key in left_pair]
    left_pair = torch.tensor(left_pair)

    right_pair = [dataset.node_mapping[key.item()] for key in right_pair]
    right_pair = torch.tensor(right_pair)

    distances = torch.tensor(distances).float().to(dev)

    return left_pair, right_pair, distances, dataset.data.edge_index


def get_pairs_directed(dataset, dev, num_pairs=10000):
    num_nodes = dataset.data.num_nodes

    keys = torch.tensor(list(dataset.G.nodes))
    start = torch.randint(0, 100, (1,)).item()

    distances = []
    keep = []

    for i in range(start, num_nodes):
        for j in range(i + 1, num_nodes):
            if len(keep) >= num_pairs:
                break

            if nx.has_path(dataset.G, keys[i].item(), keys[j].item()):
                distances.append(nx.shortest_path_length(dataset.G, keys[i].item(), keys[j].item()))
                keep.append([keys[i], keys[j]])

            if nx.has_path(dataset.G, keys[j].item(), keys[i].item()):
                distances.append(nx.shortest_path_length(dataset.G, keys[j].item(), keys[i].item()))
                keep.append([keys[j], keys[i]])

        if len(keep) >= num_pairs:
            break

    keep = torch.tensor(keep)

    left_pair = [dataset.node_mapping[keep[i, 0].item()] for i in range(len(keep))]
    left_pair = torch.tensor(left_pair)

    right_pair = [dataset.node_mapping[keep[i, 1].item()] for i in range(len(keep))]
    right_pair = torch.tensor(right_pair)

    distances = torch.tensor(distances).float().to(dev)

    return left_pair, right_pair, distances, dataset.data.edge_index


# def get_pairs_short(dataset, dev, num_pairs=10000):
#     keys = torch.tensor(list(dataset.G.nodes))
#
#     distances = []
#     keep = []
#     sources = list(torch.randint(0, len(keys) // 10, (num_pairs // 10,)).numpy())
#
#
#     for d in range(10):
#         for source in sources:
#             children = nx.bfs_successors(dataset.G, keys[source].item(), d + 1)
#             children = list(children)[-1][1]
#
#             for child in children:
#                 distances.append(nx.shortest_path_length(dataset.G, keys[source].item(), child))
#                 keep.append([keys[source].item(), child])
#
#                 if len(keep) >= num_pairs:
#                     break
#
#             if len(keep) >= num_pairs:
#                 break
#
#         if len(keep) >= num_pairs:
#             break
#
#     keep = torch.tensor(keep)
#
#     left_pair = [dataset.node_mapping[keep[i, 0].item()] for i in range(len(keep))]
#     left_pair = torch.tensor(left_pair)
#
#     right_pair = [dataset.node_mapping[keep[i, 1].item()] for i in range(len(keep))]
#     right_pair = torch.tensor(right_pair)
#
#     distances = torch.tensor(distances).float().to(dev)
#
#     return left_pair, right_pair, distances, dataset.data.edge_index
#


def get_pairs_short(dataset, dev, num_pairs=10000):
    keys = torch.tensor(list(dataset.G.nodes))

    distances = []
    keep = {i: [] for i in range(10)}
    sources = list(torch.randint(0, len(keys) // 10, (num_pairs // 10,)).numpy())

    for d in range(10):
        for source in sources:
            children = nx.bfs_successors(dataset.G, keys[source].item(), d + 1)
            children = list(children)[-1][1]

            for child in children:
                distances.append(nx.shortest_path_length(dataset.G, keys[source].item(), child))
                keep[d].append([keys[source].item(), child])

                if len(keep[d]) > num_pairs // 10:
                    break

            if len(keep[d]) >= num_pairs // 10:
                break

    temp = []
    for key, value in keep.items():
        temp = temp + value

    keep = torch.tensor(temp)

    left_pair = [dataset.node_mapping[keep[i, 0].item()] for i in range(len(keep))]
    left_pair = torch.tensor(left_pair)

    right_pair = [dataset.node_mapping[keep[i, 1].item()] for i in range(len(keep))]
    right_pair = torch.tensor(right_pair)

    distances = torch.tensor(distances).float().to(dev)

    return left_pair, right_pair, distances, dataset.data.edge_index

