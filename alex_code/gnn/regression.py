import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from alex_code.gnn.gnn_pair_dataset import get_pairs
from alex_code.utils.similarity import euclidean_distance


class RegressionGCN(torch.nn.Module):
    def __init__(self, num_features):
        super(RegressionGCN, self).__init__()
        self.conv1 = GCNConv(num_features, 25)
        self.conv2 = GCNConv(25, 10)
        self.conv3 = GCNConv(10, 8)

    def forward(self, x, edge_index):
        features = self.extract_features(x, edge_index)

        return features

    def extract_features(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return x