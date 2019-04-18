import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from alex_code.gnn.gnn_pair_dataset import get_pairs
from alex_code.gnn.activations import get_activation
from alex_code.utils.similarity import euclidean_distance

from torch_geometric.nn import ARMAConv



class RegressionGCN(torch.nn.Module):
    def __init__(self, activation, num_features):
        super(RegressionGCN, self).__init__()
        self.conv1 = GCNConv(num_features, 30)
        self.conv2 = GCNConv(30, 25)

        self.activation = get_activation(activation)

    def forward(self, x, edge_index):
        features = self.extract_features(x, edge_index)

        return features

    def extract_features(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class RegressionARMA(torch.nn.Module):
    def __init__(self, activation, num_features):
        super(RegressionARMA, self).__init__()

        self.conv1 = ARMAConv(
            num_features,
            16,
            num_stacks=3,
            num_layers=2,
            shared_weights=True,
            dropout=0.25)

        self.conv2 = ARMAConv(
            16,
            10,
            num_stacks=3,
            num_layers=2,
            shared_weights=True,
            dropout=0.25,
            act=None)

        self.activation = get_activation(activation)

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x