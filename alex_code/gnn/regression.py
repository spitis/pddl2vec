import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear

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
            30,
            num_stacks=3,
            num_layers=2,
            shared_weights=True,
            dropout=0.25)

        self.conv2 = ARMAConv(
            30,
            25,
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


class RegressionNN(torch.nn.Module):
    def __init__(self, activation, num_features):
        super(RegressionNN, self).__init__()

        self.fc1 = Linear(num_features, 75)
        self.fc2 = Linear(75, 50)
        self.fc3 = Linear(50, 25)

        self.activation = get_activation(activation)

    def forward(self, x, edge_index):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)

        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, activation, num_features):
        super(DeepGCN, self).__init__()

        self.conv1 = GCNConv(num_features, 100)
        self.conv2 = GCNConv(100, 75)
        self.conv3 = GCNConv(75, 50)
        self.fc1 = Linear(50, 40)

        self.activation = get_activation(activation)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x


class OverFit(torch.nn.Module):
    def __init__(self, activation, num_features):
        super(OverFit, self).__init__()

        self.conv1 = GCNConv(num_features, 200)
        self.conv2 = GCNConv(200, 175)
        self.conv3 = GCNConv(175, 125)
        self.conv4 = GCNConv(125, 75)
        self.conv5 = GCNConv(75, 50)

        self.activation = get_activation(activation)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        # x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.activation(x)

        x = self.conv4(x, edge_index)
        x = self.activation(x)

        x = self.conv5(x, edge_index)

        return x