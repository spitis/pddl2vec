import numpy as np
import torch

def cosine_similarity(x, y):
    numerator = np.dot(x, y)
    denominator = np.linalg.norm(x, 2) * np.linalg.norm(y, 2)

    return numerator / denominator


def euclidean_distance(x, y):
    return torch.norm(x - y, p=2, dim=1)
