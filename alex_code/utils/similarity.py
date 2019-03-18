import numpy as np


def cosine_similarity(x, y):
    numerator = np.dot(x, y)
    denominator = np.linalg.norm(x, 2) * np.linalg.norm(y, 2)

    return numerator / denominator
