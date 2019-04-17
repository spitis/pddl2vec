import torch


class NormalizeSamples(object):
    r"""Makes each feature have 0 mean and 1 std"""

    def __init__(self):
        pass

    def __call__(self, data):
        mean = torch.mean(data.x, 0)
        std = torch.std(data.x, 0)
        std[std == 0.0] = 1.0

        data.x = (data.x - mean) / std

        return data