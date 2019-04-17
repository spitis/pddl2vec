from torch_geometric.transforms import NormalizeFeatures

from alex_code.gnn.transforms import NormalizeSamples

def apply_normalization(dataset, normalization):
    if normalization == "none":
        data = dataset.data
    elif normalization == "features" or normalization == "normalize":
        data = NormalizeFeatures()(dataset.data)
    elif normalization == "samples":
        data = NormalizeSamples()(dataset.data)
    else:
        raise Exception("Normalization: {} not implemented".format(normalization))

    return data