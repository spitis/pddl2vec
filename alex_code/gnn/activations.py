import torch.nn.functional as F


def get_activation(activation):
    if activation == "relu":
        return F.relu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "tanh":
        return F.tanh
    elif activation == "selu":
        return F.selu
    elif activation == "elu":
        return F.elu
    else:
        raise Exception("Activation: {} not implemented".format(activation))