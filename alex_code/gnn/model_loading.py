from alex_code.gnn.regression import RegressionARMA, RegressionGCN, RegressionNN, DeepGCN


def create_model(name, activation, num_features):
    if name == "arma":
        model = RegressionARMA(activation, num_features)
    elif name == "gcn":
        model = RegressionGCN(activation, num_features)
    elif name == "nn":
        model = RegressionNN(activation, num_features)
    elif name == "deepgcn":
        model = DeepGCN(activation, num_features)
    else:
        raise Exception("Specified model: {} does not exist".format(name))

    return model