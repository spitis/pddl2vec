from alex_code.gnn.regression import RegressionARMA, RegressionGCN


def create_model(name, num_features):
    if name == "arma":
        model = RegressionARMA(num_features)
    elif name == "gcn":
        model = RegressionGCN(num_features)
    else:
        raise Exception("Specified model: {} does not exist".format(name))

    return model