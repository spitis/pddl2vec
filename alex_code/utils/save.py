import _pickle as pickle
import json

def write_pickle(content, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(content, f)


def read_pickle(file_path):
    with open(file_path, "rb") as f:
        temp = pickle.load(f)

    return temp


def read_json(file_path):
    with open(file_path, "r") as f:
        temp = json.load(f)

    return temp