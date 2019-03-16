# Learning Planning Heuristics

This project investigates using embeddings, GNN, and RL based approaches to learn heuristics for planning problems.

## Directory Structure

Here is layout of the project. Some of the directories are omitted to avoid clutter.

```
project
|   baselines
|   embeddings
|   graphs
|   heuristics
|   pddl
|   pddl_files
|   planning_api_tools
|   pyperplan_docs
|   scripts
|   search
|   snap (your responsibility to download)
│   README.md
```

* `baselines` is where the baseline search results are stored in json files
* `embeddings` is where the node2vec embeddings generated for graphs fom the `graphs` dir are stored
* `graphs` is where the expanded state space for pddl problems is stored in an edgelist format that is expected by node2vec
* `heuristics` contains pyperplan heuristic function definitions
* `pddl` contains pyperplan parsing utilities
* `pddl_files` is where  problem instances from ICAPS competitions for various domains are stored
* `planning_api_tools` contains the planning.domains api files
* `scripts` is where python scripts for things like evaluating embedding quality and running node2vec are stored
* `search` contains pyperplan search functions

It is important to note that the library that actually has the node2vec utility (https://github.com/snap-stanford/snap) is not included in our repo since it is too large. Download this and name the directory `snap` if you'd like to run node2vec.

## Notebook Descriptions

* `api_tutorial.ipynb` shows how to use the planning.domains api to download problem  and domain files
* `create_graph_from_pddl.ipynb` takes a domain and problem file, performs a BFS on it to expand all nodes until a solution is found, and then stores that graph in an edgelist format that is expected by node2vec
* `evaluate_embedding.ipynb` reads in an edgelist file, and an embeddings file, and computes the spearman correlation between the dot product  vs actual distance for all possible node pairs
* `get_problem_files.ipynb` downloads all relevant problem and domain files for a specific domain into `pddl_files` using the following path convention `pddl_files/<domain name>/<competition number>/`
* `pyperplan --- basic binary and continuous embeddings.ipynb` shows how to create basic (non-node2vec) embeddings for pyperplan state representations
* `pyperplan.ipynb` shows how to load a pddl problem using pyperplan, and how to search for a plan using a specified heuristic, and then validate that plan

## Running Scripts

The scripts in `scripts` asssume that `project` is part of `PTYHONPATH`, so simpy run `source add_python_path.sh` from the main directory to ensure that the scripts have access to pyperplan.