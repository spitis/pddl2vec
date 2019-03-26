from collections import deque
import logging

from search import searchspace

import networkx as nx


def expand_state_space_node2vec(planning_task, token_mapping, limit=1000000):
    '''
    Searches for a plan on the given task using breadth first search and
    duplicate detection.

    @param planning_task: The planning task to solve.
    @return: The solution as a list of operators or None if the task is
    unsolvable.
    '''
    # counts the number of loops (only for printing)
    G = nx.Graph()
    iteration = 0
    # fifo-queue storing the nodes which are next to explore
    queue = deque()
    queue.append(searchspace.make_root_node(planning_task.initial_state))
    # set storing the explored nodes, used for duplicate detection
    closed = {planning_task.initial_state}
    expansions = 0

    while queue:
        if expansions >= limit:
            return G, node
        iteration += 1
        logging.debug("breadth_first_search: Iteration %d, #unexplored=%d"
                      % (iteration, len(queue)))
        # get the next node to explore
        node = queue.popleft()
        # exploring the node or if it is a goal node extracting the plan
        if planning_task.goal_reached(node.state):
            logging.info("Goal reached. Start extraction of solution.")
            logging.info("%d Nodes expanded" % iteration)
            return G, node
        for operator, successor_state in planning_task.get_successor_states(
                                                                   node.state):
            # duplicate detection
            if successor_state not in closed:
                new_node = searchspace.make_child_node(node, operator,
                                                         successor_state)
                queue.append(new_node)
                G.add_edge(hash_state(node.state, token_mapping), hash_state(new_node.state, token_mapping))
                 # remember the successor state
                # print("node.g: {} | new_node.g: {}".format(node.g, new_node.g))
                closed.add(successor_state)
                expansions += 1
    logging.info("No operators left. Task unsolvable.")
    logging.info("%d Nodes expanded" % iteration)
    return G, None


def hash_state(state, token_mapping):
    hash = 1
    
    if type(state) == tuple:
        temp = state[0].name[1:-1].split(" ")[0]
        hash *= token_mapping[temp]
        
        for fact in state[1]:
            temp_tokens = fact[1:-1]
            temp_tokens = temp_tokens.split(" ")

            for temp_token in temp_tokens:
                hash *= token_mapping[temp_token]        
        
    else:    
        for fact in state:
            temp_tokens = fact[1:-1]
            temp_tokens = temp_tokens.split(" ")

            for temp_token in temp_tokens:
                hash *= token_mapping[temp_token]
            
    return hash


def gen_primes():
    """ Generate an infinite sequence of prime numbers.
    """
    # Maps composites to primes witnessing their compositeness.
    # This is memory efficient, as the sieve is not "run forward"
    # indefinitely, but only as long as required by the current
    # number being tested.
    #
    D = {}
    
    # The running integer that's checked for primeness
    q = 2
    
    while True:
        if q not in D:
            # q is a new prime.
            # Yield it and mark its first multiple that isn't
            # already marked in previous iterations
            # 
            yield q
            D[q * q] = [q]
        else:
            # q is composite. D[q] is the list of primes that
            # divide it. Since we've reached q, we no longer
            # need it in the map, but we'll mark the next 
            # multiples of its witnesses to prepare for larger
            # numbers
            # 
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]
        
        q += 1
        

def expand_state_space_gnn(problem, planning_task, token_mapping, limit=1000000):
    G = nx.Graph()
    counts = {}
    iteration = 0
    # fifo-queue storing the nodes which are next to explore
    queue = deque()
    queue.append(searchspace.make_root_node(planning_task.initial_state))
    # set storing the explored nodes, used for duplicate detection
    closed = {planning_task.initial_state}
    expansions = 0

    while queue:
        if expansions >= limit:
            return G, node
        iteration += 1
        logging.debug("breadth_first_search: Iteration %d, #unexplored=%d"
                      % (iteration, len(queue)))
        # get the next node to explore
        node = queue.popleft()
        # exploring the node or if it is a goal node extracting the plan
        if planning_task.goal_reached(node.state):
            logging.info("Goal reached. Start extraction of solution.")
            logging.info("%d Nodes expanded" % iteration)
            return G, node, counts

        node_hash = hash_state(node.state, token_mapping)
        # Store count-based features for node
        counts[node_hash] = get_counts(problem, planning_task, node.state)

        for operator, successor_state in planning_task.get_successor_states(
                                                                   node.state):
            # duplicate detection
            if successor_state not in closed:
                new_node = searchspace.make_child_node(node, operator,
                                                         successor_state)
                queue.append(new_node)
                new_node_hash = hash_state(new_node.state, token_mapping)
                # Stroe count-based features for new state
                counts[new_node_hash] = get_counts(problem, planning_task, new_node.state)

                G.add_edge(node_hash, new_node_hash)
                 # remember the successor state
                closed.add(successor_state)
                expansions += 1
    logging.info("No operators left. Task unsolvable.")
    logging.info("%d Nodes expanded" % iteration)
    return G, None, counts


def get_predicate_counts(problem, task, state):
    state_grouped_counts = {pred: 0 for pred in sorted(problem.domain.predicates.keys())}
    common_grouped_counts = {pred: 0 for pred in sorted(problem.domain.predicates.keys())}

    state_grouped_list = []
    common_grouped_list = []

    for fact in list(state):
        parsed = fact[1:-1]
        parsed = parsed.split(" ")

        state_grouped_counts[parsed[0]] += 1

        if fact in task.goals:
            common_grouped_counts += 1

    for key in state_grouped_counts.keys():
        state_grouped_list.append(state_grouped_counts[key])
        common_grouped_list.append(common_grouped_counts[key])

    return state_grouped_list + common_grouped_list


def get_action_counts(problem, task, state):
    binary_counts = {action: 0 for action in sorted(problem.domain.actions.keys())}
    relevant_counts = {action: 0 for action in sorted(problem.domain.actions.keys())}

    for op in task.operators:
        parsed = op.name[1: -1]
        parsed = parsed.split(" ")[0]

        if len(op.add_effects.intersection(task.goals)) > 0 and op.applicable(state):
            binary_counts[parsed] = 1

        if op.applicable(state):
            relevant_counts[parsed] += len(op.add_effects.intersection(task.goals))

    binary_list = []
    relevant_list = []

    for key in binary_counts.keys():
        binary_list.append(binary_counts[key])
        relevant_list.append(relevant_counts[key])

    return binary_list + relevant_list


def get_counts(problem, task, state):
    return get_predicate_counts(problem, task, state) + get_action_counts(problem, task, state)

