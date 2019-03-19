from collections import deque
import logging

from search import searchspace

import networkx as nx


def expand_state_space(planning_task, token_mapping, limit=1000000):
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
        

