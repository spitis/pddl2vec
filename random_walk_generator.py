import random
import numpy as np


class RandomWalkGenerator(object):
    """A class that for sampling random trajectories from a given transition system."""
    
    def __init__(self, task, walk_length=10, reservoir_size=50):
        """Initializes a RandomWalkGenerator.
        
        Args:
            task: A pyperlan "ground problem".
            walk_length: The length of each generated walk.
            reservoir_size: The number of (randomly-encountered) states to store. 
                These are used as starting points for future random walks. 
        """
        self.task = task
        self.walk_length = walk_length
        self.reservoir_size = reservoir_size
        self.reservoir = set()  # Init empty reservoir.
        
    def add_to_reservoir(self, states):
        """Adds a state to the reservoir set and removes others if the storage limit is reached.
        
        Args:
            state: A list of frozensets.
        """
        msg = "The requested number of states exceed the reservoir's capacity."
        assert len(states) < self.reservoir_size, msg
        
        # Remove some states if adding the new ones would surpass the reservoir's capacity.
        num_states_to_remove = len(self.reservoir) + len(states) - self.reservoir_size
        while num_states_to_remove > 0:
            self.reservoir.pop()
            num_states_to_remove -= 1
        
        # Add the new states.
        for state in states:
            self.reservoir.add(state)
        
    def sample_walk(self):
        """Returns a random trajectory."""
        
        # Get a starting state from the reservoir or, if it's empty, use the initial state.
        current_state = self.task.initial_state
        if len(self.reservoir) > 0:
            current_state = self.reservoir.pop()
        
        states_to_save = []
        walk = [current_state]
        while len(walk) < self.walk_length:
            
            # Randomly choose whether to store the state to use as a start point later.
            add_to_reservoir = random.choice([True, False])
            if add_to_reservoir:
                states_to_save.append(current_state)
        
            # A list of (op, frozenset) tuples.
            successors = self.task.get_successor_states(current_state)

            # Randomly choose a successor state.
            chosen_successor_idx = random.randint(0, len(successors) - 1)
            current_state = successors[chosen_successor_idx][1]
            walk.append(current_state)
        
        self.add_to_reservoir(states_to_save)
        return walk