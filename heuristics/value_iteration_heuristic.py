import numpy as np

from task import Operator, Task
from heuristics.heuristic_base import Heuristic

from alex_code.utils.similarity import cosine_similarity


class ValueIterationHeuristic(Heuristic):
    """Implements a heuristic based on a learned value function."""

    def __init__(self, obs_ph, values, sess, env):
        """Initializes a ValueIterationHeuristic.
        
        Args:
          obs_ph:
          values: A Tensor of shape [None, ] containing the estimated value
            of each state in the associated observation placeholder.
        """

        super(ValueIterationHeuristic, self).__init__()
        self.values = values
        self.obs_ph = obs_ph
        self.sess = sess
        self.env = env

    def __call__(self, node):
        """Returns the value of the given state."""
        state = node.state  # A frozenset.
        state_emb = self.env.E.state_to_emb(state)  # A list of int facts.
        state_emb = np.array([state_emb])  # [1, num facts True in state].
        value = self.sess.run(
            self.values, feed_dict={self.obs_ph: state_emb})[0]
        return -value
