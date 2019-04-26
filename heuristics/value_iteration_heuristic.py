import numpy as np

from task import Operator, Task
from heuristics.heuristic_base import Heuristic

from alex_code.utils.similarity import cosine_similarity


class ValueIterationHeuristic(Heuristic):
    """Implements a heuristic based on a learned value function."""

    def __init__(self, obs_ph, goal_ph, is_goal_agent, values, sess, env):
        """Initializes a ValueIterationHeuristic.

        Args:
          obs_ph:
          values: A Tensor of shape [None, ] containing the estimated value
            of each state in the associated observation placeholder.
        """

        super(ValueIterationHeuristic, self).__init__()
        self.values = values
        self.obs_ph = obs_ph
        self.goal_ph = goal_ph
        self.is_goal_agent = is_goal_agent
        self.sess = sess
        self.env = env

    def __call__(self, node):
        """Returns the value of the given state."""
        state = node.state  # A frozenset.
        state_emb = self.env.E.state_to_emb(state)  # A list of int facts.
        state_emb = np.array([state_emb])  # [1, num facts True in state].

        # Grab the goal from the ground task.
        goal = self.env.task.goals
        goal_emb = self.env.E.state_to_emb(goal)
        goal_emb = np.array([goal_emb])
        
        feed_dict_ = {self.obs_ph: state_emb}
        if self.is_goal_agent:
          feed_dict_[self.goal_ph] = goal_emb
        value = self.sess.run(self.values, feed_dict=feed_dict_)[0]
        return -value
