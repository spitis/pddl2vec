#
# This file is part of pyperplan.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import numpy as np

from task import Operator, Task
from heuristics.heuristic_base import Heuristic

from alex_code.utils.graph import hash_state
from alex_code.utils.similarity import cosine_similarity


class Node2VecHeuristic(Heuristic):
    """
    Implements a simple blind heuristic for convenience.
    It returns 0 if the goal was reached and 1 otherwise.
    """
    def __init__(self, task, embeddings, node_mapping, goal_idx):
        super(Node2VecHeuristic, self).__init__()
        self.goals = task.goals
        self.embeddings = embeddings
        self.node_mapping = node_mapping
        self.goal_idx = goal_idx

    def __call__(self, node):
        node_idx = hash_state(node.state)
        
        if node_idx not in self.node_mapping.keys():
            return 0.0
        
        node_idx = self.node_mapping[node_idx]

        # return np.linalg.norm(self.embeddings[node_idx] - self.embeddings[self.goal_idx])
        return - 15 * cosine_similarity(self.embeddings[node_idx], self.embeddings[self.goal_idx])
        # return -np.dot(self.embeddings[node_idx], self.embeddings[self.goal_idx])
        # return 0.0