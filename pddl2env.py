import gym, gym.spaces as spaces
from pyperplan import _parse, _ground
from embedding import NaiveEmb


class PddlBasicEnv(gym.Env):
  """
  Wraps a pddl domain+instance in a basic Gym env.

  Note that because actions are deterministic / variable by state, we encode
  actions directly as the next state. An agent should get the actions for a state by
  calling "get_actions"
  """

  def __init__(self, domain, instance, embedding_fn=NaiveEmb):
    self.problem = _parse(domain_file=domain, problem_file=instance)
    self.task = _ground(self.problem)
    self.E = embedding_fn(self.task)

    self.action_space = spaces.Discrete(1000)
    self.observation_space = spaces.MultiBinary(len(self.task.facts))
    self.reward_range = (-1., 0.)

    self._state = None
    self._actions = None
    self.reset()

  def step(self, action):
    next_state = self.E.emb_to_state(action)
    if next_state not in self._actions:
      raise ValueError('bad action!')
    self._state = next_state
    self._actions = [
        next_state
        for op, next_state in self.task.get_successor_states(self._state)
    ]

    reward = -1
    if self.task.goal_reached(self._state):
      reward = 0

    return self.E.state_to_emb(self._state), reward, reward + 1, None

  def get_actions(self):
    return [self.E.state_to_emb(a) for a in self._actions]

  def reset(self):
    self._state = self.task.initial_state
    self._actions = [
        next_state
        for op, next_state in self.task.get_successor_states(self._state)
    ]
    return self.E.state_to_emb(self._state)

  def render(self, mode=None):
    raise NotImplementedError

  def close(self):
    pass

  def seed(self, seed=None):
    pass
