import gym, gym.spaces as spaces
from gym.utils import seeding
from pyperplan import _parse, _ground
from embedding import NaiveEmb, IntegerEmb
from collections import defaultdict
import numpy as np

class PddlBasicEnv(gym.Env):
  """
  Wraps a pddl domain+instance in a basic Gym env.

  Note that because actions are deterministic / variable by state, we encode
  actions directly as the next state. An agent should get the actions for a state by
  calling "get_actions"
  """

  def __init__(self, domain, instance, embedding_fn=IntegerEmb):
    self.problem = _parse(domain_file=domain, problem_file=instance)
    self.task = _ground(self.problem)
    self.E = embedding_fn(self.task)

    self.action_space = spaces.Discrete(1000)
    if embedding_fn is NaiveEmb:
      self.observation_space = spaces.MultiBinary(len(self.task.facts))
    else:
      self.observation_space = spaces.MultiDiscrete([len(self.task.facts)] * len(self.task.initial_state))
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

  def get_actions_and_rewards(self):
    return zip(*[(self.E.state_to_emb(a), int(self.task.goal_reached(a))-1) for a in self._actions])

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

class PddlSimpleMultiGoalEnv(gym.GoalEnv):
  def __init__(self, domain, instance, embedding_fn=IntegerEmb):
    self.problem = _parse(domain_file=domain, problem_file=instance)
    self.task = _ground(self.problem)
    self.E = embedding_fn(self.task)
    
    mutexes = defaultdict(set)
    for f in self.task.facts:
      a = f.replace('(','').replace(')','')
      key_val = a.split(' ')
      mutexes[' '.join(key_val[:2])].add(key_val[2])

      
    special_obj = ''.join([i for i in list(self.task.goals)[0].split(' ')[1] if not i.isdigit()])
    
    self.mutexes = {k:v for k, v in mutexes.items() if special_obj in k}
    self._goal_set = [f for f in self.task.facts if special_obj in f]
    assert(len(self._goal_set) == sum([len(v) for v in self.mutexes.values()]))
    
    self.goal_len = sum([len(v) for v in self.mutexes.values()])
    
    self.basic_init = frozenset([f for f in self.task.initial_state if not special_obj in f])
    
    self.action_space = spaces.Discrete(1000)
    if embedding_fn is IntegerEmb:
      self.observation_space = spaces.Dict(dict(
        desired_goal=spaces.MultiDiscrete([len(self.task.facts)] * len(self.mutexes)),
        achieved_goal=spaces.MultiDiscrete([len(self.task.facts)] * len(self.mutexes)),
        observation=spaces.MultiDiscrete([len(self.task.facts)] * len(self.task.initial_state)),
      ))
    elif embedding_fn is NaiveEmb:
      self.observation_space = spaces.Dict(dict(
        desired_goal=spaces.MultiBinary(self.goal_len),
        achieved_goal=spaces.MultiBinary(self.goal_len),
        observation=spaces.MultiBinary(len(self.task.facts)),
      ))
    else:
      raise ValueError
    
    self.reward_range = (-1., 0.)

    self.seed()
    self._state = None
    self._goal = None
    self._actions = None
    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def get_actions(self):
    return [self.E.state_to_emb(a) for a in self._actions]

  def get_actions_and_rewards(self):
    return zip(*[(self.E.state_to_emb(a), float(self._goal <= a) - 1.) for a in self._actions])

  def compute_reward(self, achieved_goal_or_state, desired_goal, info=None):
    """Computes reward with embeddings"""
    success = frozenset(desired_goal) <= frozenset(achieved_goal_or_state)
    return float(success) - 1.

  def get_actions_for_emb(self, state_emb):
    actions = [
      next_state for _, next_state in self.task.get_successor_states(
        self.E.emb_to_state(state_emb)
      )
    ]
    return [self.E.state_to_emb(a) for a in actions]
  
  def _sample_goal(self):
    return frozenset(['({} {})'.format(k, np.random.choice(list(v))) for k, v in self.mutexes.items()])
  
  def reset(self):
    self._state = self.basic_init.union(self._sample_goal())
    self._goal = self._sample_goal()
    self._goal_emb = self.E.state_to_emb(self._goal)
    self._actions = [
        next_state
        for op, next_state in self.task.get_successor_states(self._state)
    ]
    return self._get_obs()
  
  def _get_obs(self):
    return {
      'observation': self.E.state_to_emb(self._state),
      'achieved_goal': self.E.state_to_emb(self._state.intersection(self._goal_set)),
      'desired_goal': self._goal_emb
    }

  def render(self, mode=None):
    raise NotImplementedError

  def close(self):
    pass

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
    if len(self._goal.intersection(self._state)) == len(self._goal):
      reward = 0

    return self._get_obs(), reward, reward + 1, None