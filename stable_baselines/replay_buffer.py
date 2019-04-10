import numpy as np, random
from collections import OrderedDict, deque

class RingBuffer(object):
  """This is a collections.deque in numpy, with pre-allocated memory"""

  def __init__(self, maxlen, shape, dtype='float32'):
    """
    A buffer object, when full restarts at the initial position

    :param maxlen: (int) the max number of numpy objects to store
    :param shape: (tuple) the shape of the numpy objects you want to store
    :param dtype: (str) the name of the type of the numpy object you want to store
    """
    self.maxlen = maxlen
    self.start = 0
    self.length = 0
    self.shape = shape
    self.data = np.zeros((maxlen, ) + shape).astype(dtype)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    if idx < 0 or idx >= self.length:
      raise KeyError()
    return self.data[(self.start + idx) % self.maxlen]

  def get_batch(self, idxs):
    """
    get the value at the indexes

    :param idxs: (int or numpy int) the indexes
    :return: (np.ndarray) the stored information in the buffer at the asked positions
    """
    return self.data[(self.start + idxs) % self.length]

  def append(self, var):
    """
    Append an object to the buffer

    :param var: (np.ndarray) the object you wish to add
    """
    if self.length < self.maxlen:
      # We have space, simply increase the length.
      self.length += 1
    elif self.length == self.maxlen:
      # No space, "remove" the first item.
      self.start = (self.start + 1) % self.maxlen
    else:
      # This should never happen.
      raise RuntimeError()

    self.data[(self.start + self.length - 1) % self.maxlen] = var

  def _append_batch_with_space(self, var):
    """
    Append a batch of objects to the buffer, *assuming* there is space.

    :param var: (np.ndarray) the batched objects you wish to add
    """
    len_batch = len(var)
    start_pos = (self.start + self.length) % self.maxlen

    self.data[start_pos : start_pos + len_batch] = var
    
    if self.length < self.maxlen:
      self.length += len_batch
      assert self.length <= self.maxlen, "this should never happen!"
    else:
      self.start = (self.start + len_batch) % self.maxlen
  
  def append_batch(self, var):
    """
    Append a batch of objects to the buffer.

    :param var: (np.ndarray) the batched objects you wish to add
    """
    len_batch = len(var)
    assert len_batch < self.maxlen, 'trying to add a batch that is too big!'
    start_pos = (self.start + self.length) % self.maxlen
    
    if start_pos + len_batch <= self.maxlen:
      # If there is space, add it
      self._append_batch_with_space(var)
    else:
      # No space, so break it into two batches for which there is space
      first_batch, second_batch = np.split(var, [self.maxlen - start_pos])
      self._append_batch_with_space(first_batch)
      # use append on second call in case len_batch > self.maxlen
      self._append_batch_with_space(second_batch)

class ReplayBuffer(object):
  def __init__(self, limit, item_shape):
    """
    The replay buffer object

    :param limit: (int) the max number of transitions to store
    :param item_shape: a list of tuples of (str) item name and (tuple) the shape for item
      Ex: [("observations0", env.observation_space.shape),\
          ("actions",env.action_space.shape),\
          ("rewards", (1,)),\
          ("observations1",env.observation_space.shape ),\
          ("terminals1", (1,))]
    """
    self.limit = limit

    self.items = OrderedDict()

    for name, shape in item_shape:
      self.items[name] = RingBuffer(limit, shape=shape)

  def sample(self, batch_size):
    """
    sample a random batch from the buffer

    :param batch_size: (int) the number of element to sample for the batch
    :return: (list) the sampled batch
    """
    if self.size==0:
      return []
    # Draw such that we always have a proceeding element.
    batch_idxs = np.random.randint(low=0, high=(self.size - 1), size=batch_size)

    transition = []
    for buf in self.items.values():
      item = buf.get_batch(batch_idxs)
      transition.append(item)

    return transition

  def add(self, *items):
    """
    Appends a single transition to the buffer

    :param items: a list of values for the transition to append to the replay buffer,
        in the item order that we initialized the ReplayBuffer with.
    """
    for buf, value in zip(self.items.values(), items):
      buf.append(value)

  def add_batch(self, *items):
    """
    Append a batch of transitions to the buffer.

    :param items: a list of batched transition values to append to the replay buffer,
        in the item order that we initialized the ReplayBuffer with.
    """
    if (items[0].shape) == 1 or len(items[0]) == 1:
      self.add(*items)
      return

    for buf, batched_values in zip(self.items.values(), items):
      buf.append_batch(batched_values)

  def __len__(self):
    return self.size

  @property
  def size(self):
    # Get the size of the RingBuffer on the first item type
    return len(next(iter(self.items.values())))

class EpisodicBuffer(object):
  def __init__(self, n_subbuffers, process_trajectory_fn, n_cpus=None):
    """
    A simple buffer for storing full length episodes (as a  list of lists).

    :param n_subbuffers: (int) the number of subbuffers to use
    """
    self._main_buffer = []
    self.n_subbuffers = n_subbuffers
    self._subbuffers = [[] for _ in range(n_subbuffers)]
    n_cpus = n_cpus or n_subbuffers
    
    self.fn = process_trajectory_fn

  def commit_subbuffer(self, i):
    """
    Adds the i-th subbuffer to the main_buffer, then clears it. 
    """
    self._main_buffer.append(self._subbuffers[i])
    self._subbuffers[i] = []

  def add_to_subbuffer(self, i, item):
    """
    Adds item to i-th subbuffer.
    """
    self._subbuffers[i].append(item)

  def __len__(self):
    return len(self._main_buffer)

  def process_trajectories(self):
    """
    Processes trajectories
    """
    return map(self.fn, self._main_buffer)

  def clear_main_buffer(self):
    self._main_buffer = []

  def clear_all(self):
    self._main_buffer = []
    self._subbuffers = [[] for _ in range(self.n_subbuffers)]

  def close(self):
    self.pool.close()

def her_future(trajectory, k, compute_reward, reward_mode, sample_her_with_replacement, process_successful_trajectories=True):
  """produces hindsight experiences where desired_goal is replaced with future achieved_goals
  if short circuit is true, cuts of the end of the trajectory where the achieved goal does not move"""
  # Determine if there are additional goal actions
  goal_with_action = len(trajectory[0]) == 8

  final_achieved_goal = trajectory[-1][4]
  if not process_successful_trajectories and np.allclose(final_achieved_goal, trajectory[-1][5]):
    return [] # don't add successful trajectories twice
  achieved_goals = np.array([transition[4] for transition in trajectory])
  if goal_with_action:
    achieved_goals_action = np.array([transition[6] for transition in trajectory])

  len_ag = len(achieved_goals)
  achieved_goals_range = np.array(range(len_ag))
  hindsight_experiences = []

  for i, transition in enumerate(trajectory):
    if goal_with_action:
      o1, action, _, o2, achieved_goal, _, achieved_goal_action, _ = transition
    else:
      o1, action, _, o2, achieved_goal, _ = transition

    if sample_her_with_replacement:
      sampled_goals_idx = np.random.choice(achieved_goals_range[i:], k, replace=True)
    else:
      sampled_goals_idx = np.random.choice(achieved_goals_range[i:], min(k, len_ag - i), replace=False)

    sampled_goals = achieved_goals[sampled_goals_idx]
    
    if goal_with_action:
      sampled_goals_action = achieved_goals_action[sampled_goals_idx]
      for g, g_act in zip(sampled_goals, sampled_goals_action):
        reward = compute_reward([achieved_goal, achieved_goal_action], [g, g_act], None)
        hindsight_experiences.append([o1, action, reward, o2, np.allclose(reward, reward_mode), g, g_act])

    else:
      for g in sampled_goals:
        reward = compute_reward(achieved_goal, g, None)
        hindsight_experiences.append([o1, action, reward, o2, np.allclose(reward, reward_mode), g])
    
  return hindsight_experiences

class HerFutureAchievedPastActual():
  def __init__(self, k, p, compute_reward, reward_mode, sample_her_with_replacement, past_goal_memory=100000):
    self.k = k # future
    self.p = p # past goals
    self.compute_reward = compute_reward
    self.goal_mem=deque(maxlen=past_goal_memory)
    self.reward_mode = reward_mode
    self.sample_her_with_replacement = sample_her_with_replacement
  
  def __call__(self, trajectory):
    actual_goal = trajectory[0][5]
    self.goal_mem.append(actual_goal)
    achieved_goals = np.array([transition[4] for transition in trajectory])
    len_ag = len(achieved_goals)
    achieved_goals_range = np.array(range(len_ag))
    hindsight_experiences = []
    for i, (o1, action, _, o2, achieved_goal, _) in enumerate(trajectory):
      if self.sample_her_with_replacement:
        sampled_goals_idx = np.random.choice(achieved_goals_range[i:], self.k, replace=True)
      if self.sample_her_with_replacement:
        sampled_goals_idx = np.random.choice(achieved_goals_range[i:], min(self.k, len_ag - i), replace=False)
      sampled_goals = list(achieved_goals[sampled_goals_idx])
      sampled_goals += random.choices(self.goal_mem, k=self.p)
      for g in sampled_goals:
        reward = self.compute_reward(achieved_goal, g, None)
        hindsight_experiences.append([o1, action, reward, o2, np.allclose(reward, self.reward_mode), g])
    return hindsight_experiences
