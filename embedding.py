import numpy as np
from abc import ABC, abstractmethod
from copy import deepcopy


class PddlEmb(ABC):

  def __init__(self, task):
    self.task = deepcopy(task)
    self._simple_test()

  @abstractmethod
  def state_to_emb(self, state):
    """Accepts a frozenset state and returns a state embedding"""
    pass

  @abstractmethod
  def emb_to_state(self, emb):
    """Accepts a state embedding and returns a frozenset state."""
    pass

  def _simple_test(self):
    s = self.emb_to_state(self.state_to_emb(self.task.initial_state))
    assert (s == self.task.initial_state), 'Auto-encoding initial state failed!'


class NaiveEmb(PddlEmb):
  """
  - Naive binary embedding for a state space with N fluents is a N-dimensional
    binary vector that has a 1 for each true fact
  - To get continuous embedding, just multiply the binary embedding by an
    NxM trainable embedding matrix
  - It is "naive" because we aren't using any shared feature representations for
    similar predicates/facts
  """

  def __init__(self, task):
    self.facts = tuple(task.facts)
    self.facts_arr = np.array(self.facts)
    self.dims = len(self.facts)
    self.fact_idxs = {f: i for i, f in enumerate(self.facts)}
    super().__init__(task)

  def state_to_emb(self, state):
    """Accepts frozenset state and returns binary embedding"""
    res = np.zeros([self.dims], dtype=np.bool)
    idxs = np.array([self.fact_idxs[fact] for fact in state])
    res[idxs] = True
    return res

  def state_to_emb_alt(self, state):
    """Accepts frozenset state and returns binary embedding
    Slower than primary method (~8 times slower on transport)."""
    return np.array([(fact in state) for fact in self.facts], dtype=np.bool)

  def emb_to_state(self, emb):
    """Accepts binary state embedding and returns frozenset state."""
    return frozenset(self.facts_arr[emb.astype(np.bool)])




class IntegerEmb(PddlEmb):
  """
  - Integer embedding is a list of integers, where each represents a true fluent
  - To get continuous embedding, need to use tf.gather or something
  - It is "naive" because we aren't using any shared feature representations for
    similar predicates/facts
  """

  def __init__(self, task):
    self.facts = tuple(task.facts)
    self.facts_arr = np.array(self.facts)
    self.dims = len(self.facts)
    self.fact_idxs = {f: i for i, f in enumerate(self.facts)}
    super().__init__(task)

  def state_to_emb(self, state):
    """Accepts frozenset state and returns binary embedding"""
    idxs = np.array(sorted([self.fact_idxs[fact] for fact in state]), dtype=np.int32)
    return idxs

  def emb_to_state(self, emb):
    """Accepts binary state embedding and returns frozenset state."""
    return frozenset(self.facts_arr[emb])


