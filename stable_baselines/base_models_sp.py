from abc import ABC, abstractmethod
import os, glob, numpy as np, gym, tensorflow as tf
import stable_baselines.logger as logger
import stable_baselines.tf_util as tf_util

class BaseRLModel(ABC):
  """
    The base RL model
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    """

  def __init__(self, env, verbose=0, *, _init_setup_model=None):
    self.env = env
    self.verbose = verbose
    self.observation_space = None
    self.action_space = None
    self.goal_space = None
    self.n_envs = None
    self._vectorize_action = False
    self.model = None
    self.sess = None
    self.initial_state = None
    self.step = lambda: None
    self.params = None
    self.writer = None

    # For the SimpleRLModel subclass (yes it should be there, but will eventually be merged anyways)
    self.exploration = None
    self.task_step = None
    self.global_step = None
    self.graph = None
    self.tensorboard_log = None
    self.eval_env = None
    self.eval_every = 10

    if env is not None:
      assert not isinstance(env, str)

      # Check if the environment is a goal-oriented type based on their observation space
      # Goal oriented Gym space have observation_space as dict
      if type(env.observation_space) == gym.spaces.dict_space.Dict:
        self.observation_space = env.observation_space.spaces["observation"]
        # Assume that desired and achieved goal have the same space
        self.goal_space = env.observation_space.spaces["desired_goal"]
      else:
        self.observation_space = env.observation_space

      self.observation_space = env.observation_space

      self.action_space = env.action_space
      self.n_envs = 1

  @abstractmethod
  def _setup_model(self):
    """
        Create all the functions and tensorflow graphs necessary to train the model
        """
    pass

  @abstractmethod
  def _setup_new_task(self, total_timesteps):
    pass

  @abstractmethod
  def _get_action_for_single_obs(self, obs, all_valid_successors):
    pass

  @abstractmethod
  def _process_experience(self, obs, action, rew, new_obs, done):
    pass

  def evaluate(self, n_episodes, max_steps=500):
    """evaluates model for n_episodes"""
    env = self.eval_env
    assert env is not None, "Must set an eval_env in order to evaluate!"
    results = []
    success_reward = 0

    for _ in range(n_episodes):
      obs, done = env.reset(), False
      reward = 0.
      total_reward = 0.
      steps = 0
      while not done and reward < 1. and steps < max_steps:
        action, _ = self.predict(obs)
        obs, rew, done, _ = env.step(action)
        total_reward += rew
        steps += 1
      results.append((float(np.allclose(rew, success_reward)), steps, total_reward))
      print(results)
    return results

  def learn(self,
            total_timesteps,
            max_steps=500,
            log_interval=100,
            tb_log_name=""):
    """Assumes VecEnv"""

    print("$&$ YAY YOU ARE USING THE CORRECT STABLE BASELINES #&#")
    with SetVerbosity(self.verbose), TensorboardWriter(
        self.graph, self.tensorboard_log, tb_log_name) as writer:
      self.writer = writer
      self._setup_new_task(total_timesteps=total_timesteps)

      obs = self.env.reset()

      # probably shouldn't have two different bookkeeping mechanisms here, but whatever
      legacy_episode_rewards_per_env = np.zeros((self.n_envs,))
      legacy_episode_rewards = []
      test_successes, test_steps, test_rewards, best_eval = [], [], [], 0.

      data_path = writer.get_logdir()
      if not os.path.exists(data_path):
        os.makedirs(data_path)

      self.successor_dict = {}

      with open(os.path.join(data_path, 'config.txt'), 'w') as f:
        for atr, val in self.__dict__.items():
          if isinstance(val, (str, float, int)):
            f.write("{} - {}\n".format(atr, val))

      steps = 0
      with open(os.path.join(data_path, 'test_results.txt'), 'w') as f:
        for _ in range(total_timesteps):
          steps +=1 

          # A list of state representations of all valid successors.
          tuple_obs = tuple(obs)
          if not tuple_obs in self.successor_dict:
            successors, rewards = self.env.get_actions_and_rewards()
            successors = np.array(successors)
            rewards = np.array(rewards)
            self.successor_dict[tuple_obs] = (successors, rewards)
          else:
            successors, _ = self.successor_dict[tuple_obs]

          action, _ = self._get_action_for_single_obs(obs, successors)
          new_obs, rewards, done, _ = self.env.step(action)
          done = (steps % max_steps == 0)

          obses = np.expand_dims(obs, 0)
          actions = np.expand_dims(action, 0)
          rewards = np.expand_dims(rewards, 0)  # [1, ]
          dones = np.expand_dims(done, 0)  # [1,]
          new_obses = np.expand_dims(new_obs, 0)

          # Do the learning and fetch tensorboard summaries
          summaries = self._process_experience(obses, actions, rewards, new_obses, dones)

          # Tensorboard logging
          if writer is not None and summaries:
            for summary in summaries:
              writer.add_summary(summary, self.global_step)

          # Command line and evaluation logging
          legacy_episode_rewards_per_env += rewards
          for idx in np.argwhere(dones):
            legacy_episode_rewards.append(
                legacy_episode_rewards_per_env[idx[0]])
            legacy_episode_rewards_per_env[idx[0]] = 0.

            num_episodes = len(legacy_episode_rewards)
            if self.eval_env is not None and num_episodes % self.eval_every == 0:
              sucs, stps, rews = self.evaluate(1)[0]
              test_successes.append(sucs)
              test_steps.append(stps)
              test_rewards.append(rews)
              mean_eval = np.mean(test_successes[-100:])
              mean_steps = np.mean(test_steps[-100:])
              mean_rewards = np.mean(test_rewards[-100:])
              f.write("Step {}---Test {}---Last100 {}\n".format(
                  num_episodes, test_successes[-1], mean_eval))
              summary = tf.Summary(value=[
                  tf.Summary.Value(
                      tag="test_reward/success",
                      simple_value=test_successes[-1])
              ])
              writer.add_summary(summary, num_episodes)
              summary = tf.Summary(value=[
                  tf.Summary.Value(
                      tag="test_reward/steps", simple_value=test_steps[-1])
              ])
              writer.add_summary(summary, num_episodes)
              summary = tf.Summary(value=[
                  tf.Summary.Value(
                      tag="test_reward/rewards", simple_value=test_rewards[-1])
              ])
              writer.add_summary(summary, num_episodes)
              if (len(test_successes) + 1) % 20 == 0:
                print("Evaluation perf for last 100 evaluations: {}"
                      .format(mean_eval))
                if mean_eval > best_eval:
                  print(
                      "Beat previous best eval performance of {}! Saving model..."
                      .format(best_eval))
                  best_eval = mean_eval
                  #self.save(os.path.join(data_path, 'best_eval.pkl'))

            if self.verbose >= 1 and log_interval is not None and num_episodes % log_interval == 0:
              if len(legacy_episode_rewards[-101:-1]) == 0:
                mean_100ep_reward = -np.inf
              else:
                mean_100ep_reward = round(
                    float(np.mean(legacy_episode_rewards[-101:-1])), 2)

              logger.record_tabular("steps", self.task_step)
              logger.record_tabular("episodes", num_episodes)
              logger.record_tabular("repl_buff_len", len(self.replay_buffer))
              if hasattr(self, 'replay_buffer_hindsight'
                        ) and self.replay_buffer_hindsight is not None:
                logger.record_tabular("repl_buff_hindsight_len",
                                      len(self.replay_buffer_hindsight))
              if hasattr(
                  self,
                  'landmark_generator') and self.landmark_generator is not None:
                logger.record_tabular("landmark_gen_length",
                                      len(self.landmark_generator))
              logger.record_tabular("mean 100 episode reward",
                                    mean_100ep_reward)
              if self.exploration:
                logger.record_tabular(
                    "% time spent exploring",
                    int(100 * self.exploration.value(self.task_step)))
              logger.dump_tabular()

          if steps % max_steps == 0:
            steps = 0
            obs = self.env.reset()
          else:
            obs = new_obs

    return self

  def predict(self,
              observation,
              state=None,
              mask=None,
              deterministic=False,
              goal=None):
    """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :param goal: (np.ndarray) the goal (can be None, used in goal-based environment)
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
    if state is None:
      state = self.initial_state
    if mask is None:
      mask = [False for _ in range(self.n_envs)]

    
    tuple_obs = tuple(observation)
    if not tuple_obs in self.successor_dict:
      successors, rewards = self.eval_env.get_actions_and_rewards()
      successors = np.array(successors)
      rewards = np.array(rewards)
      self.successor_dict[tuple_obs] = (successors, rewards)
    else:
      successors, _ = self.successor_dict[tuple_obs]

    action, deterministic_action = self._get_action_for_single_obs(observation, successors)

    return deterministic_action, None

  @abstractmethod
  def save(self, save_path):
    """
        Save the current parameters to file

        :param save_path: (str) the save location
        """
    # self._save_to_file(save_path, data={}, params=None)
    raise NotImplementedError()

  @classmethod
  def load(cls, load_path, env=None, **kwargs):
    data, params = cls._load_from_file(load_path)

    model = cls(policy=data["policy"], env=None, _init_setup_model=False)
    model.__dict__.update(data)
    model.__dict__.update(kwargs)
    model.env = env
    model._setup_model()

    restores = []
    for param, loaded_p in zip(model.params, params):
      restores.append(param.assign(loaded_p))
    model.sess.run(restores)

    return model

  @staticmethod
  def _save_to_file(save_path, data=None, params=None):
    _, ext = os.path.splitext(save_path)
    if ext == "":
      save_path += ".pkl"

    with open(save_path, "wb") as file:
      cloudpickle.dump((data, params), file)

  @staticmethod
  def _load_from_file(load_path):
    if not os.path.exists(load_path):
      if os.path.exists(load_path + ".pkl"):
        load_path += ".pkl"
      else:
        raise ValueError(
            "Error: the file {} could not be found".format(load_path))

    with open(load_path, "rb") as file:
      data, params = cloudpickle.load(file)

    return data, params


class SetVerbosity:

  def __init__(self, verbose=0):
    """
        define a region of code for certain level of verbosity

        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        """
    self.verbose = verbose

  def __enter__(self):
    self.tf_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
    self.log_level = logger.get_level()
    self.gym_level = gym.logger.MIN_LEVEL

    if self.verbose <= 1:
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if self.verbose <= 0:
      logger.set_level(logger.DISABLED)
      gym.logger.set_level(gym.logger.DISABLED)

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self.verbose <= 1:
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tf_level

    if self.verbose <= 0:
      logger.set_level(self.log_level)
      gym.logger.set_level(self.gym_level)


class TensorboardWriter:

  def __init__(self, graph, tensorboard_log_path, tb_log_name):
    """
        Create a Tensorboard writer for a code segment, and saves it to the log directory as its own run

        :param graph: (Tensorflow Graph) the model graph
        :param tensorboard_log_path: (str) the save path for the log (can be None for no logging)
        :param tb_log_name: (str) the name of the run for tensorboard log
        """
    self.graph = graph
    self.tensorboard_log_path = tensorboard_log_path
    self.tb_log_name = tb_log_name
    self.writer = None

  def __enter__(self):
    if self.tensorboard_log_path is not None:
      save_path = os.path.join(
          self.tensorboard_log_path, "{}_{}".format(
              self.tb_log_name,
              self._get_latest_run_id() + 1))
      self.writer = tf.summary.FileWriter(save_path, graph=self.graph)
    return self.writer

  def _get_latest_run_id(self):
    """
        returns the latest run number for the given log name and log path,
        by finding the greatest number in the directories.

        :return: (int) latest run number
        """
    max_run_id = 0
    for path in glob.glob(self.tensorboard_log_path +
                          "/{}_[0-9]*".format(self.tb_log_name)):
      file_name = path.split("/")[-1]
      ext = file_name.split("_")[-1]
      if self.tb_log_name == "_".join(file_name.split(
          "_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
        max_run_id = int(ext)
    return max_run_id

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self.writer is not None:
      self.writer.add_graph(self.graph)
      self.writer.flush()
