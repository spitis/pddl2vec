import tensorflow as tf
import numpy as np
import gym
import copy
import os

from itertools import chain

from pddl2env import PddlBasicEnv, PddlSimpleMultiGoalEnv
from stable_baselines.base_models_sp import BaseRLModel, SetVerbosity
import stable_baselines.tf_util as tf_util
from stable_baselines.replay_buffer import ReplayBuffer, EpisodicBuffer, her_future, HerFutureAchievedPastActual

tf.flags.DEFINE_string('tensorboard_log', './test/', 'Where to store the logs.')
tf.flags.DEFINE_string('ckpt_dir', './test/', 'Where to store the checkpoints.')
tf.flags.DEFINE_string('restore_dir', '', 'The location from where to grab '
    'and restore the latest checkpoint.')
FLAGS = tf.flags.FLAGS

def embed(placeholder, embedding_mtx):
  """Embeds the states (or goals) into latent space using embedding mtx
  Returns:
    latent: A float Tensor of shape [batch size, emb_dim] storing the embedding of
      each placeholder in the batch."""
  batch_size = tf.shape(placeholder)[0]

  # Flatten so we can embed the facts of all states simultaneously.
  all_state_facts = tf.reshape(placeholder, [-1])
  fact_embeds = tf.nn.embedding_lookup(embedding_mtx, all_state_facts)

  # Re-structure it to store different state embeddings in different rows.
  # [batch size, num total facts, embed dim].
  fact_embeds = tf.reshape(fact_embeds, [batch_size, -1, embedding_mtx.get_shape()[1]])

  # The embedding of each state is the average of the embeddings of the
  # facts that are True in it. [batch size, embed dim].
  h = tf.reduce_mean(fact_embeds, 1)

  return h

def forward_pass(latent, layers):
  """Forward passes embedded states(+goals) through the value networks.

    Args:
      latent: 

    Returns:
      state_vals: A float Tensor of shape [batch size,] storing the value of
        each state in the batch.
    """
  h = latent
  # Now forward pass through the value network.
  for layer in layers[:-1]:
    h = tf.nn.relu(layer(h))
  h = layers[-1](h)
  return h

class SimpleValueIteration(BaseRLModel):

  def __init__(self,
               env,
               embed_dim=256,
               layer_sizes=[128,64,1],
               gamma=0.98,
               learning_rate=3e-3,
               *,
               exploration_fraction=.25,
               buffer_size=1000000,
               train_freq=10,
               batch_size=500,
               learning_starts=2500,
               target_network_update_frac=0.05,
               target_network_update_freq=50,
               hindsight_mode=None,
               grad_norm_clipping=5.,
               verbose=1,
               tensorboard_log=None,
               ckpt_dir=None,
               restore_dir=None,
               eval_env=None,
               eval_every=10):

    super(SimpleValueIteration, self).__init__(env=env, verbose=verbose)

    self.embed_dim = embed_dim
    self.layer_sizes = layer_sizes
    self.learning_rate = learning_rate
    self.gamma = gamma

    self.exploration_fraction = exploration_fraction

    self.learning_starts = learning_starts
    self.train_freq = train_freq
    self.batch_size = batch_size
    self.buffer_size = buffer_size

    self.target_network_update_frac = target_network_update_frac
    self.target_network_update_freq = target_network_update_freq

    self.hindsight_mode = hindsight_mode
    self.hindsight_frac = 0.

    self.grad_norm_clipping = grad_norm_clipping

    self.tensorboard_log = tensorboard_log
    self.ckpt_dir = ckpt_dir
    self.restore_dir = restore_dir
    self.eval_env = eval_env
    self.eval_every = eval_every

    # Below props are set in self._setup_new_task()
    self.hindsight_subbuffer = None
    self.hindsight_fn = None
    self.global_step = 0
    self.task_step = 0
    self.replay_buffer = None
    self.replay_buffer_hindsight = None
    self.state_buffer = None

    # Several additional props to be set in self._setup_model()
    # The reason for _init_setup_model = False is to set the action/env
    # space from a saved model, without loading an environment (e.g., to do
    # transfer learning)
    self._setup_model()


  def _setup_model(self):
    with SetVerbosity(self.verbose):
      self.graph = tf.Graph()
      with self.graph.as_default():
        ob_space = self.observation_space
        goal_space = self.goal_space
        
        ## Placeholders ##

        # A batch of states, each represented as a collection of facts.
        obs_ph  = tf.placeholder(shape=[None, len(ob_space.nvec)], dtype=tf.int32, name='obs_ph')
        goal_ph = None
        if goal_space:
          goal_ph = tf.placeholder(shape=[None, len(goal_space.nvec)], dtype=tf.int32, name='goal_ph')

        # A batch of all corresponding valid successor states (each is a collaction of facts).
        all_valid_successors_ph        = tf.placeholder(shape=[None, None, len(ob_space.nvec)], dtype=tf.int32, name='all_valid_successors_ph')
        all_valid_successors_target_ph = tf.placeholder(shape=[None, None, len(ob_space.nvec)], dtype=tf.int32, name='all_valid_successors_target_ph')
        target_goal_ph = None
        if goal_space:
          target_goal_ph = tf.placeholder(shape=[None, len(goal_space.nvec)], dtype=tf.int32, name='target_goal_ph')

        # The number of given obs, next obs and valid successors given.
        batch_size = tf.shape(obs_ph)[0]

        with tf.variable_scope("main_network"):
          with tf.variable_scope("embedding"):
            W_fact_embed = tf.get_variable('fact_embed_W', initializer=tf.truncated_normal([len(self.env.task.facts), self.embed_dim]))

          with tf.variable_scope("layers"):
            # Initialize the weights of the value network.
            layers = []
            for layer_size in self.layer_sizes:
              layers.append(tf.layers.Dense(layer_size))

          # The value of each state in obs_ph. [batch size, 1].
          latent = embed(obs_ph, W_fact_embed)
          if goal_space:
            latent_goal = embed(goal_ph, W_fact_embed)
            latent = tf.concat([latent, latent_goal], axis=-1)
          self.values = forward_pass(latent, layers)

          # Gather all successors of all states into the batch dim.
          # [num successors of all obs, num total facts].
          all_valid_successors = tf.reshape(all_valid_successors_ph, [-1, len(ob_space.nvec)])

          # [num successors of all obs, 1].
          all_successor_latent = embed(all_valid_successors, W_fact_embed)
          if goal_space:
            all_successor_latent = tf.reshape(all_successor_latent, [batch_size, -1, self.embed_dim])
            latent_goal = tf.reshape(latent_goal, [batch_size, 1, self.embed_dim])
            latent_goal_tiled = tf.broadcast_to(latent_goal, shape=tf.shape(all_successor_latent))
            all_successor_latent = tf.concat([all_successor_latent, latent_goal_tiled], axis=-1)
            all_successor_latent = tf.reshape(all_successor_latent, [-1, self.embed_dim * 2])
          all_successor_values = forward_pass(all_successor_latent, layers)

          # [batch size, num successors].
          self.all_successor_values = tf.reshape(all_successor_values, [batch_size, -1])

          self.main_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

        with tf.variable_scope("target_network", reuse=False):
          with tf.variable_scope("embedding"):
            W_fact_embed_target = tf.get_variable('fact_embed_W_target', initializer=tf.truncated_normal([len(self.env.task.facts), self.embed_dim]))

          with tf.variable_scope("layers"):
            # Initialize the weights of the value network.
            target_layers = []
            for layer_size in self.layer_sizes:
              target_layers.append(tf.layers.Dense(layer_size))

          # Gather all successors of all states into the batch dim.
          # [num successors of all obs, num total facts].
          all_valid_successors = tf.reshape(all_valid_successors_target_ph, [-1, len(ob_space.nvec)])

          # [num successors of all obs, 1].
          target_successor_latent = embed(all_valid_successors, W_fact_embed_target)
          if goal_space:
            latent_goal = embed(target_goal_ph, W_fact_embed_target)
            target_successor_latent = tf.reshape(target_successor_latent, [batch_size, -1, self.embed_dim])
            latent_goal = tf.reshape(latent_goal, [batch_size, 1, self.embed_dim])
            latent_goal_tiled = tf.broadcast_to(latent_goal, shape=tf.shape(target_successor_latent))
            target_successor_latent = tf.concat([target_successor_latent, latent_goal_tiled], axis=-1)
            target_successor_latent = tf.reshape(target_successor_latent, [-1, self.embed_dim * 2])
          target_successor_values = forward_pass(target_successor_latent, target_layers)

          # [batch size, num successors].
          r_t = tf.placeholder(tf.float32, [None, None], name="reward")
          target_successor_values = tf.reshape(target_successor_values, [batch_size, -1])

          next_q_values = r_t + self.gamma * (-1 * r_t) * target_successor_values

          # The value of each *chosen* successor of states in obs_ph.
          # [batch size, 1].
          self.next_values = tf.reduce_max(next_q_values, axis=-1, keepdims=True)

          self.target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)


        # For each obs in obs_ph, the index of the action that will
        # be chosen according to the epsilon-greedy policy.
        (actions_index, argmax_actions_index) = self.get_argmax_and_epsilon_greedy_actions(
              self.exploration_fraction)

        with tf.variable_scope("loss"):
          # Target values based on 1-step bellman.
          target = tf.clip_by_value(tf.stop_gradient(self.next_values), -np.inf, 0.)
          l2_loss = 0.5 * tf.square(target - self.values)
          mean_l2_loss = tf.reduce_mean(l2_loss)
          tf.summary.scalar("loss", mean_l2_loss)
          tf.summary.histogram("preds", self.values)

          # compute optimization op (potentially /w gradient clipping)
          optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

          gradients = optimizer.compute_gradients(
              mean_l2_loss, var_list=tf.trainable_variables())
          if self.grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients):
              if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping),
                                var)

          with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            training_step = optimizer.apply_gradients(gradients)

        with tf.name_scope('update_target_network_ops'):
          init_target_network = []
          update_target_network = []
          for var, var_target in zip(
              sorted(self.main_vars, key=lambda v: v.name), sorted(self.target_vars, key=lambda v: v.name)):
            new_target = self.target_network_update_frac       * var +\
                         (1 - self.target_network_update_frac) * var_target
            update_target_network.append(var_target.assign(new_target))
            init_target_network.append(var_target.assign(var))
          update_target_network = tf.group(*update_target_network)
          init_target_network = tf.group(*init_target_network)

        with tf.variable_scope("input_info", reuse=False):
          tf.summary.scalar('rewards', tf.reduce_mean(r_t))

        with tf_util.COMMENT("attribute assignments:"):
          self._train_step = training_step
          self._obs_ph = obs_ph
          self._all_valid_successors_ph = all_valid_successors_ph
          self._all_valid_successors_target_ph = all_valid_successors_target_ph
          self._reward_ph = r_t
          self._act_index = actions_index
          self._argmax_act_index = argmax_actions_index
          self.update_target_network = update_target_network
          self._goal_ph = goal_ph
          self._goal2_ph = target_goal_ph

          with tf.variable_scope("deep_vi"):
            self.params = tf.trainable_variables()

          self._summary_op = tf.summary.merge_all()

        # Create an object for saving model checkpoints.
        if self.ckpt_dir is not None:
          self.saver = tf.train.Saver(max_to_keep=10)
          if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        # Now initialize a session.
        self.sess = tf_util.make_session(graph=self.graph)

        # Initialize the parameters and copy them to the target network.
        with tf_util.COMMENT("graph initialization"):
          tf_util.initialize(self.sess)

          # If specified, restore the weights from a pretrained model.
          if self.restore_dir:
            restore_from = tf.train.latest_checkpoint(self.restore_dir)
            self.saver.restore(self.sess, restore_from)
            print('Restored weights from {}'.format(restore_from))

          self.sess.run(init_target_network)

          self.summary = tf.summary.merge_all() 

  def _setup_new_task(self, total_timesteps):
    """Sets up new task.

        Re-initializes step, replay buffer, and exploration schedule.
        """
    self.task_step = 0

    if isinstance(self.observation_space.shape[0], tuple):
      items = [("observations0", (self.observation_space.shape[0])),
                ("rewards", (1,)),
                ("observations1", (self.observation_space.shape[0])),
                ("terminals1", (1,))]
    else:
      items = [("observations0", (self.observation_space.shape[0],)),
                ("rewards", (1,)),
                ("observations1", (self.observation_space.shape[0],)),
                ("terminals1", (1,))]

    if self.goal_space is not None:
      if isinstance(self.goal_space.shape[0], tuple):
        items += [("desired_goal", (self.goal_space.shape[0]))]
      else:
        items += [("desired_goal", (self.goal_space.shape[0],))]

    print(items)

    self.replay_buffer = ReplayBuffer(self.buffer_size, items)


    if isinstance(self.hindsight_mode, str):
      assert self.goal_space is not None

    if isinstance(self.hindsight_mode, str) and 'future_' in self.hindsight_mode:
      _, k = self.hindsight_mode.split('_')
      self.hindsight_fn = (
          lambda trajectory: her_future(trajectory, int(k), self.env.compute_reward, 0, True)
      )
      self.hindsight_frac = 1. - 1. / (1. + float(k))
    elif isinstance(self.hindsight_mode, str) and 'futureactual_' in self.hindsight_mode:
      _, k, p = self.hindsight_mode.split('_')
      self.hindsight_fn = HerFutureAchievedPastActual(int(k), int(p), self.env.compute_reward, 0, True)
      self.hindsight_frac = 1. - 1. / (1. + float(k + p))
    else:
      self.hindsight_fn = None

    self.hindsight_subbuffer = EpisodicBuffer(1, self.hindsight_fn, n_cpus=min(self.n_envs, 8))

  def pad_and_stack_obs(self, states):
    """Pads state representations to a fixed-length array and stacks them.

        Args:
          states: A list of np arrays, each representing a state. Each such
            array stores the inds of True facts in that state.

        Returns:
          A np array that has as many rows as there are given states, and as
          many columns as the total number of facts in the environment. The
          representations of states in which fewer facts are True than that
          total number are padded with that number at the end. e.g. if there are
          four facts and a state's representation is [3, 1], its padded version
          will be [3, 1, 4, 4].
        """
    num_total_facts = len(self.env.task.facts)
    states = [
        np.pad(
            state, (0, num_total_facts - len(state)),
            'constant',
            constant_values=num_total_facts) for state in states
    ]
    states = np.stack(states, 0)
    return states

  def _get_action_for_single_obs(self, obs, all_valid_successors):
    """
    Called during training loop to get online action (/w exploration).

    Args:
      obs: A state represented as a list of (numerical indices of) facts.
      all_valid_successors: A list of states, each represented as a list of facts.
    """
    #obs = self.pad_and_stack_obs([obs])
    #all_valid_successors = self.pad_and_stack_obs(all_valid_successors)

    # Add the batch dimension to all_valid_successors. The batch size is 1
    # since all these successors are successors of the single given obs.
    # all_valid_successors = np.expand_dims(all_valid_successors, 0)

    if self.goal_space is not None:
      feed_dict = {
        self._obs_ph: [obs['observation']],
        self._goal_ph: [obs['desired_goal']],
        self._all_valid_successors_ph: [all_valid_successors],
      }
    else:
      feed_dict = {
          self._obs_ph: [obs],
          self._all_valid_successors_ph: [all_valid_successors],
      }

    epsilon_greedy_action_ind, det_action_ind = self.sess.run(
        [self._act_index, self._argmax_act_index], feed_dict=feed_dict)

    epsilon_greedy_action = all_valid_successors[epsilon_greedy_action_ind[0]]
    deterministic_action = all_valid_successors[det_action_ind[0]]
    return epsilon_greedy_action, deterministic_action

  def _process_experience(self, obs, action, rew, new_obs, done):
    """Called during training loop after action is taken; includes learning;
        returns a summary"""

    summaries = []
    expanded_done = np.expand_dims(done, 1).astype(np.float32)
    rew = np.expand_dims(rew, 1)

    goal_agent = self.goal_space is not None

    # Store transition in the replay buffer, and hindsight subbuffer
    if goal_agent:
      self.replay_buffer.add_batch(obs['observation'], rew, new_obs['observation'], expanded_done, new_obs['desired_goal'])
    else:
      self.replay_buffer.add_batch(obs, rew, new_obs, expanded_done)

    if self.hindsight_fn is not None:
      for idx in range(self.n_envs):
        # add the transition to the HER subbuffer for that worker
        self.hindsight_subbuffer.add_to_subbuffer(idx,
          [obs['observation'][idx], action[idx], rew[idx], new_obs['observation'][idx], new_obs['achieved_goal'][idx], new_obs['desired_goal'][idx]])

        if done[idx]:
          # commit the subbuffer
          self.hindsight_subbuffer.commit_subbuffer(idx)
          if len(self.hindsight_subbuffer) == self.n_envs:
            hindsight_experiences = self.hindsight_subbuffer.process_trajectories()
            for h in chain.from_iterable(hindsight_experiences):             
              self.replay_buffer.add(h[0],h[2],h[3],h[4],h[5])
            self.hindsight_subbuffer.clear_main_buffer()

    self.global_step += 1
    self.task_step += 1
    # If have enough data, train on it.
    if self.task_step > self.learning_starts:
      if self.task_step % self.train_freq == 0:
        if goal_agent:
          (obses_t, rewards, obses_tp1, dones, desired_g) = self.replay_buffer.sample(self.batch_size)
          
          successors_rewards_list = [self.successor_dict[tuple(o)] for o in obses_t]
          successors, _ = zip(*successors_rewards_list)
          new_sucs = []
          new_rews = []
          max_succ_len = max(map(len, successors))
          for (suc, _), dg in zip(successors_rewards_list, desired_g):
            if len(suc) == max_succ_len:
              new_sucs.append(suc)
              new_rews.append([self.env.compute_reward(s, dg) for s in suc])
            else:
              new_suc = list(suc) + [suc[-1]] * (max_succ_len - len(suc))
              new_sucs.append(np.array(new_suc))
              new_rews.append([self.env.compute_reward(s, dg) for s in new_suc])

        else:
          (obses_t, rewards, obses_tp1, dones) = self.replay_buffer.sample(self.batch_size)


          #successors = np.expand_dims(obses_tp1, 1)
          successors_rewards_list = [self.successor_dict[tuple(o)] for o in obses_t]
          successors, rewards = zip(*successors_rewards_list)
          new_sucs = []
          new_rews = []
          max_succ_len = max(map(len, successors))
          for suc, rew in successors_rewards_list:
            if len(suc) == max_succ_len:
              new_sucs.append(suc)
              new_rews.append(rew)
            else:
              new_suc = list(suc) + [suc[-1]] * (max_succ_len - len(suc))
              new_rew = list(rew) + [rew[-1]] * (max_succ_len - len(suc))
              new_sucs.append(np.array(new_suc))
              new_rews.append(np.array(new_rew))

        feed_dict = {
            self._obs_ph: obses_t,
            self._all_valid_successors_target_ph: new_sucs,
            self._reward_ph: new_rews,
        }

        if goal_agent:
          feed_dict[self._goal_ph] = desired_g
          # Assuming that the goal does not change in episode.
          feed_dict[self._goal2_ph] = desired_g

        _, summary = self.sess.run([self._train_step, self._summary_op], feed_dict=feed_dict)

        summaries.append(summary)

        if self.task_step % self.target_network_update_freq == 0:
          self.sess.run(self.update_target_network)

    return summaries

  def save(self, save_path):
    # Things set in the __init__ method should be saved here, because the
    # model is called with default args on load(), which are subsequently
    # updated using this dict.
    data = {
        "learning_rate": self.learning_rate,
        "gamma": self.gamma,
        "exploration_fraction": self.exploration_fraction,
        "param_noise": self.param_noise,
        "learning_starts": self.learning_starts,
        "train_freq": self.train_freq,
        "batch_size": self.batch_size,
        "buffer_size": self.buffer_size,
        "target_network_update_frac": self.target_network_update_frac,
        "target_network_update_freq": self.target_network_update_freq,
        'hindsight_mode': self.hindsight_mode,
        "double_q": self.double_q,
        "grad_norm_clipping": self.grad_norm_clipping,
        "tensorboard_log": self.tensorboard_log,
        "verbose": self.verbose,
        "observation_space": self.observation_space,
        "action_space": self.action_space,
        "_vectorize_action": self._vectorize_action
    }

    # Model paramaters to be restored
    params = self.sess.run(self.params)

    self._save_to_file(save_path, data=data, params=params)

  def get_argmax_and_epsilon_greedy_actions(self, epsilon_placeholder):
    """Gets the inds of obs' e-greedily and greedily-chosen next states."""
    deterministic_actions = tf.argmax(self.all_successor_values, axis=1)
    batch_size = tf.shape(self.all_successor_values)[0]
    num_valid_successors = tf.cast(
        tf.shape(self.all_successor_values)[1], tf.int64)
    random_actions = tf.random_uniform(
        tf.stack([batch_size]),
        minval=0,
        maxval=num_valid_successors,
        dtype=tf.int64)
    chose_random = tf.random_uniform(
        tf.stack([batch_size]), minval=0, maxval=1,
        dtype=tf.float32) < epsilon_placeholder

    epsilon_greedy_actions = tf.where(chose_random, random_actions,
                                      deterministic_actions)
    return epsilon_greedy_actions, deterministic_actions

if __name__ == '__main__':
  env = PddlSimpleMultiGoalEnv(
      domain='pddl_files/modded_transport/domain.pddl',
      instance='pddl_files/modded_transport/ptest5.pddl')
  eval_env = copy.deepcopy(env)
  model = SimpleValueIteration(env=env, tensorboard_log=FLAGS.tensorboard_log, 
      ckpt_dir=FLAGS.ckpt_dir, restore_dir=FLAGS.restore_dir, eval_env=eval_env,
      hindsight_mode='future_8')

  # env = PddlBasicEnv(
  #     domain='pddl_files/modded_transport/domain.pddl',
  #     instance='pddl_files/modded_transport/ptest.pddl')
  # eval_env = copy.deepcopy(env)
  # model = SimpleValueIteration(env=env, tensorboard_log=FLAGS.tensorboard_log, 
  #     ckpt_dir=FLAGS.ckpt_dir, restore_dir=FLAGS.restore_dir, eval_env=eval_env)
  model.learn(total_timesteps=1000000, max_steps=100, tb_log_name='value_iteration')
