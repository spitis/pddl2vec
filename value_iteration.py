import tensorflow as tf
import numpy as np
import gym
import copy

from types import FunctionType
from itertools import chain

from pddl2env import PddlBasicEnv
from stable_baselines.base_models import SimpleRLModel, SetVerbosity
import stable_baselines.tf_util as tf_util
from stable_baselines.replay_buffer import ReplayBuffer, EpisodicBuffer
from stable_baselines.schedules import LinearSchedule


class SimpleValueIteration(SimpleRLModel):
    """Simple Value Iteration.

    :param env: (Gym environment or str) The environment to learn from (if
      registered in Gym, can be str)

    :param embed_dim: (int) The embedding dimensionality for each fact.

    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer

    :param exploration_fraction: (float) fraction of entire training period over
      which gamme is annealed
    :param exploration_final_eps: (float) final value of random action
      probability

    :param buffer_size: (int) size of the replay buffer
    :param train_freq: (int) update the model every `train_freq` steps
    :param batch_size: (int) size of a batched sampled from replay buffer for
      training
    :param learning_starts: (int) how many steps of the model to collect
      transitions for before learning starts

    :param target_network_update_frac: (float) fraction by which to update the
      target network every time.
    :param target_network_update_freq: (int) update the target network every
      `target_network_update_freq` steps.

    :param hindsight_mode: (str) e.g., "final", "none", "future_4"

    :param grad_norm_clipping: (float) amount of gradient norm clipping

    :param verbose: (int) the verbosity level: 0 none, 1 training information,
      2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None,
      no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the
      creation of the instance
    """

    def __init__(self,
                 env,
                 embed_dim=16,
                 gamma=0.99,
                 learning_rate=1e-3,
                 exploration_fraction=0.1,
                 exploration_final_eps=0.02,
                 buffer_size=50000,
                 train_freq=1,
                 batch_size=32,
                 learning_starts=1000,
                 target_network_update_frac=1.,
                 target_network_update_freq=500,
                 hindsight_mode=None,
                 hindsight_frac=0.,
                 grad_norm_clipping=10.,
                 verbose=0,
                 tensorboard_log=None,
                 eval_env=None,
                 eval_every=100):

        super(SimpleValueIteration, self).__init__(
            policy=None, env=env, verbose=verbose, requires_vec_env=True)

        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction

        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.target_network_update_frac = target_network_update_frac
        self.target_network_update_freq = target_network_update_freq

        self.hindsight_mode = hindsight_mode
        self.hindsight_frac = hindsight_frac

        self.grad_norm_clipping = grad_norm_clipping

        self.tensorboard_log = tensorboard_log
        self.eval_env = eval_env
        self.eval_every = eval_every

        # Below props are set in self._setup_new_task()
        self.reset = None
        self.hindsight_subbuffer = None
        self.hindsight_fn = None
        self.global_step = 0
        self.task_step = 0
        self.replay_buffer = None
        self.replay_buffer_hindsight = None
        self.state_buffer = None
        self.exploration = None

        # Several additional props to be set in self._setup_model()
        # The reason for _init_setup_model = False is to set the action/env
        # space from a saved model, without loading an environment (e.g., to do
        # transfer learning)
        self._setup_model()

    def forward_pass(self, obs):
        """Forward passes states obs through the embedding and value networks.

        Args:
          obs: An int Tensor storing the representations of a number of states.
          There is one row per state, and the number of columns is equal to the
          total number of facts (a state's representation is padded with the
          number of total facts if fewer facts are True in that state than that
          total number). So the shape of obs is [batch size, num total facts].

        Returns:
          state_vals: A float Tensor of shape [batch size,] storing the value of
            each state in the batch.
        """
        batch_size = tf.shape(obs)[0]

        # Count the number of facts that are True in each state (excl. padding).
        num_state_facts = tf.count_nonzero(
            tf.less(obs, len(self.env.task.facts)), 1, keepdims=True)
        num_state_facts = tf.cast(num_state_facts, tf.float32)
        num_state_facts = tf.tile(
            num_state_facts, multiples=[1, self.embed_dim])

        # Flatten so we can embed the facts of all states simultaneously.
        all_state_facts = tf.reshape(obs, [-1])
        fact_embeds = tf.gather(self.W_fact_embed, all_state_facts)

        # Re-structure it to store different state embeddings in different rows.
        # [batch size, num total facts, embed dim].
        fact_embeds = tf.reshape(fact_embeds, [batch_size, -1, self.embed_dim])

        # The embedding of each state is the average of the embeddings of the
        # facts that are True in it. [batch size, embed dim].
        state_embeds = tf.reduce_sum(fact_embeds, 1) / num_state_facts

        # Now forward pass through the value network.
        # TODO: Which activation should I use here?
        state_vals = tf.nn.sigmoid(
            tf.matmul(state_embeds, self.W_value_network) +
            self.b_value_network)
        return state_vals

    def _setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(graph=self.graph)

                # Placeholders.
                # A batch of states, each represented as a collection of facts.
                obs_ph = tf.placeholder(
                    shape=[None, self.env.observation_space.n],
                    dtype=tf.int32,
                    name='obs_ph')
                # A batch of corresponding next states (collections of facts).
                obs_next_ph = tf.placeholder(
                    shape=[None, self.env.observation_space.n],
                    dtype=tf.int32,
                    name='obs_next_ph')
                # A batch of all corresponding valid successor states (each is
                # a collaction of facts).
                all_valid_successors_ph = tf.placeholder(
                    shape=[None, None, self.env.observation_space.n],
                    dtype=tf.int32,
                    name='all_valid_successors_ph')

                # The number of given obs, next obs and valid successors given.
                batch_size = tf.shape(obs_ph)[0]

                with tf.variable_scope("embedding_network"):
                    W_fact_embed = tf.get_variable(
                        'fact_embed_W',
                        initializer=tf.truncated_normal(
                            [len(self.env.task.facts), self.embed_dim]))
                    # Add an extra row of all zeros. The padding will index into
                    # this row and due to it containing only zeros it won't
                    # affect the sum of fact embeddings.
                    zeros = tf.zeros([1, self.embed_dim])
                    self.W_fact_embed = tf.concat([W_fact_embed, zeros], 0)

                with tf.variable_scope("deep_value_iteration"):
                    # Initialize the weights of the value network.
                    self.W_value_network = tf.get_variable(
                        'value_network_W',
                        initializer=tf.truncated_normal([self.embed_dim, 1]))
                    self.b_value_network = tf.get_variable(
                        'value_network_b', initializer=tf.zeros(1))

                    # The value of each state in obs_ph. [batch size, 1].
                    self.values = self.forward_pass(obs_ph)

                    # The value of each *chosen* successor of states in obs_ph.
                    # [batch size, 1].
                    self.next_values = self.forward_pass(obs_next_ph)

                    # Gather all successors of all states into the batch dim.
                    # [num successors of all obs, num total facts].
                    all_valid_successors = tf.reshape(
                        all_valid_successors_ph,
                        [-1, self.env.observation_space.n])
                    # [num successors of all obs, 1].
                    all_successor_values = self.forward_pass(
                        all_valid_successors)
                    # [batch size, num successors].
                    self.all_successor_values = tf.reshape(
                        all_successor_values, [batch_size, -1])

                    # Placeholders for exploration.
                    epsilon_ph = tf.placeholder_with_default(
                        0., shape=(), name="epsilon_ph")
                    threshold_ph = tf.placeholder_with_default(
                        0., shape=(), name="param_noise_threshold_ph")
                    reset_ph = tf.placeholder(
                        tf.float32, shape=[None, 1], name="reset_ph")

                    # Set these to None for now.
                    goal_phs = goal_ph = goal_action_phs = None
                    goal_or_goalstate_ph = None

                    # For each obs in obs_ph, the index of the action that will
                    # be chosen according to the epsilon-greedy policy.
                    (actions_index, argmax_actions_index
                     ) = self.get_argmax_and_epsilon_greedy_actions(epsilon_ph)

                with tf.variable_scope("loss"):

                    # Note: naming conventions from trfl (see https://github.com/deepmind/trfl/blob/master/docs/index.md)

                    # Placeholders for bellman equation
                    r_t = tf.placeholder(tf.float32, [None], name="reward")
                    done_mask_ph = tf.placeholder(
                        tf.float32, [None], name="done")

                    # gamma
                    pcont_t = tf.constant([self.gamma])
                    pcont_t = tf.tile(pcont_t, tf.shape(r_t))
                    pcont_t *= (1 - done_mask_ph) * pcont_t

                    # Target values based on 1-step bellman.
                    target = tf.stop_gradient(r_t + pcont_t * self.next_values)
                    l2_loss = 0.5 * tf.square(target - self.values)
                    mean_l2_loss = tf.reduce_mean(l2_loss)
                    tf.summary.scalar("loss", mean_l2_loss)

                    # compute optimization op (potentially /w gradient clipping)
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate)

                    gradients = optimizer.compute_gradients(
                        mean_l2_loss, var_list=tf.trainable_variables())
                    if self.grad_norm_clipping is not None:
                        for i, (grad, var) in enumerate(gradients):
                            if grad is not None:
                                gradients[i] = (tf.clip_by_norm(
                                    grad, self.grad_norm_clipping), var)

                    # For BatchNorm
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        training_step = optimizer.apply_gradients(gradients)

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('rewards', tf.reduce_mean(r_t))
                    tf.summary.histogram('rewards', r_t)
                    if len(obs_ph.shape) == 3:
                        tf.summary.image('observation', obs_ph)
                    else:
                        tf.summary.histogram('observation', obs_ph)

                with tf_util.COMMENT("attribute assignments:"):
                    self._train_step = training_step
                    self._obs_ph = obs_ph
                    self._obs_next_ph = obs_next_ph
                    self._all_valid_successors_ph = all_valid_successors_ph
                    self._reward_ph = r_t
                    self._dones_ph = done_mask_ph
                    self._act_index = actions_index
                    self._argmax_act_index = argmax_actions_index
                    #self._goal_ph = policy.goal_ph
                    #self._goal2_ph = target_policy.goal_ph
                    #self._is_train_ph = is_train_ph
                    #self._is_train2_ph = target_policy.is_train_ph
                    #self._is_train3_ph = double_policy.is_train_ph
                    self.epsilon_ph = epsilon_ph
                    self.reset_ph = reset_ph
                    self.threshold_ph = threshold_ph

                    #if isinstance(self.goal_space,
                    #              gym.spaces.tuple_space.Tuple):
                    #    self._goal_action_ph = policy.goal_action_ph
                    #    self._goal2_action_ph = target_policy.goal_action_ph

                    with tf.variable_scope("deep_vi"):
                        self.params = tf.trainable_variables()

                # Log the state, action, goal, and landmark state
                # with tf.variable_scope("input_info_viz", reuse=False):
                #tf.summary.image(
                #    'input state',
                #    tf.cast(self._obs_ph, tf.float32),
                #    max_outputs=1)
                #tf.summary.image(
                #   'input goal',
                #    tf.cast(self._goal_ph, tf.float32),
                #    max_outputs=1)
                #action_shape = tf.reshape(self._action_ph,
                #                          [-1, self.action_space.n, 1, 1])
                #tf.summary.image(
                #    'input action',
                #    tf.cast(action_shape, tf.float32),
                #    max_outputs=1)

                with tf_util.COMMENT("attribute assignments:"):
                    self._summary_op = tf.summary.merge_all()

                # Initialize the parameters and copy them to the target network.
                with tf_util.COMMENT("graph initialization"):
                    tf_util.initialize(self.sess)
                    #self.sess.run(self.init_target_network)

                    self.summary = tf.summary.merge_all()

    def _setup_new_task(self, total_timesteps):
        """Sets up new task.

        Re-initializes step, replay buffer, and exploration schedule.
        """
        self.task_step = 0
        self.reset = np.ones([self.n_envs, 1])

        items = [("observations0", self.observation_space.shape),
                 ("rewards", (1, )),
                 ("observations1", self.observation_space.shape),
                 ("terminals1", (1, ))]

        if self.goal_space is not None:
            if not isinstance(self.goal_space, gym.spaces.tuple_space.Tuple):
                items += [
                    ("desired_goal",
                     self.env.observation_space.spaces['desired_goal'].shape)
                ]
            else:
                items += [("desired_goal", self.env.observation_space.
                           spaces['desired_goal'].spaces[0].shape)]
                items += [("desired_goal_action", self.env.observation_space.
                           spaces['desired_goal'].spaces[1].shape)]

        self.replay_buffer = ReplayBuffer(self.buffer_size, items)

        if self.hindsight_mode == 'final':
            self.hindsight_fn = (
                lambda trajectory: her_final(trajectory, self.env.compute_reward)
            )
        elif isinstance(self.hindsight_mode,
                        str) and 'future_' in self.hindsight_mode:
            _, k = self.hindsight_mode.split('_')
            self.hindsight_fn = (
                lambda trajectory: her_future(trajectory, int(k), self.env.compute_reward)
            )
            self.hindsight_frac = 1. - 1. / (1. + float(k))
        elif isinstance(self.hindsight_mode,
                        str) and 'futureactual_' in self.hindsight_mode:
            _, k, p = self.hindsight_mode.split('_')
            if self.landmark_generator is not None:
                self.hindsight_fn = HerFutureAchievedPastActualLandmark(
                    int(k), int(p), self.env.compute_reward)
            else:
                self.hindsight_fn = HerFutureAchievedPastActual(
                    int(k), int(p), self.env.compute_reward)
            self.hindsight_frac = 1. - 1. / (1. + float(k + p))
        else:
            self.hindsight_fn = None
        hindsight_items = copy.deepcopy(items)

        # Create a secondary replay buffer.
        if self.hindsight_fn is not None:
            self.replay_buffer_hindsight = ReplayBuffer(
                self.buffer_size, hindsight_items)

        self.hindsight_subbuffer = EpisodicBuffer(
            self.n_envs, self.hindsight_fn, n_cpus=min(self.n_envs, 8))

        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(
            schedule_timesteps=int(
                self.exploration_fraction * total_timesteps),
            initial_p=1.0,
            final_p=self.exploration_final_eps)

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
        """Called during training loop to get online action (/w exploration).

        Args:
          obs: A state represented as a list of (numerical indices of) facts.
          all_valid_successors: A list of states, each represented as a list of
            facts.
        """
        obs = self.pad_and_stack_obs([obs])
        all_valid_successors = self.pad_and_stack_obs(all_valid_successors)

        # Add the batch dimension to all_valid_successors. The batch size is 1
        # since all these successors are successors of the single given obs.
        all_valid_successors = np.expand_dims(all_valid_successors, 0)

        feed_dict = {
            self._obs_ph: obs,
            self._all_valid_successors_ph: all_valid_successors,
            self.epsilon_ph: self.exploration.value(self.task_step),
            self.reset_ph: self.reset
        }

        epsilon_greedy_action_ind, det_action_ind = self.sess.run(
            [self._act_index, self._argmax_act_index], feed_dict=feed_dict)

        epsilon_greedy_actions = [
            self.env.get_actions()[epsilon_greedy_action_ind[0]]
        ]
        deterministic_actions = [self.env.get_actions()[det_action_ind[0]]]
        return epsilon_greedy_actions, deterministic_actions

    def _process_experience(self, obs, action, rew, new_obs, done):
        """Called during training loop after action is taken; includes learning;
        returns a summary"""

        # new_obs fully determines the action that was taken.
        del action

        summaries = []
        expanded_done = np.expand_dims(done, 1).astype(np.float32)
        rew = np.expand_dims(rew, 1)

        goal_agent = self.goal_space is not None

        # Reset the episode if done
        self.reset = expanded_done

        # Store transition in the replay buffer, and hindsight subbuffer
        if goal_agent:
            raise NotImplementedError(
                'Need to fix this part after the changes.')
        else:
            obs_padded = self.pad_and_stack_obs([obs])
            new_obs_padded = self.pad_and_stack_obs(new_obs)

            self.replay_buffer.add_batch(obs_padded, rew, new_obs_padded,
                                         expanded_done)

        if self.hindsight_fn is not None:
            for idx in range(self.n_envs):
                # add the transition to the HER subbuffer for that worker
                # self.hindsight_subbuffer.add_to_subbuffer(
                #     idx, [obs['observation'][idx], action[idx], rew[idx], new_obs['observation'][idx], new_obs['achieved_goal'][idx], new_obs['desired_goal'][idx]])

                items = [
                    obs['observation'][idx],
                    None,
                    rew[idx],
                    new_obs['observation'][idx],
                ]
                if isinstance(self.goal_space, gym.spaces.tuple_space.Tuple):
                    items += [
                        new_obs["achieved_goal"][idx][0],
                        new_obs["desired_goal"][idx][0],
                        new_obs["achieved_goal"][idx][1],
                        new_obs["desired_goal"][idx][1]
                    ]
                else:
                    items += [
                        new_obs["achieved_goal"][idx],
                        new_obs["desired_goal"][idx]
                    ]
                self.hindsight_subbuffer.add_to_subbuffer(idx, items)

                if done[idx]:
                    # commit the subbuffer
                    self.hindsight_subbuffer.commit_subbuffer(idx)
                    if len(self.hindsight_subbuffer) == self.n_envs:
                        hindsight_experiences = self.hindsight_subbuffer.process_trajectories(
                        )

                        # for hindsight_experience in chain.from_iterable(self.hindsight_subbuffer.process_trajectories()):
                        #   self.replay_buffer_hindsight.add(*hindsight_experience)

                        for hindsight_experience in chain.from_iterable(
                                hindsight_experiences):
                            self.replay_buffer_hindsight.add(
                                *hindsight_experience)
                        self.hindsight_subbuffer.clear_main_buffer()

        self.global_step += self.n_envs
        for _ in range(self.n_envs):
            self.task_step += 1
            # If have enough data, train on it.
            if self.task_step > self.learning_starts:
                if self.task_step % self.train_freq == 0:
                    if goal_agent:
                        raise NotImplementedError('Need to adjust this.')
                    else:
                        (obses_t, rewards, obses_tp1,
                         dones) = self.replay_buffer.sample(self.batch_size)

                    rewards = np.squeeze(rewards, 1)
                    dones = np.squeeze(dones, 1)

                    feed_dict = {
                        self._obs_ph: obses_t,
                        self._obs_next_ph: obses_tp1,
                        self._reward_ph: rewards,
                        self._dones_ph: dones,
                    }

                    if goal_agent:
                        feed_dict[self._goal_ph] = desired_g
                        # Assuming that the goal does not change in episode.
                        feed_dict[self._goal2_ph] = desired_g

                        if isinstance(self.goal_space,
                                      gym.spaces.tuple_space.Tuple):
                            feed_dict[self._goal2_action_ph] = desired_g_action
                            feed_dict[self._goal_action_ph] = desired_g_action

                    _, summary = self.sess.run(
                        [self._train_step, self._summary_op],
                        feed_dict=feed_dict)

                    summaries.append(summary)

                #if self.task_step % self.target_network_update_freq == 0:
                #self.sess.run(self.update_target_network)

        return summaries

    def save(self, save_path):
        pass

    #    # Things set in the __init__ method should be saved here, because the
    #    # model is called with default args on load(), which are subsequently
    #    # updated using this dict.
    #    data = {
    #        "learning_rate": self.learning_rate,
    #        "gamma": self.gamma,
    #        "exploration_final_eps": self.exploration_final_eps,
    #        "exploration_fraction": self.exploration_fraction,
    #        "param_noise": self.param_noise,
    #        "learning_starts": self.learning_starts,
    #        "train_freq": self.train_freq,
    #        "batch_size": self.batch_size,
    #        "buffer_size": self.buffer_size,
    #        "target_network_update_frac": self.target_network_update_frac,
    #        "target_network_update_freq": self.target_network_update_freq,
    #        'hindsight_mode': self.hindsight_mode,
    #        'landmark_training': self.landmark_training,
    #        'landmark_mode': self.landmark_mode,
    #        'landmark_training_per_batch': self.landmark_training_per_batch,
    #        'landmark_width': self.landmark_width,
    #        'landmark_error': self.landmark_error,
    #        "double_q": self.double_q,
    #        "grad_norm_clipping": self.grad_norm_clipping,
    #        "tensorboard_log": self.tensorboard_log,
    #        "verbose": self.verbose,
    #        "observation_space": self.observation_space,
    #        "action_space": self.action_space,
    #        "n_envs": self.n_envs,
    #        "_vectorize_action": self._vectorize_action
    #    }

    #    # Model paramaters to be restored
    #    params = self.sess.run(self.params)

    #    self._save_to_file(save_path, data=data, params=params)

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
    env = PddlBasicEnv(
        domain='pddl_files/modded_transport/domain.pddl',
        instance='pddl_files/modded_transport/6/p01.pddl')
    model = SimpleValueIteration(
        env=env, tensorboard_log='/scratch/gobi1/eleni/csc2542/')
    model.learn(total_timesteps=100000, tb_log_name='value_iteration')
