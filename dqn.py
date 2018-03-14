#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import scipy
import gym
import sys
import copy
import argparse
import random
import matplotlib.pyplot as plt
from collections import deque
import pickle
import getch

def tf_shape_to_np_shape(shape):
    return [-1 if x.value is None else x.value for x in shape]

# resizing and down-sizing the observations we see to reduce the state space
# and make the cnn converge faster
def parse_space_obs(obs):
    # grayscaling
    image = obs
    gray_image = image[...,:3].dot([0.299, 0.587, 0.114])
    # magic numbers that people use to reduce images grayscale

    # resizing
    gray_small_image = scipy.misc.imresize(gray_image, (110, 80))
    obs = gray_small_image
    return obs


# step function we use as a wrapper for env.step
def step(env, action, game):
    obs, reward, done, info = env.step(action)
    if game == 'SpaceInvaders-v0':
        obs = parse_space_obs(obs)

    return (obs, reward, done, info)

# reset function we use as a wrapper for env.reset
def reset(env, game):
    if game == 'SpaceInvaders-v0':
        return parse_space_obs(env.reset())
    return env.reset()

class QNetwork():
    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, model, environment_name, debug=False,
        learning_rate=1e-4, replay=False, num_actions=None):
        # Define your networ3 architecture here. It is also a good idea to
        # define any training operations and optimizers here, initialize your
        # variables, or alternately compile your model here.
        if debug:
            self.report = print
        else:
            def id(*args, **kwargs):
                return
            self.report = id

        self.replay = replay


        self.learning_rate = learning_rate
        self.sess = tf.Session()

        self.env = gym.make(environment_name)

        if num_actions is None:
            self.num_actions = self.env.action_space.n
        else:
            self.num_actions = num_actions


        self.num_observations = [self.env.observation_space.shape[0]]

        # we manually have to write down the observation space for the
        # input in order to make space-invaders behave as if it was being
        # given [110, 80] images
        if environment_name == 'SpaceInvaders-v0':
            self.num_observations = [110, 80]

        self.create_model(model)
        self.sess.run(tf.global_variables_initializer())


    # create_model: Initializes a tensorflow model
    def create_model(self, model='linear', hidden_width=100):
        # MountainCar has 2 state inputs: position and velocity
        #
        # Cartpole has 4 state inputs: position, velocity, pole angle, pole
        # velocity at tip
        with self.sess:
            self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None,
                *self.num_observations])

            self.action_ph = tf.placeholder(shape=[None], dtype=tf.int32)
            self.expected_ph = tf.placeholder(tf.float32, shape=[None])

            print('model is {}'.format(model))
            print('using replay? {}'.format(self.replay))


            if model == 'deep' or model == 'linear':
                activation_fn = None
                if model == 'deep':
                    activation_fn = tf.nn.tanh

                # Note: plots for writeup generated with 4 layer architecture of
                # 3*hidden_width, 3*hidden_width, 2*hidden_width, 3*hidden_width
                inner1 = tf.contrib.layers.fully_connected(self.state_ph,
                    3*hidden_width, activation_fn=activation_fn)

                inner2 = tf.contrib.layers.fully_connected(inner1,
                    2*hidden_width, activation_fn=activation_fn)

                inner3 = tf.contrib.layers.fully_connected(inner2,
                    hidden_width, activation_fn=activation_fn)

                # the output
                self.qvalue_logits = tf.contrib.layers.fully_connected(
                    inner3,
                    self.num_actions,
                    activation_fn = None,
                )

            elif model == 'cnn': # convolutional layer for space invaders
                # Model architecture similar to vgg
                reshape = tf.reshape(self.state_ph, [*tf_shape_to_np_shape(self.state_ph.shape), 1])
                inner1 = tf.layers.conv2d(
                    inputs=reshape,
                    filters=16,
                    kernel_size=[8, 8],
                    strides=(4, 4),
                    padding="same",
                    activation=tf.nn.relu
                )

                pool1 = tf.layers.max_pooling2d(
                    inputs=inner1,
                    pool_size=[2, 2],
                    strides=2
                )

                inner2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=32,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=tf.nn.relu
                )

                pool2 = tf.layers.max_pooling2d(
                    inputs=inner2,
                    pool_size=[2, 2],
                    strides=2
                )

                '''
                inner3 = tf.layers.conv2d(
                    inputs=pool2,
                    filters=8,
                    kernel_size=[3, 3],
                    padding="same",
                    activation=tf.nn.relu
                )

                pool3 = tf.layers.max_pooling2d(
                    inputs=inner3,
                    pool_size=[2, 2],
                    strides=2
                )

                pool3_flat = tf.reshape(pool3, [-1, 1040])
                '''
                pool2_flat = tf.reshape(pool2, [-1, 192])


                fully_connected = tf.contrib.layers.fully_connected(pool2_flat,
                    256, activation_fn=tf.nn.relu)

                self.qvalue_logits = tf.contrib.layers.fully_connected(
                    fully_connected,
                    self.num_actions,
                    activation_fn = None,
                )

            elif model == 'dueling':
                # Decoupling the shared first layer into two makes training
                # faster - takes less time to not get -200 reward on MountainCar
                v_inner1 = tf.contrib.layers.fully_connected(self.state_ph,
                    50, activation_fn=None)

                # dropout1 = tf.nn.dropout(inner1, 0.5)
                v_inner2 = tf.contrib.layers.fully_connected(v_inner1, 50, )
                v_inner3 = tf.contrib.layers.fully_connected(v_inner2, 50, )

                # dropout1 = tf.nn.dropout(inner1, 0.5)

                a_inner1 = tf.contrib.layers.fully_connected(self.state_ph,
                    50, activation_fn=None)

                a_inner2 = tf.contrib.layers.fully_connected(a_inner1, 50, )
                a_inner3 = tf.contrib.layers.fully_connected(a_inner2, 50, )

                # dropout1 = tf.nn.dropout(inner1, 0.5)

                # The output
                self.v_qvalue_logits = tf.contrib.layers.fully_connected(
                    v_inner3,
                    self.num_actions,
                    activation_fn = None,
                )

                # Take the expectation with respect to our policy - for now
                # let's do this according to a simple greedy policy for
                # simplicity, can change parameters to take according to an
                # epsilon-greedy policy.

                #self.input_policy = tf.placeholder(shape=[None],
                #   dtype=tf.float32)
                self.v_out = tf.reduce_max(self.v_qvalue_logits)

                self.a_qvalue_logits = tf.contrib.layers.fully_connected(
                    a_inner3,
                    self.num_actions,
                    activation_fn = None,
                )

                self.avg_a = tf.reduce_mean(self.a_qvalue_logits)

                # Let's use the second equation, (9), for now, which is
                # subtracting an average across the advantage functions. We can
                # always replace it with (8), which is subtracting the max
                # action instead.

                self.qvalue_logits = self.v_out + (self.a_qvalue_logits -
                    self.avg_a)

            self.action = tf.argmax(input=self.qvalue_logits, axis=1)

            # does same thing as tf.gather
            self.prediction = tf.reduce_sum(
                tf.multiply(
                    self.qvalue_logits,
                    tf.one_hot(self.action_ph, self.num_actions),
                ),
                axis=1,
            )

            self.loss = tf.reduce_mean(tf.square(self.expected_ph -
                self.prediction))
            self.optimizer =    \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(loss=self.loss)


    def update_params(self, prev_obs, action, reward, obs, done,
        discount_factor):
        q_primes = self.get_qvalues(obs)
        #Train our network using target and predicted Q values

        done = np.vectorize(lambda b: 0 if b else 1)(done)
        discount_factor = done * discount_factor

        feed_dict = {
            self.state_ph: #state_ph is the placeholder value for the state
                prev_obs.reshape((tf_shape_to_np_shape(self.state_ph.shape))),
            self.action_ph: np.array(action), # action_ph
            self.expected_ph : reward + (discount_factor
                * np.max(q_primes, axis=1))
        }

        _,loss, pred, exp = \
            self.sess.run([self.train_op, self.loss, self.prediction,
                self.expected_ph],feed_dict=feed_dict)

        return loss, pred, exp

    def get_qvalues(self, obs):
        # return's the qvalue for a given state
        return self.sess.run(
            self.qvalue_logits,
            feed_dict={self.state_ph:
                obs.reshape((tf_shape_to_np_shape(self.state_ph.shape)))},
        )


    def save_model_weights(self, environment_name):
        # Helper function to save your model / weights.
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "./%s_model.ckpt" % environment_name)

    def load_model(self, environment_name):
        # Helper function to load an existing model.
        saver = tf.train.Saver()
        saver.restore(self.sess, "/saved_models/%s_model.ckpt" % environment_name)

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights.
        raise NotImplementedError


class Replay_Memory():

    def __init__(self, memory_size=600, burn_in=200):
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into
        # the memory from the randomly initialized agent. Memory size is the
        # maximum size after which old elements in the memory are replaced.
        #
        # A simple (if not the most efficient) was to implement the memory is
        # as a list of transitions.

        # Can change this data structure later - we will use the burn_in
        # function under DQN_Agent to do the actual burning-in. Need a deque
        # instead of a list because we want to have queue functionality.
        self.memory = deque([])
        self.weights = deque([]) # we have weights that correspond to the
                                 # the probability that we choose the item
                                 # once we sample. The default weight is 1
                                 # and increasing the weight linearly increases their
                                 # chance of being chosen.
        self.weights_sum = 0
        self.current_size = 0
        self.memory_size = memory_size
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e.
        #
        # state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.

        # Samples without replacement
        idx = np.random.choice(
            range(len(self.memory)),
            batch_size,
            p=(np.array(self.weights) / self.weights_sum).reshape([-1])
        )
        return np.array([self.memory[v] for v in idx])

    def append(self, transition, weight=1):
        # Appends transition to the memory.

        # The "left" side of the deque is first-in (least recently added) and
        # the "right" side is last-in (most recently added)
        if self.current_size == self.memory_size:
            # Discard oldest transition
            (prev_obs, action, reward, done, obs) = self.memory.popleft()
            self.weights_sum -= self.weights.popleft()
            self.current_size -= 1

        self.memory.append(transition)
        self.weights.append(weight)
        self.weights_sum += weight
        self.current_size += 1



class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted
    #       by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the
    #       environment.
    # (4) Create a function to test the Q Network's performance on the
    #       environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, model, replay, environment_name, render=False,
        video_capture=False, demo=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        self.environment_name = environment_name
        self.render = render
        self.env = gym.make(self.environment_name)

        self.model = model
        self.replay = replay
        self.update_frequency = 100
        self.video_capture = video_capture

        if environment_name == 'MountainCar-v0':
            self.qn = QNetwork(model, environment_name, debug=False,
                learning_rate=1e-4, replay=replay, num_actions=2)
        else:
            self.qn = QNetwork(model, environment_name, debug=False,
                learning_rate=1e-4, replay=replay)


        # For deep and dueling with replay, 0.2 and 0.01 work fine for CartPole
        # For linear, the best I've been able to get is 1.0 and 0.1 with rewards
        # averaging in the mid-high 100s, e.g. 170-something on CartPole.
        self.epsilon_max = 1
        self.epsilon_min = 0.01

        self.reward_shaper = lambda reward, state, done:reward

        if environment_name == 'MountainCar-v0':
            self.discount_factor = 1.0
            self.training_iterations = 10000     # 3000-5000 iterations

            def reward_shaper(reward, state, done):
                position, velocity = state
                if done:
                    reward = 0
                return reward

            self.reward_shaper = reward_shaper

        else:   # For CartPole and Space Invaders
            self.discount_factor = 1.0
            # For CartPole, 1000000 iterations
            self.training_iterations = 1e6

        # Initialize replay memory - can pass different memory_size and burn_in
        # args if necessary.
        if demo == 'live':
            self.replay_memory = Replay_Memory()
            self.demo()
        elif demo == 'old':
            self.replay_memory = Replay_Memory()

            # Currently only have generated transitions for MountainCar
            transitions = pickle.load(open('mountaincar-transitions', 'rb'))
            print(transitions)
            for transition in transitions:
                self.replay_memory.append(transition)
        elif self.replay:
            self.replay_memory = Replay_Memory()
            self.burn_in_memory()


    def epsilon_greedy_policy(self, q_values, epsilon):
        # Compute max action and assign it a probability of 1-epsilon
        q_values = q_values.reshape((-1,)) # WARNING: we're flattening q_values
        action = np.argmax(q_values)

        def policy():
            res = action
            if np.random.rand() < epsilon:
                new_action = self.env.action_space.sample()
                # while (new_action == res):
                #     new_action = self.env.action_space.sample()
                res = new_action
            return res
        return policy

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        q_values = q_values.reshape((-1,)) # WARNING: flattening q_values here
        # Compute max action and assign it a probability of 1
        action = np.argmax(q_values)
        return (lambda :action)

    # sample_action:    Given a policy of tuples of (action, probability),
    #                   samples and returns an action to take according to the
    #                   policy.
    def sample_action(self, policy):
        res = policy()
        if self.environment_name == 'MountainCar-v0':
            res *= 2
        return policy()


    def train(self):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact
        # with the environment in this function, while also updating your
        # network parameters.

        # If you are using a replay memory, you should interact with environment
        # here, and store these transitions to memory, while also updating your
        # model.


        # Is there a way to get around having an env in DQN_Agent and QNetwork?
        env = gym.make(self.environment_name)
        if self.video_capture:
            env = gym.wrappers.Monitor(env, '.', force=True,
                video_callable=lambda iterations: iterations % 100 == 0)

        iterations = 0 # number of episodes

        updates = 0 # number of times we add to the memory (number of memories)
        rewards = []
        epsilon = self.epsilon_max
        decay_rate = (self.epsilon_max - self.epsilon_min)  \
            / self.training_iterations
        avg_total_reward = 0

        done = False
        prev_obs = None
        obs = reset(env, self.environment_name)

        print('calling train')

        loss = None
        t = 1.0

        training_rewards = []
        last_20_episode_rewards = deque([])
        cur_total_reward = 0

        while iterations < self.training_iterations:

            if self.render and iterations % 100 == 0 and not done:
                env.render()

            if done:
                iterations += 1
                obs = reset(env, self.environment_name)
                loss = None
                t = 1.0

                # last_20_episode_rewards is used for plotting training rewards
                if len(last_20_episode_rewards) == 20:
                    last_20_episode_rewards.popleft()
                    last_20_episode_rewards.append(cur_total_reward)
                    cur_total_reward = 0
                else:
                    last_20_episode_rewards.append(cur_total_reward)
                    cur_total_reward = 0


            epsilon = self.epsilon_max - decay_rate * iterations

            prev_obs = obs
            policy = self.epsilon_greedy_policy(self.qn.get_qvalues(prev_obs),
                epsilon)
            action = self.sample_action(policy)

            # Take action and train QNetwork
            # Do we need to discount reward here?
            obs, reward, done, info = step(env, action, self.environment_name)
            # Reward shaping didn't really help MountainCar that much :/
            reward = self.reward_shaper(reward, obs, done)
            cur_total_reward += reward
            if done == True and cur_total_reward > -199:
                print(cur_total_reward)


            if self.replay:
                if done == True and t < 199:
                    self.replay_memory.append((prev_obs, action, reward, done, obs), weight=1000)
                else:
                    self.replay_memory.append((prev_obs, action, reward, done, obs))
                if updates % self.update_frequency == 0:
                    # Using np arrays here, tensorflow supports batch updates
                    # in this style
                    batch = self.replay_memory.sample_batch(batch_size=200)
                    prev_obs_vector = np.array([value[0] for value in batch])
                    action_vector = np.array([value[1] for value in batch])
                    reward_vector = np.array([value[2] for value in batch])
                    done_vector = np.array([value[3] for value in batch])
                    obs_vector = np.array([value[4] for value in batch])

                    cur_loss, cur_pred, cur_exp = \
                        self.qn.update_params(prev_obs_vector, action_vector,
                                              reward_vector, obs_vector,
                                              done_vector, self.discount_factor)
                    loss = np.average(cur_loss)

            else:
                cur_loss, cur_pred, cur_exp = \
                    self.qn.update_params(prev_obs, np.array([action]),
                        np.array([reward]), obs, np.array([done]),
                        self.discount_factor)

                if t == 1.0:
                    loss = cur_loss
                else:
                    loss = (cur_loss + loss * (t - 1)) / t
                t += 1.0

            updates += 1


            if iterations % 200 == 0 and done:
                cur_reward = self.test()
                print(cur_reward, loss, epsilon, iterations)
                rewards.append(cur_reward)

                if len(last_20_episode_rewards) > 0:
                    training_rewards.append(sum(last_20_episode_rewards) /
                        len(last_20_episode_rewards))
                else:
                    if self.environment_name == 'MountainCar-v0':
                        training_rewards.append(-200)
                    elif self.environment_name == 'CartPole-v0':
                        training_rewards.append(0)

                self.qn.save_model_weights("network")

                '''
                print('training rewards:', file=sys.stderr)
                for reward in training_rewards:
                    print(reward, file=sys.stderr)

                print('test rewards:', file=sys.stderr)
                for reward in rewards:
                    print(reward, file=sys.stderr)
                '''

                plt.clf()
                training_line = plt.plot([100*i for i in
                    range(len(training_rewards))], training_rewards, aa=True,
                    label='Training rewards')
                test_line = plt.plot([100*i for i in range(len(rewards))],
                    rewards, aa=True, label='Test rewards')
                plt.legend()
                if self.environment_name == 'MountainCar-v0':
                    plt.axis([0, 100*len(rewards)+1, -250, 0])
                elif self.environment_name == 'CartPole-v0':
                    plt.axis([0, 100*len(rewards)+1, 0, 250])

                plt.xlabel('Weight updates (training iterations)')
                plt.ylabel('Average reward per 20 episodes')
                plt.savefig('%s.png' % self.model)
                plt.savefig('%s.pdf' % self.model)


                print(self.test(num_iterations=100))


        return rewards

    def test(self, model_file=None, video_capture = False, num_iterations=20,
        epsilon_greedy=False):
        # Evaluate the performance of your agent over 100 episodes, by
        # calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of
        # whether you are using a memory.
        env = self.env

        if model_file is not None:
            self.qn.load_model(environment_name)

        if video_capture:
            env = gym.wrappers.Monitor(env, '.', force=True)

        iterations = 0

        cum_reward = 0
        rewards = []
        while iterations < num_iterations:
            done = False
            obs = reset(env, self.environment_name)

            current_reward = 0

            while not done:
                # if self.render:
                #     env.render()

                if epsilon_greedy:
                    policy =    \
                        self.epsilon_greedy_policy(self.qn.get_qvalues(obs),
                        0.05)
                else:
                    policy = self.greedy_policy(self.qn.get_qvalues(obs))

                # Sample action from current policy
                action_to_take = self.sample_action(policy)

                # Take action and record reward
                if obs is not None:
                    prev_obs = obs
                # TODO: Do we need to discount reward here?

                obs, reward, done, info = step(env, action_to_take, self.environment_name)

                current_reward += reward
                cum_reward += reward

            rewards.append(current_reward)

            iterations += 1

        rewards_arr = np.array(rewards)
        print('mean: %f std: %f' % (np.mean(rewards_arr, axis=0),
            np.std(rewards_arr, axis=0)))

        return cum_reward / num_iterations

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes /
        # transitions.

        env = self.env
        print('commencing burn_in')

        done = False
        obs = reset(env, self.environment_name)
        prev_obs = None

        for i in range(self.replay_memory.burn_in):
            # We want to initialize memory with on-policy experience from our
            # just-initialized QNetwork

            if done:
                obs = reset(env, self.environment_name)


            # Should we use an epsilon-greedy policy?
            #policy = self.epsilon_greedy_policy(self.qn.get_qvalues(obs), 0.05)
            policy = self.epsilon_greedy_policy(self.qn.get_qvalues(obs),
                self.epsilon_max)

            # Sample action from current policy
            action_to_take = self.sample_action(policy)
            # Take action and record reward
            prev_obs = obs
            # Do we need to discount reward here?
            obs, reward, done, info = step(env, action_to_take, self.environment_name)
            reward = self.reward_shaper(reward, obs, done)
            self.replay_memory.append((prev_obs, action_to_take, reward, done,
                obs))
        print('burn in complete')

    # Demo function for playing the game yourself to initialize the burn-in
    # memory i.e. behavioral cloning
    def demo(self):
        env = gym.make(self.environment_name)
        while True:
            done = False
            prev_obs = None
            obs = env.reset()
            total_reward = 0
            while not done:
                env.render()
                action = None
                while action is None:
                    key_pressed = getch.getch()

                    if self.environment_name == 'MountainCar-v0':
                        if key_pressed == 'a':
                            action = 0
                        elif key_pressed == 'd':
                            action = 2
                        elif key_pressed == 'q':
                            break
                    if self.environment_name == 'CartPole-v0':
                        if key_pressed == 'a':
                            action = 0
                        elif key_pressed == 'd':
                            action = 1
                        elif key_pressed == 'q':
                            break
                if key_pressed == 'q':
                    break
                prev_obs = obs
                obs, reward, done, info = step(env, action, self.environment_name)
                self.replay_memory.append((prev_obs, action, reward, done, obs))
                total_reward += reward
            print('Your reward was: %d' % total_reward)
            if key_pressed == 'q':
                break





def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument   \
        Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    #parser.add_argument('--train',dest='train',type=int,default=1)
    #parser.add_argument('--model-file',dest='model_file',type=str)
    parser.add_argument('--model',dest='model',type=str, default='linear')
    parser.add_argument('--replay',dest='replay', action='store_true')
    parser.add_argument('--video_capture',dest='video_capture',
        action='store_true')
    parser.add_argument('--demo',dest='demo')
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env
    render = args.render
    model = args.model
    replay = args.replay
    video_capture = args.video_capture
    demo = args.demo

    # Setting the session to allow growth, so it doesn't allocate all GPU memory
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    # keras.backend.tensorflow_backend.set_session(sess)

    # You want to create an instance of the DQN_Agent class here, and then train
    # / test it.
    dqn = DQN_Agent(model, replay, environment_name, render, video_capture, demo)
    #dqn.test(num_iterations=1, video_capture=True, model_file='network')



    eval_rewards = dqn.train()

    '''
    print('average total reward per 100 episodes: %f' %
        dqn.test(num_iterations=100))
    print('capturing video')
    dqn.test(num_iterations=1, video_capture=True)
    '''

    # training_line = plt.plot([100*i for i in range(len(eval_rewards))],
    #     eval_rewards, aa=False)
    # plt.axis([0, 100*len(eval_rewards)+1, 0, 250])
    # plt.xlabel('weight update (iterations)')
    # plt.ylabel('average reward per 20 episodes')
    # plt.savefig('%s.png' % model)
    # plt.savefig('%s.pdf' % model)
    # plt.show()

    cum_reward = dqn.test()
    print(cum_reward)

    # TODO: print metrics, make plots, video capture, make tables, etc.

if __name__ == '__main__':
    main(sys.argv)
