#!/usr/bin/env python
import tensorflow as tf
import numpy as np
#import scipy
from skimage.transform import resize
import gym
import sys
import copy
import argparse
import random
import matplotlib.pyplot as plt
from collections import deque

#import time



def tf_shape_to_np_shape(shape):
    return [-1 if x.value is None else x.value for x in shape]

class QNetwork():

    def __init__(self, sess, obs_shape, num_actions, debug=False,
        learning_rate=1e-4, model_file=None):

        self.learning_rate = learning_rate
        self.sess = sess

        # By passing observation_space and num_actions, we don't have to have
        # an gym env in this class too.
        #self.env = gym.make(environment_name)

        self.obs_shape = obs_shape
        self.num_actions = num_actions

        self.create_model()
        if model_file is not None:
            self.load_model()
        else:
            self.sess.run(tf.global_variables_initializer())


    # create_model: Initializes a tensorflow model
    def create_model(self):
        with self.sess:
            with tf.variable_scope('online'):
                self.state_ph = tf.placeholder(dtype=tf.float32,
                    shape=self.obs_shape)

                self.expected_ph = tf.placeholder(dtype=tf.float32, shape=[None])

                self.action_ph = tf.placeholder(dtype=tf.uint8, shape=[None])

                self.online_conv1 = tf.layers.conv2d(
                        inputs=self.state_ph,
                        filters=32,
                        kernel_size=[8,8],
                        strides=4,
                        padding="same",
                        activation=tf.nn.relu
                )
                self.online_conv2 = tf.layers.conv2d(
                        inputs=self.online_conv1,
                        filters=64,
                        kernel_size=[4,4],
                        strides=2,
                        padding="same",
                        activation=tf.nn.relu
                )
                self.online_conv3 = tf.layers.conv2d(
                        inputs=self.online_conv2,
                        filters=64,
                        kernel_size=[3,3],
                        strides=1,
                        padding="same",
                        activation=tf.nn.relu
                )

                self.online_conv3_flat = tf.reshape(self.online_conv3, [-1,
                    np.prod(np.array(self.online_conv3.shape[1:]))])

                self.online_fully_connected = \
                    tf.contrib.layers.fully_connected(self.online_conv3_flat,
                    512, activation_fn=tf.nn.relu)

                self.qvalue_logits = tf.contrib.layers.fully_connected(
                        self.online_fully_connected,
                        self.num_actions,
                        activation_fn = None,
                )
                tf.summary.histogram('qvalue_logits', self.qvalue_logits)

                self.action = tf.argmax(input=self.qvalue_logits, axis=1)

                # TODO: does same thing as tf.gather - can simplify?
                self.prediction = tf.reduce_sum(
                    tf.multiply(
                        self.qvalue_logits,
                        tf.one_hot(self.action_ph, self.num_actions),
                    ),
                    axis=1,
                )
                self.loss = tf.reduce_mean(tf.square(self.expected_ph -
                self.prediction))
                tf.summary.scalar('loss', self.loss)
                self.optimizer =    \
                tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                self.train_op = self.optimizer.minimize(loss=self.loss)

                self.file_writer = tf.summary.FileWriter('./logs', self.sess.graph)
                self.merged = tf.summary.merge_all()

            with tf.variable_scope('target'):
                self.target_conv1 = tf.layers.conv2d(
                        inputs=self.state_ph,
                        filters=32,
                        kernel_size=[8,8],
                        strides=4,
                        padding="same",
                        activation=tf.nn.relu
                )
                self.target_conv2 = tf.layers.conv2d(
                        inputs=self.target_conv1,
                        filters=64,
                        kernel_size=[4,4],
                        strides=2,
                        padding="same",
                        activation=tf.nn.relu
                )
                self.target_conv3 = tf.layers.conv2d(
                        inputs=self.target_conv2,
                        filters=64,
                        kernel_size=[3,3],
                        strides=1,
                        padding="same",
                        activation=tf.nn.relu
                )

                self.target_conv3_flat = tf.reshape(self.target_conv3, [-1,
                    np.prod(np.array(self.target_conv3.shape[1:]))])

                self.target_fully_connected =   \
                    tf.contrib.layers.fully_connected(self.target_conv3_flat,
                    512, activation_fn=tf.nn.relu)

                self.target_qvalue_logits = tf.contrib.layers.fully_connected(
                    self.target_fully_connected,
                    self.num_actions,
                    activation_fn = None,
                )


    def update_params(self, prev_obs, action, reward, obs, done,
        discount_factor):
        q_primes = self.get_qvalues(obs)
        max_action = np.argmax(q_primes, axis=1)


        target_q_primes = self.get_target_qvalues(obs)

        done = np.vectorize(lambda b: 0 if b else 1)(done)
        discount_factor = done * discount_factor

        targets = np.array([target_q_primes[i][max_action[i]] for i in
            range(len(target_q_primes))])

        feed_dict = {
            self.state_ph: #state_ph is the placeholder value for the state
                prev_obs.reshape((tf_shape_to_np_shape(self.state_ph.shape))),
            self.action_ph: action,
            self.expected_ph: reward + (discount_factor
                * targets)
        }

        summary, _,loss, pred, exp = \
            self.sess.run([self.merged, self.train_op, self.loss, self.prediction,
                self.expected_ph],feed_dict=feed_dict)
        self.file_writer.add_summary(summary)

        return loss, pred, exp

    def sync_params(self):
        print('syncing params')
        online_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'online')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target')

        for i in range(len(online_vars)):
            target_vars[i].assign(online_vars[i]).eval(session=self.sess)
            '''
            time.sleep(2)
            print(online_vars[i])
            print(target_vars[i])
            '''



    def get_qvalues(self, obs):
        # return's the qvalue for a given state
        return self.sess.run(
            self.qvalue_logits,
            feed_dict={self.state_ph:
                obs.reshape((tf_shape_to_np_shape(self.state_ph.shape)))},)

    def get_target_qvalues(self, obs):
        return self.sess.run(
            self.target_qvalue_logits,
            feed_dict={self.state_ph:
                obs.reshape((tf_shape_to_np_shape(self.state_ph.shape)))},)


    def save_model(self):
        # Helper function to save your model / weights.
        print('saving model')
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "./double_model.ckpt")


    def load_model(self):
        # Helper function to load an existing model.
        print('restoring model')
        saver = tf.train.Saver()
        saver.restore(self.sess, "./double_model.ckpt")




# If necessary, will implement prioritized replay
class Replay_Memory():

    # About 0.0003 gb per transition, probably want 50000 for ~15 gb
    def __init__(self, memory_size=100000, burn_in=5000):
        self.memory = deque([])
        self.current_size = 0
        self.memory_size = memory_size
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # Samples without replacement
        idx = np.random.choice(
            range(len(self.memory)),
            batch_size,
            replace=False
            #p=(np.array(self.weights) / self.weights_sum).reshape([-1])
        )
        return np.array([self.memory[v] for v in idx])

    def append(self, transition):
        # Appends transition to the memory.

        # The "left" side of the deque is first-in (least recently added) and
        # the "right" side is last-in (most recently added)
        if self.current_size == self.memory_size:
            # Discard oldest transition
            self.memory.popleft()
            self.current_size -= 1

        self.memory.append(transition)
        self.current_size += 1



class DQN_Agent():

    def __init__(self, sess, environment_name, render=False,
        video_capture=False, model_file=None, elapsed_episodes=None,
        elapsed_updates=None):

        self.environment_name = environment_name
        self.render = render
        self.env = gym.make(self.environment_name)

        self.video_capture = video_capture

        self.elapsed_episodes = elapsed_episodes
        self.elapsed_updates = elapsed_updates


        # batch, in_height, in_width, in_channels
        self.qn = QNetwork(sess, (None, 110, 84, 4),
            self.env.action_space.n, learning_rate=1e-4, model_file=model_file)


        # Annealed linearly from 0.2 to 0.01 over the first million frames then
        # fixed afterwards
        self.epsilon_max = 1
        self.epsilon_min = 0.1
        self.decay_iterations = 1e6

        self.tau = 10000

        self.discount_factor = 0.99
        self.training_iterations = 5e7  # 50,000,000
        # We initialize last_three_frames to be all zeros 110x84 arrays at
        # first, which will end up making relatively little difference in the
        # long run. Can use uint8 to save space but we'll just use default float
        # for now
        self.last_three_frames = deque([np.zeros((110, 84), dtype=np.uint8),
            np.zeros((110, 84), dtype=np.uint8), np.zeros((110, 84),
            dtype=np.uint8)])

        self.replay_memory = Replay_Memory()
        self.burn_in_memory()

    # We use the following three functions to wrap the last 3 frames in with the
    # current frame - elsewhere, the state is treated as an opaque thing.
    def preprocess(self, frame):
        # grayscaling
        image = frame
        # 0.299, 0.587, 0.114
        # Other option is 0.2126, 0.7152, 0.0722
        gray_image = image[...,:3].dot([0.2126, 0.7152, 0.0722])
        # magic numbers that people use to reduce images grayscale

        # Resizing to 110x84 since Tensorflow doesn't care about square images
        # Could consider downsampling in a more intelligent way
        #
        # Could add in anti-aliasing=true but need newest scikit-image
        gray_small_image = resize(gray_image, (110, 84))

        # Include last 3 frames as well, in the order earliest to latest

        # This should help save on memory
        gray_integer_small_image = gray_small_image.astype(np.uint8)

        obs = np.append(np.append(np.append(self.last_three_frames[0],
            self.last_three_frames[1], axis=0), self.last_three_frames[2],
            axis=0), gray_small_image, axis=0)

        # Drop oldest frame and add newest
        self.last_three_frames.popleft()
        self.last_three_frames.append(gray_small_image)

        return obs

    # step function we use as a wrapper for env.step
    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        obs = self.preprocess(frame)

        return (obs, reward, done, info)

    # reset function we use as a wrapper for env.reset
    def reset(self):
        obs = self.env.reset()
        return self.preprocess(obs)


    def epsilon_greedy_policy(self, q_values, epsilon):
        # Compute max action and assign it a probability of 1-epsilon
        q_values = q_values.reshape((-1,)) # WARNING: we're flattening q_values
        action = np.argmax(q_values)

        def policy():
            res = action
            if np.random.rand() < epsilon:
                new_action = self.env.action_space.sample()
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
        return policy()


    # How about instead of taking a step and then training on a whole minibatch
    # we run an episode, then sample and train on a minibatch?
    def train(self):
        if self.video_capture:
            self.env = gym.wrappers.Monitor(self.env, './training_videos',
                force=True, video_callable=lambda episodes: episodes % 10
                == 0)

        if self.elapsed_episodes is not None and self.elapsed_updates is not  \
            None:
            episodes = self.elapsed_episodes
            updates = self.elapsed_updates
        else:
            episodes = 0
            updates = 0

        epsilon = self.epsilon_max
        decay_rate = (self.epsilon_max - self.epsilon_min)  \
            / self.decay_iterations
        avg_total_reward = 0

        done = False
        prev_obs = None
        obs = self.reset()

        print('calling train')

        loss = None
        t = 1.0
        current_updates = 0

        training_rewards = []
        test_rewards = []
        last_20_episode_rewards = deque([])
        cur_total_reward = 0

        while updates < self.training_iterations:

            if self.render:
                self.env.render()

            # Copy parameters to target network
            if updates % self.tau == 0:
                self.qn.sync_params()

            if updates <= self.decay_iterations:
                epsilon = self.epsilon_max - decay_rate * updates
            else:
                epsilon = self.epsilon_min

            prev_obs = obs
            policy = self.epsilon_greedy_policy(self.qn.get_qvalues(prev_obs),
                epsilon)
            action = self.sample_action(policy)

            # Take action and train QNetwork
            obs, reward, done, info = self.step(action)

            cur_total_reward += reward

            # We just append to our replay memory and do not train on the
            # transition that just happened
            '''
            cur_loss, cur_pred, cur_exp = \
                self.qn.update_params(prev_obs, np.array([action]),
                    np.array([reward]), obs, np.array([done]),
                    self.discount_factor)
            '''

            self.replay_memory.append((prev_obs, action, reward, done, obs))


            updates += 1
            current_updates += 1

            # Train every 4 steps
            if updates % 4 == 0:
                # Using np arrays here, tensorflow supports batch updates
                # in this style
                batch = self.replay_memory.sample_batch(batch_size=32)
                prev_obs_vector = np.array([value[0] for value in batch])
                action_vector = np.array([value[1] for value in batch])
                reward_vector = np.array([value[2] for value in batch])
                done_vector = np.array([value[3] for value in batch])
                obs_vector = np.array([value[4] for value in batch])

                # prev_obs and obs are np.uint8, we want to be tf.float32
                cur_loss, cur_pred, cur_exp = \
                    self.qn.update_params(prev_obs_vector.astype(np.float32), action_vector,
                                          reward_vector, obs_vector.astype(np.float32),
                                          done_vector, self.discount_factor)
                loss = np.average(cur_loss)


            if done:
                # Should be 1M per evaluation, and keeping the best policy
                if episodes % 10 == 0:
                    cur_reward = self.test()
                    print('test rewards: %d' % cur_reward)
                    test_rewards.append(cur_reward)

                # last_20_episode_rewards is used for plotting training rewards
                if len(last_20_episode_rewards) == 20:
                    last_20_episode_rewards.popleft()
                    last_20_episode_rewards.append(cur_total_reward)
                else:
                    last_20_episode_rewards.append(cur_total_reward)


                if len(last_20_episode_rewards) > 0:
                    training_rewards.append(sum(last_20_episode_rewards) /
                        len(last_20_episode_rewards))
                else:
                    training_rewards.append(0)


                episodes += 1

                print('training reward: %d loss: %f epsilon: %f episodes: %d' %
                    (cur_total_reward, loss, epsilon, episodes))
                print('current updates: %d total updates %d' % (current_updates,
                    updates))


                obs = self.reset()
                loss = None
                current_updates = 0
                cur_total_reward = 0

                plt.clf()
                training_line = plt.plot([i for i in
                    range(len(training_rewards))], training_rewards, aa=True,
                    label='Training rewards')
                test_line = plt.plot([i*10 for i in range(len(test_rewards))],
                    test_rewards, aa=True, label='Test rewards')
                plt.legend()
                plt.axis([0, len(training_rewards)+1, 0, 20])

                plt.xlabel('Episodes (training iterations)')
                plt.ylabel('Average reward per 1 episode')
                plt.savefig('breakout.png')
                plt.savefig('breakout.pdf')

                if episodes % 10 == 0:
                    self.qn.save_model()


            '''
            print('training episode: %d total updates: %d current updates: %d'
                % (episodes, updates, current_updates))
            '''

        return test_rewards

    def test(self, model_file=None, video_capture=False, num_iterations=20,
        epsilon_greedy=False):
        if model_file is not None:
            self.qn.load_model()

        if video_capture:
            self.env = gym.wrappers.Monitor(self.env, './test_videos', force=True)


        episodes = 0
        iterations = 0

        cum_reward = 0
        rewards = []
        while episodes < num_iterations:
            print('testing episode: %d' % episodes)
            done = False
            obs = self.reset()

            current_reward = 0
            current_iterations = 0

            while not done:
                if self.render:
                    env.render()
                #print('test episodes: %d total iterations: %d current iterations: %d' % (episodes, iterations, current_iterations))

                '''
                if epsilon_greedy:
                    policy =    \
                        self.epsilon_greedy_policy(self.qn.get_qvalues(obs),
                        0.05)
                else:
                    policy = self.greedy_policy(self.qn.get_qvalues(obs))
                '''
                policy = self.epsilon_greedy_policy(self.qn.get_qvalues(obs),
                        0.05)


                # Sample action from current policy
                action_to_take = self.sample_action(policy)

                # Take action and record reward
                obs, reward, done, info = self.step(action_to_take)

                current_reward += reward
                cum_reward += reward
                current_iterations += 1
                iterations += 1

            rewards.append(current_reward)
            print('reward: %d' % current_reward)

            episodes += 1

        rewards_arr = np.array(rewards)
        print('mean: %f std: %f' % (np.mean(rewards_arr, axis=0),
            np.std(rewards_arr, axis=0)))

        return cum_reward / num_iterations

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes /
        # transitions.

        print('commencing burn_in')

        done = False
        obs = self.reset()
        prev_obs = None

        if self.render:
            self.env.render()

        for i in range(self.replay_memory.burn_in):
            if i % 100 == 0:
                print('burn in transition: %d' % i)
            # We want to initialize memory with on-policy experience from our
            # just-initialized QNetwork

            if done:
                obs = self.reset()

            # Should we use an epsilon-greedy policy?
            #policy = self.epsilon_greedy_policy(self.qn.get_qvalues(obs), 0.05)
            policy = self.epsilon_greedy_policy(self.qn.get_qvalues(obs),
                self.epsilon_max)

            # Sample action from current policy
            action_to_take = self.sample_action(policy)
            prev_obs = obs
            # Take action and record reward
            obs, reward, done, info = self.step(action_to_take)
            self.replay_memory.append((prev_obs, action_to_take, reward, done,
                obs))
        print('burn in complete')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument   \
        Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',action='store_true')
    #parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model_file',dest='model_file',type=str)

    #parser.add_argument('--model',dest='model',type=str, default='linear')
    #parser.add_argument('--replay',dest='replay', action='store_true')
    parser.add_argument('--video_capture',dest='video_capture',
        action='store_true')
    parser.add_argument('--episodes',dest='elapsed_episodes',type=int,default=0)
    parser.add_argument('--updates', dest='elapsed_updates',type=int,default=0)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env
    render = args.render
    model_file = args.model_file
    #replay = args.replay
    video_capture = args.video_capture

    elapsed_episodes = args.elapsed_episodes
    elapsed_updates = args.elapsed_updates

    # Setting the session to allow growth, so it doesn't allocate all GPU memory
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)


    # Setting this as the default tensorflow session.
    # keras.backend.tensorflow_backend.set_session(sess)

    dqn = DQN_Agent(sess, environment_name, render, video_capture,
        model_file, elapsed_episodes, elapsed_updates)

    eval_rewards = dqn.train()

    avg_test_reward = dqn.test()
    print(cum_reward)

if __name__ == '__main__':
    main(sys.argv)
