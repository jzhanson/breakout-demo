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

def tf_shape_to_np_shape(shape):
    return [-1 if x.value is None else x.value for x in shape]

class QNetwork():

    def __init__(self, obs_shape, num_actions, debug=False,
        learning_rate=1e-4):

        self.learning_rate = learning_rate
        self.sess = tf.Session()

        # By passing observation_space and num_actions, we don't have to have
        # an gym env in this class too.
        #self.env = gym.make(environment_name)

        self.obs_shape = obs_shape
        self.num_actions = num_actions


        self.create_model()
        self.sess.run(tf.global_variables_initializer())


    # create_model: Initializes a tensorflow model
    def create_model(self):
        with self.sess:
            '''
            self.state_ph = tf.placeholder(dtype=tf.float32,
                shape=self.obs_shape)
            '''
            self.state_ph = tf.placeholder(dtype=tf.float32,
                shape=self.obs_shape)

            self.action_ph = tf.placeholder(shape=[None], dtype=tf.int32)
            self.expected_ph = tf.placeholder(tf.float32, shape=[None])

            # Convolution layers go here
            conv1 = tf.layers.conv2d(
                inputs=self.state_ph,
                filters=32,
                kernel_size=[8,8],
                strides=4,
                padding="same",
                activation=tf.nn.relu
            )
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=64,
                kernel_size=[4,4],
                strides=2,
                padding="same",
                activation=tf.nn.relu
            )
            conv3 = tf.layers.conv2d(
                inputs=conv2,
                filters=64,
                kernel_size=[3,3],
                strides=1,
                padding="same",
                activation=tf.nn.relu
            )

            fully_connected = tf.contrib.layers.fully_connected(conv3,
                512, activation_fn=tf.nn.relu)

            self.qvalue_logits = tf.contrib.layers.fully_connected(
                fully_connected,
                self.num_actions,
                activation_fn = None,
            )

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


# If necessary, will implement prioritized replay
class Replay_Memory():

    def __init__(self, memory_size=1000000, burn_in=50000):
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
            (prev_obs, action, reward, done, obs) = self.memory.popleft()
            self.current_size -= 1

        self.memory.append(transition)
        self.current_size += 1



class DQN_Agent():
    # We use the following three functions to wrap the last 3 frames in with the
    # current frame - elsewhere, the state is treated as an opaque thing.
    def preprocess(self, frame):
        # grayscaling
        image = frame
        # Other option is 0.2126, 0.7152, 0.0722
        gray_image = image[...,:3].dot([0.299, 0.587, 0.114])
        # magic numbers that people use to reduce images grayscale

        # Resizing to 110x84 since Tensorflow doesn't care about square images
        # Could consider downsampling in a more intelligent way
        gray_small_image = resize(gray_image, (110, 84))

        # Include last 3 frames as well, in the order earliest to latest

        obs = np.append(np.append(np.append(self.last_three_frames[0],
            self.last_three_frames[1], axis=0), self.last_three_frames[2],
            axis=0), gray_small_image, axis=0)

        # Drop oldest frame and add newest
        self.last_three_frames[1:].append(gray_small_image)

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



    def __init__(self, environment_name, render=False, video_capture=False):

        self.environment_name = environment_name
        self.render = render
        self.env = gym.make(self.environment_name)

        self.update_frequency = 100
        self.video_capture = video_capture

        # batch, in_height, in_width, in_channels
        self.qn = QNetwork((None, 110, 84, 4),
            self.env.action_space.n, learning_rate=1e-4)

        # Annealed linearly from 1 to 0.1 over the first million frames then
        # fixed afterwards
        self.epsilon_max = 1
        self.epsilon_min = 0.1

        self.discount_factor = 0.99
        self.training_iterations = 1e7  # 10,000,000
        # We initialize last_three_frames to be all zeros 110x84 arrays at
        # first, which will end up making relatively little difference in the
        # long run. Can use uint8 to save space but we'll just use default float
        # for now
        self.last_three_frames = [np.zeros((110, 84), dtype=np.uint8),
            np.zeros((110, 84), dtype=np.uint8), np.zeros((110, 84),
            dtype=np.uint8)]

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
        return policy()


    def train(self):
        if self.video_capture:
            self.env = gym.wrappers.Monitor(self.env, '.', force=True,
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
        obs = self.reset()

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
                obs = self.reset()
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
            obs, reward, done, info = self.step(action)

            cur_total_reward += reward
            if done:
                print(cur_total_reward)

            # Do we just append to our replay memory and not train on the
            # transition that just happened?
            '''
            cur_loss, cur_pred, cur_exp = \
                self.qn.update_params(prev_obs, np.array([action]),
                    np.array([reward]), obs, np.array([done]),
                    self.discount_factor)
            '''

            self.replay_memory.append((prev_obs, action, reward, done, obs))
            if updates % self.update_frequency == 0:
                # Using np arrays here, tensorflow supports batch updates
                # in this style
                batch = self.replay_memory.sample_batch(batch_size=32)
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


                if t == 1.0:
                    loss = cur_loss
                else:
                    loss = (cur_loss + loss * (t - 1)) / t
                t += 1.0

            updates += 1


            if iterations % 200 == 0 and done:
                cur_reward = self.test()
                print(cur_reward, loss, epsilon, iterations)
                rewards.a1ppend(cur_reward)

                if len(last_20_episode_rewards) > 0:
                    training_rewards.append(sum(last_20_episode_rewards) /
                        len(last_20_episode_rewards))
                else:
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
        if model_file is not None:
            self.qn.load_model(environment_name)

        if video_capture:
            self.env = gym.wrappers.Monitor(self.env, '.', force=True)

        iterations = 0

        cum_reward = 0
        rewards = []
        while iterations < num_iterations:
            done = False
            obs = self.reset()

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
                # TODO: Do we need to discount reward here?

                obs, reward, done, info = self.step(action_to_take)

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

        print('commencing burn_in')

        done = False
        obs = self.reset()
        prev_obs = None

        for i in range(self.replay_memory.burn_in):
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
            # Take action and record reward
            prev_obs = obs
            # Do we need to discount reward here?
            obs, reward, done, info = self.step(action_to_take)
            self.replay_memory.append((prev_obs, action_to_take, reward, done,
                obs))
        print('burn in complete')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument   \
        Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    #parser.add_argument('--train',dest='train',type=int,default=1)
    #parser.add_argument('--model-file',dest='model_file',type=str)
    #parser.add_argument('--model',dest='model',type=str, default='linear')
    #parser.add_argument('--replay',dest='replay', action='store_true')
    parser.add_argument('--video_capture',dest='video_capture',
        action='store_true')
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env
    render = args.render
    #model = args.model
    #replay = args.replay
    video_capture = args.video_capture

    # Setting the session to allow growth, so it doesn't allocate all GPU memory
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    # keras.backend.tensorflow_backend.set_session(sess)

    dqn = DQN_Agent(environment_name, render, video_capture)
    #dqn.test(num_iterations=1, video_capture=True, model_file='network')

    eval_rewards = dqn.train()

    cum_reward = dqn.test()
    print(cum_reward)

if __name__ == '__main__':
    main(sys.argv)
