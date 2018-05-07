import os
import sys
import argparse
from skimage.transform import resize
import numpy as np
from collections import deque

'''
import matplotlib
matplotlib.use('Agg')
'''
import matplotlib.pyplot as plt
import tensorflow as tf

import gym

import time

gpu_ops = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)

# tf.set_random_seed(seed)

env = gym.make('BreakoutNoFrameskip-v4')
env = gym.wrappers.Monitor(env, './videos_a2c', force=True,
        video_callable=lambda episode_id: episode_id%100==0)

# Copied from https://github.com/MG2033/A2C/blob/master/layers.py
def openai_entropy(logits):
    # Entropy proposed by OpenAI in their A2C baseline
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

class A2C:
    # Initialize model, variables
    def __init__(self, lr, gamma, n):
        self.lr = lr
        self.n = n
        self.gamma = gamma
        self.num_actions = env.action_space.n

        self.make_model()

    def preprocess(self, frame):
        # grayscaling
        # 0.299, 0.587, 0.114
        # Other option is 0.2126, 0.7152, 0.0722

        gray_image = np.array(frame)[...,:3].dot([0.2126, 0.7152, 0.0722])

        # magic numbers that people use to reduce images grayscale

        # Resizing to 110x84 since Tensorflow doesn't care about square images
        # Could consider downsampling in a more intelligent way
        #
        # Could add in anti-aliasing=true but need newest scikit-image
        gray_small_image = resize(gray_image, (110, 84))

        return gray_small_image


    def make_model(self):
        self.input_tensor = tf.placeholder(tf.float32, shape=(None, 110, 84))

        self.R_tensor = tf.placeholder(tf.float32, shape=[None])
        self.A_tensor = tf.placeholder(tf.int32, shape=[None])
        # We have advantage_tensor be separate in order so the gradients for
        # the actor don't go through the critic
        self.advantage_tensor = tf.placeholder(tf.float32, shape=[None])

        self.global_step = tf.Variable(0, trainable=False)


        with tf.variable_scope("policy"):
            conv1 = tf.layers.conv2d(
                    inputs=tf.expand_dims((tf.cast(self.input_tensor, tf.float32) / 255), -1),
                    filters=32,
                    kernel_size=[8,8],
                    strides=4,
                    padding='valid',    # 'same'?
                    activation=tf.nn.relu,
                    kernel_initializer=tf.initializers.orthogonal()
            )
            conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=64,
                    kernel_size=[4,4],
                    strides=2,
                    padding='valid',    # 'same'?
                    activation=tf.nn.relu,
                    kernel_initializer=tf.initializers.orthogonal()
            )
            conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=64,
                    kernel_size=[3,3],
                    strides=1,
                    padding='valid',    # 'same'?
                    activation=tf.nn.relu,
                    kernel_initializer=tf.initializers.orthogonal()
            )

            conv3_flat = tf.contrib.layers.flatten(conv3)

            fully_connected = tf.contrib.layers.fully_connected(
                    conv3_flat,
                    512,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.initializers.orthogonal()

            )

            self.actor_policy_logits = tf.contrib.layers.fully_connected(
                    fully_connected,
                    env.action_space.n,
                    activation_fn=None,
                    weights_initializer=tf.initializers.orthogonal()
            )

            # Add a little noise to encourage exploration!
            # self.actor_output_three = tf.nn.softmax(self.actor_policy_logits - tf.log(-tf.log(noise)))
            # self.actor_output_tesnor = tf.nn.softmax(self.actor_policy_logits + noise)
            noise = tf.random_uniform(tf.shape(self.actor_policy_logits))
            self.actor_output_tensor = tf.nn.softmax(self.actor_policy_logits - tf.log(-tf.log(noise)), 1)


            self.critic_output_tensor = tf.contrib.layers.fully_connected(
                    fully_connected,
                    1,
                    activation_fn=None,
                    weights_initializer=tf.initializers.orthogonal()
            )

        neg_log_action_probabilities = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.actor_policy_logits,
            labels=self.A_tensor
        )

        # sparse_softmax_cross_entropy is already negative, so don't need - here
        self.actor_loss = tf.reduce_mean(self.advantage_tensor *    \
            neg_log_action_probabilities)
        self.critic_loss = tf.reduce_mean(tf.square(self.R_tensor -     \
            tf.squeeze(self.critic_output_tensor)) / 2)
        # Maybe this is the missing ingredient?
        self.entropy = tf.reduce_mean(openai_entropy(self.actor_policy_logits))
        self.loss = self.actor_loss + 0.5 * self.critic_loss - 0.01 * self.entropy

        with tf.variable_scope("policy"):
            params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        grads = list(zip(grads, params))

        '''
        learning_rate = tf.train.exponential_decay(self.lr, self.global_step,
                                                    100000, 0.96, staircase=True)

        self.train_both = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
            decay=0.99, epsilon=1e-5)
        self.train_both = self.train_both.apply_gradients(grads,
            global_step=self.global_step)
        '''
        self.train_both = tf.train.RMSPropOptimizer(self.lr,
            decay=0.99, epsilon=1e-5)
        self.train_both = self.train_both.apply_gradients(grads)


        self.saver = tf.train.Saver(max_to_keep=5)
        sess.run(tf.global_variables_initializer())


    # Performs one episode of training in iterations of n steps each
    def train(self, render=False):
        obs = self.preprocess(env.reset())
        if render:
            env.render()
        done = False
        cur_steps = 0
        total_steps = 0
        total_reward = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        iterations = 0

        while not done:
            states = []
            actions = []
            rewards = []
            dones = []


            while cur_steps < self.n and not done:
                if render:
                    env.render()
                states.append(obs)
                # Predict takes a np array of observations as the batch
                probs = sess.run(
                    self.actor_output_tensor,
                    feed_dict={
                        self.input_tensor: np.array([obs])  # A dirty shape hack
                    },
                )

                #action = np.argmax(probs)
                action = np.random.choice(env.action_space.n,
                    p=np.ndarray.flatten(probs))


                actions.append(action)
                frame, reward, done, info = env.step(action)
                obs = self.preprocess(frame)
                rewards.append(reward)
                dones.append(done)

                cur_steps += 1

            values = sess.run(
                    self.critic_output_tensor,
                    feed_dict={
                        self.input_tensor: np.stack(states)
                        },
            )

            R = np.zeros_like(values)

            # If the episode did not end, len(rewards) == self.n, but if it did,
            # len(rewards) <= self.n
            for t in range(self.n):
                if dones[t] == 1:
                    cumulative_discounted = 0
                else:
                    cumulative_discounted = values[t]

                for t_prime in range(len(rewards) - t - 1, t-1, -1):
                    cumulative_discounted = rewards[t_prime] + self.gamma * \
                        cumulative_discounted
                R[t] = cumulative_discounted

                if dones[t] == 1:
                    break


            '''
            N = self.n
            T = len(rewards)
            exp = self.gamma ** N
            for t in range(T-1, -1, -1):
                v_end = 0 if t+N >= T else values[t + N]
                cumulative = 0
                for k in range(N-1):
                    cumulative += (self.gamma ** k) * (rewards[t+k] if t+k < T else 0)
                R[t] = exp * v_end + cumulative
            '''

            actor_loss, critic_loss, entropy, _ = sess.run(
                    [self.actor_loss, self.critic_loss, self.entropy, self.train_both],
                    feed_dict={
                        self.input_tensor: np.stack(states),
                        self.A_tensor: actions,
                        self.R_tensor: R.reshape((-1)),
                        self.advantage_tensor: np.ndarray.flatten(R - values),
                    },
            )

            total_steps += cur_steps
            total_reward += np.sum(rewards)
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_entropy += entropy
            iterations += 1
            cur_steps = 0

        return (total_actor_loss / iterations, total_critic_loss / iterations,
            total_entropy / iterations, total_reward, total_steps, iterations)


    def test(self):
        pass

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int, default=1000000)
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.99)
    parser.add_argument('--n', dest='n', type=int, default=10000)

    return parser.parse_args()


def main(args):
    args = parse_arguments()
    lr = args.lr
    gamma = args.gamma
    n = args.n
    num_episodes = args.num_episodes

    a2c = A2C(lr, gamma, n)

    rewards = []
    episode_lens = []
    losses = []
    critic_losses = []
    entropies = []
    total_iterations = 0

    current_total_reward = 0
    current_total_ep_len = 0
    current_total_loss = 0
    current_total_critic_loss = 0
    current_total_entropy = 0

    for episode in range(num_episodes):
        loss, critic_loss, entropy, total_reward, episode_len, iterations = a2c.train()
        total_iterations += iterations
        print('#' * 50)
        print('Episode: %d' % episode)
        print('Reward: %f' % total_reward)
        print('Steps: %d' % episode_len)
        print('Loss: %f' % loss)
        print('Critic loss: %f' % critic_loss)
        print('Entropy: %f' % entropy)
        print('Iterations: %d' % iterations)
        print('Total iterations: %d' % total_iterations)

        current_total_reward += total_reward
        current_total_ep_len += episode_len
        current_total_loss += loss
        current_total_critic_loss += critic_loss
        current_total_entropy += entropy

        if episode % 100 == 0:
            rewards.append(current_total_reward / 100)
            episode_lens.append(current_total_ep_len / 100)
            losses.append(current_total_loss / 100)
            critic_losses.append(current_total_critic_loss / 100)
            entropies.append(current_total_entropy / 100)

            current_total_reward = 0
            current_total_ep_len = 0
            current_total_loss = 0
            current_total_critic_loss = 0
            current_total_entropy = 0

            if not os.path.exists('saves'):
                os.mkdir('saves')
                a2c.saver.save(sess, 'saves/')
            plt.clf()
            rewards_line = plt.plot([100*i for i in range(len(rewards))], rewards, aa=True)
            plt.axis([0, 100*(len(rewards)+1), 0, 200])

            plt.xlabel('Episodes')
            plt.ylabel('Average reward per 100 episodes')
            plt.savefig('rewards_a2c.png')

            plt.clf()
            episode_length_line = plt.plot([100*i for i in range(len(episode_lens))], episode_lens, aa=True)
            plt.axis([0, 100*(len(episode_lens)+1), 0, 1000])

            plt.xlabel('Episodes')
            plt.ylabel('Average episode length per 100 episodes')
            plt.savefig('episode_length_a2c.png')

            plt.clf()
            loss_line = plt.plot([100*i for i in range(len(losses))], losses, aa=True, label='Actor loss')
            critic_loss_line = plt.plot([100*i for i in range(len(critic_losses))], critic_losses, aa=True, label='Critic loss')

            plt.legend()
            plt.axis([0, 100*(len(losses)+1), 0, 0.5])

            plt.xlabel('Episodes')
            plt.ylabel('Average loss per 100 episodes')
            plt.savefig('losses_a2c.png')

            plt.clf()
            entropy_line = plt.plot([100*i for i in range(len(entropies))], entropies, aa=True)
            plt.axis([0, 100*(len(episode_lens)+1), 0, 5])

            plt.xlabel('Episodes')
            plt.ylabel('Average entropy per 100 episodes')
            plt.savefig('entropy_a2c.png')






if __name__ == '__main__':
    main(sys.argv)
