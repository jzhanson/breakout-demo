import sys
import argparse
import numpy as np
import tensorflow as tf
import subprocess
import keras
import gym
import os
import json
from time import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'} 
from keras.layers import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rev = str(subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE).stdout, 'utf-8').strip()[:5]
print("rev: %s" % rev)

key = str(np.random.randint(1000))
print("key: %s" % key)
last = time()

sess = tf.Session()
    # dissables annonying tensorflow warnings

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr, normalized_values=False, reduce_reward=True):
        self.normalized_values = normalized_values
        self.reduce_reward = reduce_reward
        self.value_bias = 0
        self.mean_values = []

        self.input_tensor = tf.placeholder(tf.float32, shape=[None, 8])
        self.G_tensor = tf.placeholder(tf.float32, shape=[None])
        self.A_tensor = tf.placeholder(tf.int32, shape=[None])
        self.optimiser = tf.train.AdamOptimizer(lr)

        self.output_tensor = model(self.input_tensor)

        action_probabilties = tf.einsum( 
            "ij,ij->i", 
            tf.one_hot(self.A_tensor, 4), 
            self.output_tensor, 
        )

        self.loss = -tf.reduce_mean(tf.log(action_probabilties) * self.G_tensor)
        self.train_op = self.optimiser.minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=100)

        sess.run(tf.global_variables_initializer())

    def run(self, states): 
        return sess.run(
            self.output_tensor, 
            feed_dict={self.input_tensor : states},
        )

    def update_bias(self, bias):
        self.mean_values.append(bias)
        if len(self.mean_values) == 100: 
            self.value_bias = np.mean(self.mean_values)
            self.mean_values.clear()

    def train(self, env, gamma=1.0, batch_size=10):
        # Trains the model on a single episode using REINFORCE.

        states_total, actions_total, G_total = ([]), ([]), ([])
        r = []
        for i in range(batch_size):
            states, actions, rewards = self.generate_episode(env)
            r.extend(rewards)
            num_episodes = len(rewards)

            G = np.zeros(num_episodes)
            cumulative = 0
            for t in reversed(range(num_episodes)): 
                cumulative = rewards[t] + cumulative * gamma
                if self.reduce_reward: 
                    G[t] = cumulative * (gamma ** t)
                else: 
                    G[t] = cumulative

            states_total.append(states)
            actions_total.append(actions)
            G_total.append(G)

        states_total = np.vstack(states_total)
        actions_total = np.hstack(actions_total)
        G_total = np.hstack(G_total)

        G_total -= np.mean(G_total) 
        if self.normalized_values: 
            G_total /= np.std(G_total)

        _, loss = sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.input_tensor : np.vstack(states_total),
                self.A_tensor : actions_total,
                self.G_tensor : G_total,
            }
        )
        return (loss, np.sum(r) // batch_size, len(r) // batch_size)

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []

        obs = env.reset()
        if render:
            env.render()
        done = False

        while not done:
            if render: 
                env.render()
            states.append(obs)
            # Predict takes a np array of observations as the batch
            probs = sess.run(
                self.output_tensor, 
                feed_dict={
                    self.input_tensor:np.matrix(obs),
                },
            )
            # Instead of taking argmax, we sample from softmax probabilities,
            # and also there's no need for actions to be one-hot
            action = np.random.choice(env.action_space.n,
                p=np.ndarray.flatten(probs))

            actions.append(action)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)

        return np.array(states), np.array(actions), np.array(rewards)


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=1.0, help="The gamma value.")
    parser.add_argument('--normalized_values', dest='normalized_values', type=bool,
                        default=True, help="normalize the G values")
    parser.add_argument('--reduce_reward', dest='reduce_reward', type=bool,
                        default=True, help="reduce the reward by a factor of 100")
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        default=1, help="train episode batch size")
    parser.add_argument('--test', dest='test', type=str,
                        default='', help="test a file")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def custom_model():
    m = keras.models.Sequential(
        layers=[
            Dense(
                32, 
                input_shape=[8],
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.03, seed=None),
            ),
            BatchNormalization(),
            Activation('relu'),
            Dense(
                24, 
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.03, seed=None),
            ),
            BatchNormalization(),
            Activation('relu'),
            Dense(
                16, 
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.03, seed=None),
            ),
            BatchNormalization(),
            Activation('relu'),
            Dense(
                4, 
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.03, seed=None),
            ),
            Activation('softmax'),
        ]
    )
    return m


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    gamma = args.gamma
    render = args.render
    batch_size = args.batch_size

    args_json = {
        'num_episodes' : args.num_episodes,
        'lr' : args.lr,
        'gamma' : args.gamma,
        'render' : args.render,
        'batch_size' : batch_size,
    }
    print(args_json)

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    learning_rate = lr
    reinforce = Reinforce(
        model, learning_rate, 
        normalized_values=args.normalized_values,
        reduce_reward=args.reduce_reward,
    )
    training_episodes = num_episodes
    current_episode = 0

    if args.test != '':
        reinforce.saver.restore(sess, args.test)
        rewards = []
        for i in range(10):
            rewards.append(sum(reinforce.generate_episode(env)[2]))
        print('Reward: %f +/- %f' % (np.mean(rewards), np.std(rewards)))
        return 

    # Every k (freeze_episodes) episodes, freeze policy and run 100 test
    # episodes, recording mean and stdev
    freeze_episodes = 10
    test_reward_means = []
    test_reward_stdevs = []
    losses = []
    rewards = []
    steps = []
    explosion = False
    first_dump = True
    batch_size = 1
    while current_episode < training_episodes:
        loss, reward, step = reinforce.train(env, gamma=gamma, batch_size=batch_size)
        losses.append(loss)
        rewards.append(reward)
        steps.append(step)

        if reward > 200:
            global last
            if not os.path.exists('saves'): 
                os.mkdir('saves')
            if last + 60*15 < time():
                file = 'rev%s-key%s-ep%d-reward%d' % (rev, key, current_episode, np.mean(reward))
                reinforce.saver.save(sess, 'saves/' + file)
                last = time()

            if first_dump == True: 
                file = 'rev%s-key%s' % (rev, key)
                with open('saves/' + file + '.args', "w+", encoding='utf-8') as f:
                   json.dump(args_json, f, ensure_ascii=False) 
                first_dump == False


        if (current_episode % freeze_episodes == 0):
            print('#' * 50)
            print('Episode: %d' % current_episode)
            print('Reward: %f +/- %f' % (np.mean(rewards), np.std(rewards)))
            print('Loss: %f +/- %f' % (np.mean(losses), np.std(losses)))
            print('Steps: %d +/- %f' % (np.mean(steps), np.std(steps)))

            if np.sum(np.abs(losses)) < 1e-6 or np.all(np.isnan(losses)):
                if explosion: # no point in going further if the loss function kapoofs
                    print('#' * 50)
                    print('losses exploded')
                    return
                else: 
                    explosion = True
            else:
                explosion = False

            rewards.clear()
            losses.clear()
            steps.clear()

            if render:
                reinforce.generate_episode(env, render=True)

            # total_rewards_arr = np.array(total_rewards)
            # test_reward_means.append(np.mean(total_rewards_arr, axis=0))
            # test_reward_stdevs.append(np.std(total_rewards_arr, axis=0))

            # plt.clf()
            # plt.errorbar(
            #     [i*freeze_episodes for i in range(current_episode //
            #         freeze_episodes)],
            #     test_reward_means,
            #     yerr=test_reward_stdevs
            # )
            # plt.xlabel('Number of training episodes')
            # plt.ylabel('Average cumulative undiscounted reward per episode over \
            #     100 test episodes')
            # plt.savefig('reinforce.png')
            # plt.savefig('reinforce.pdf')
        current_episode += 1



if __name__ == '__main__':
    main(sys.argv)
