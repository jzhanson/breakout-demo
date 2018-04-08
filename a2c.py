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

def critic_model():
    layers = [
        Dense(
            16,
            input_shape=[8],
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
            16,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.03, seed=None),
        ),
        BatchNormalization(),
        Activation('relu'),
        Dense(
            1,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.03, seed=None),
        ),
    ]
    return keras.models.Sequential(layers=layers)



class A2C(object):
    def __init__(self, model, lr, critic_model, critic_lr, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.n = n
        self.lr = lr
        self.critic_lr = critic_lr
        self.make_model(model, critic_model)

    def make_model(self, model, critic_model):
        # actor
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, 8])
        self.R_tensor = tf.placeholder(tf.float32, shape=[None])
        self.A_tensor = tf.placeholder(tf.int32, shape=[None])

        with tf.variable_scope("actor"):
            self.actor_output_tensor = model(self.input_tensor)

        with tf.variable_scope("critic"):
            self.critic_output_tensor = critic_model(self.input_tensor)

        self.actor_optimizer = tf.train.AdamOptimizer(self.lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr)

        action_probabilties = tf.einsum(
            "ij,ij->i",
            tf.one_hot(self.A_tensor, 4),
            self.actor_output_tensor,
        )

        difference = self.R_tensor - self.actor_output_tensor

        self.actor_loss = -tf.reduce_sum(difference * tf.log(action_probabilties))

        self.critic_loss = tf.reduce_sum(tf.square(difference))

        self.train_critic = self.critic_optimizer.minimize(
            self.critic_loss,
            var_list=tf.trainable_variables(scope="critic"),
        )

        self.train_actor = self.actor_optimizer.minimize(
            self.actor_loss,
            var_list=tf.trainable_variables(scope="actor"),
        )

        self.saver = tf.train.Saver(max_to_keep=100)
        sess.run(tf.global_variables_initializer())


    def train(self, env, gamma=1.0):
        states, actions, rewards = self.generate_episode(env)

        values = sess.run(
            self.critic_output_tensor,
            feed_dict={
                self.input_tensor: np.vstack(states)
            },
        )

        R = np.zeros_like(values)
        N = self.n
        T = len(values)
        cumulative = 0
        exp = gamma ** N
        for t in reversed(range(T)):
            v_end = 0 if (t + N) >= T else values[t + N]
            rem = rewards[t + N] if t + N < T else 0
            cumulative = rewards[t] + gamma * cumulative
            cumulative -= rem * exp
            R[t] = exp * v_end + cumulative

        actor_loss, critic_loss, _, _ = sess.run(
            [self.actor_loss, self.critic_loss, self.train_actor, self.train_actor],
            feed_dict={
                self.input_tensor: np.vstack(states),
                self.A_tensor : actions,
                self.R_tensor : R,
            },
        )

        return (actor_loss, critic_loss, np.sum(rewards), T)




        return

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
                self.actor_output_tensor,
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
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.99, help="gamma value")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

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


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    gamma = args.gamma
    render = args.render
    args_json = {
        'num_episodes' : args.num_episodes,
        'lr' : args.lr,
        'critic_lr' : args.critic_lr,
        'n' : args.n,
        'gamma' : args.gamma,
        'render' : args.render,
    }

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the actor model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    a2c = A2C(
        model,
        lr,
        critic_model(),
        critic_lr,
    )

    training_episodes = num_episodes
    current_episode = 0

    if args.test != '':
        a2c.saver.restore(sess, args.test)
        rewards = []
        for i in range(10):
            rewards.append(sum(a2c.generate_episode(env)[2]))
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
        loss, _, reward, step = a2c.train(env, gamma=gamma)
        losses.append(loss)
        rewards.append(reward)
        steps.append(step)

        if reward > 200:
            global last
            if not os.path.exists('saves'):
                os.mkdir('saves')
            if last + 60*15 < time():
                file = 'rev%s-key%s-ep%d-reward%d' % (rev, key, current_episode, np.mean(reward))
                a2c.saver.save(sess, 'saves/' + file)
                last = time()

            if first_dump == True:
                file = 'rev%s-key%s' % (rev, key)
                with open('saves/' + file + '.args', "w+", encoding='utf-8') as f:
                    json.dump(args_json, f, ensure_ascii=False)
                first_dump == False


        if (current_episode % freeze_episodes == 0):
            print('#' * 50)
            print('Episode: %d' % current_episode)
            print('Reward: %f +/- %f' % (np.mean(rewards) / batch_size, np.std(rewards) / batch_size))
            print('Loss: %f +/- %f' % (np.mean(losses) / batch_size, np.std(losses) / batch_size))
            print('Steps: %d +/- %f' % (np.mean(steps) / batch_size, np.std(steps) / batch_size))

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
                a2c.generate_episode(env, render=True)

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
            # plt.savefig('a2c.png')
            # plt.savefig('a2c.pdf')
        current_episode += 1


if __name__ == '__main__':
    main(sys.argv)
