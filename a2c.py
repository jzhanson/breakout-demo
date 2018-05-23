import os
import sys
import argparse
from skimage.transform import resize
from PIL import Image
import numpy as np
from collections import deque

'''
import matplotlib
matplotlib.use('Agg')
'''
import matplotlib.pyplot as plt
import tensorflow as tf

import gym
from gym import spaces

import time

gpu_ops = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_ops)
sess = tf.Session(config=config)

# tf.set_random_seed(seed)

# Wraps Breakout environment by grayscaling and cropping + resizing the images
# to 84x84, stacking 4 frames to form the observation, takes fire action to
# begin episode, treats loss of life as end of an episode.
class WrapperEnv(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        gym.Wrapper.__init__(self, env)
        self.num_stack = num_stack
        shape = env.observation_space.shape
        self.side = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(shape[0],
            shape[1], self.num_stack))

        self.frames = deque([], maxlen=self.num_stack)

        self.lives = 0
        self.out_of_lives = False
        self.started = False

    def grayscale_resize(self, frame):
        #obs = resize(np.array(frame)[...,:3].dot([0.299, 0.587, 0.114]), (84, 84))
        cropped_frame = frame[34:-16,:,:]
        gray_frame = np.dot(cropped_frame.astype('float32'),
            np.array([0.299, 0.587, 0.114], 'float32'))
        obs = np.array(Image.fromarray(gray_frame).resize((self.side, self.side),
                        resample=Image.BILINEAR), dtype=np.uint8)
        return obs

    def _reset(self):
        if self.out_of_lives or not self.started:
            frame = self.grayscale_resize(self.env.reset())
            for i in range(self.num_stack):
                self.frames.append(frame)
            self.started = True
            self.out_of_lives = False
            obs, reward, done, info = self._step(1)
        # Press fire at the beginning
        else:
            obs, reward, done, info = self._step(1)

        self.lives = self.env.unwrapped.ale.lives()
        return self._observation()

    def _step(self, action):
        frame, reward, done, info = self.env.step(action)
        self.out_of_lives = done
        self.frames.append(self.grayscale_resize(frame))

        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        # This if is redundant
        if lives == 0:
            self.out_of_lives = True
        self.lives = lives

        return self._observation(), reward, done, info

    def _observation(self):
        return np.stack(self.frames, axis=-1)



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
    def __init__(self, lr, gamma, n, num_envs=4):
        self.lr = lr
        self.n = n
        self.gamma = gamma
        self.num_envs = num_envs

        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.env = WrapperEnv(self.env)
        self.env = gym.wrappers.Monitor(self.env, './videos_a2c', force=True,
            video_callable=lambda episode_id: episode_id%100==0)


        '''
        self.envs = [gym.make('BreakoutNoFrameskip-v4') for i in
            range(self.num_envs)]

        self.envs = list(map(lambda e: WrapperEnv(e), self.envs))

        self.envs = list(map(lambda e: gym.wrappers.Monitor(e, './videos_a2c', force=True,
            video_callable=lambda episode_id: episode_id%100==0), self.envs))

        self.obs_arr = []
        '''

        self.make_model()


    def make_model(self):
        self.input_tensor = tf.placeholder(tf.float32, shape=(None, 84, 84, 4))

        self.R_tensor = tf.placeholder(tf.float32, shape=[None])
        self.A_tensor = tf.placeholder(tf.int32, shape=[None])
        # We have advantage_tensor be separate in order so the gradients for
        # the actor don't go through the critic
        self.advantage_tensor = tf.placeholder(tf.float32, shape=[None])

        self.global_step = tf.Variable(0, trainable=False)

        self.lr_tensor = tf.placeholder(tf.float32, shape=[])

        with tf.variable_scope("policy"):
            conv1 = tf.layers.conv2d(
                    inputs=(tf.cast(self.input_tensor, tf.float32) / 255),
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
                    self.env.action_space.n,
                    activation_fn=None,
                    weights_initializer=tf.initializers.orthogonal()
            )

            # Add a little noise to encourage exploration!
            # self.actor_output_three = tf.nn.softmax(self.actor_policy_logits - tf.log(-tf.log(noise)))
            # self.actor_output_tesnor = tf.nn.softmax(self.actor_policy_logits + noise)
            '''
            noise = tf.random_uniform(tf.shape(self.actor_policy_logits))
            self.actor_output_tensor = tf.nn.softmax(self.actor_policy_logits - tf.log(-tf.log(noise)), 1)
            '''
            self.actor_output_tensor = tf.nn.softmax(self.actor_policy_logits)


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
        # For some reason reference implementation has mse be / 2
        self.critic_loss = tf.reduce_mean(tf.square(self.R_tensor -     \
            tf.squeeze(self.critic_output_tensor)) / 2)
        self.entropy = tf.reduce_mean(openai_entropy(self.actor_policy_logits))
        self.loss = self.actor_loss + 0.5 * self.critic_loss - 0.01 * self.entropy

        with tf.variable_scope("policy"):
            params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        grads = list(zip(grads, params))

        '''
        self.optimizer = tf.train.RMSPropOptimizer(self.lr_tensor,
            decay=0.99, epsilon=1e-5)
        '''
        self.optimizer = tf.train.RMSPropOptimizer(self.lr,
            decay=0.99, epsilon=1e-5)

        self.train_both = self.optimizer.apply_gradients(grads)

        self.saver = tf.train.Saver(max_to_keep=1)
        sess.run(tf.global_variables_initializer())


    # Performs one iteration of n or fewer steps on each environment
    def train(self, render=True):
        obs = self.env.reset()

        if render:
            self.env.render()
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
                    self.env.render()
                states.append(obs)

                probs, logits = sess.run(
                    [self.actor_output_tensor, self.actor_policy_logits],
                    feed_dict={
                        self.input_tensor: np.array([obs])
                    },
                )

                sys.stdout.write('\r' + str(probs))
                sys.stdout.flush()
                #action = np.argmax(probs)
                action = np.random.choice(self.env.action_space.n,
                    p=np.ndarray.flatten(probs))

                actions.append(action)
                obs, reward, done, info = self.env.step(action)
                rewards.append(reward)
                dones.append(done)

                cur_steps += 1

            last_value = sess.run(
                self.critic_output_tensor,
                feed_dict={
                    self.input_tensor: np.array([obs])
                },
            )

            values = sess.run(
                self.critic_output_tensor,
                feed_dict={
                    self.input_tensor: np.array(states)
                },
            )

            R = np.zeros_like(np.array(values))

            if dones[-1] == 1:
                cumulative_discounted = 0
            else:
                cumulative_discounted = last_value

            for t in range(len(rewards)-1, -1, -1):
                cumulative_discounted = rewards[t] + self.gamma * cumulative_discounted
                R[t] = cumulative_discounted


            '''
            R = np.zeros_like(values)

            for t in range(self.n):
                if dones[t] == 1:
                    cumulative_discounted = 0
                else:
                    cumulative_discounted = values[t]

                for t_prime in range(len(rewards) - 1, t-1, -1):
                    cumulative_discounted = rewards[t_prime] + self.gamma * \
                        cumulative_discounted

                R[t] = cumulative_discounted

                if dones[t] == 1:
                    break
            '''

            actor_loss, critic_loss, entropy, _ = sess.run(
                [self.actor_loss, self.critic_loss, self.entropy, self.train_both],
                feed_dict={
                    self.input_tensor: np.array(states),
                    self.A_tensor: np.array(actions),
                    self.R_tensor:  R.reshape((-1)),
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

        '''
        obs = list(map(lambda env: env.reset(), self.envs))

        if render:
            map(lambda env: env.render(), self.envs)
        all_dones = [False for i in range(self.num_envs)]

        cur_steps = 0
        total_steps = 0
        total_reward = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        iterations = 0

        while False in all_dones:

            all_states = []
            all_actions = []
            all_rewards = []

            #all_values = []
            all_Rs = []
            all_advs = []

            for i in range(self.num_envs):
                if all_dones[i]:
                    continue

                states = []
                actions = []
                rewards = []
                dones = []

                while cur_steps < self.n and not all_dones[i]:
                    if render:
                        self.envs[i].render()
                    states.append(obs[i])

                    probs = sess.run(
                        self.actor_output_tensor,
                        feed_dict={
                            self.input_tensor: np.array([obs[i]])
                        },
                    )

                    #action=np.argmax(probs)
                    action = np.random.choice(self.envs[i].action_space.n,
                        p=np.ndarray.flatten(probs))

                    actions.append(action)
                    curr_obs, reward, done, info = self.envs[i].step(action)
                    obs[i] = curr_obs
                    rewards.append(reward)
                    dones.append(done)
                    all_dones[i] = done

                    cur_steps += 1


                values = sess.run(
                    self.critic_output_tensor,
                    feed_dict={
                        self.input_tensor: np.stack(states)
                    },
                )

                R = np.zeros_like(values)

                for t in range(self.n):
                    if dones[t] == 1:
                        cumulative_discounted = 0
                    else:
                        cumulative_discounted = values[t]

                    for t_prime in range(len(rewards) - 1, t-1, -1):
                        cumulative_discounted = rewards[t_prime] + self.gamma * \
                            cumulative_discounted

                    R[t] = cumulative_discounted

                    if dones[t] == 1:
                        break


                all_states.append(np.array(states))
                all_actions.append(np.array(actions))
                all_rewards.append(np.array(rewards))

                #all_values.append(np.array(values))
                all_Rs.append(np.array(R))
                all_advs.append(np.array(R) - np.array(values))

                total_steps += cur_steps
                total_reward += np.sum(rewards)

                cur_steps = 0


            all_states = np.array(all_states)
            all_actions = np.concatenate(all_actions)
            all_rewards = np.array(all_rewards)
            #all_values = np.array(all_values)
            all_Rs = np.concatenate(all_Rs)
            all_advs = np.concatenate(all_advs)

            actor_loss, critic_loss, entropy, _ = sess.run(
                [self.actor_loss, self.critic_loss, self.entropy, self.train_both],
                feed_dict={
                    self.input_tensor: np.concatenate(all_states),
                    self.A_tensor: all_actions.reshape((-1)),
                    self.R_tensor:  all_Rs.reshape((-1)),
                    self.advantage_tensor: all_advs.reshape((-1)),
                },
            )

            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_entropy += entropy
            iterations += 1

        return (total_actor_loss / iterations, total_critic_loss / iterations,
            total_entropy / iterations, total_reward / self.num_envs, total_steps / self.num_envs, iterations)
        '''

    def test(self):
        pass

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int, default=1000000)
    parser.add_argument('--lr', dest='lr', type=float, default=7e-4)
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


    '''
    a2c.obs_arr = np.array(list(map(lambda env: env.reset(), a2c.envs)))
    '''

    for episode in range(num_episodes):
        # 5 lives/sub-episodes per episode
        loss, critic_loss, entropy, total_reward, episode_len, iterations = 0, \
            0, 0, 0, 0, 0
        for i in range(5):
            sub_loss, sub_critic_loss, sub_entropy, sub_total_reward, sub_episode_len, sub_iterations = a2c.train()
            loss += sub_loss
            critic_loss += sub_critic_loss
            entropy += sub_entropy
            total_reward += sub_total_reward
            episode_len += sub_episode_len
            iterations += sub_iterations
        total_iterations += iterations
        entropy = entropy / 5
        loss = loss / 5
        critic_loss = critic_loss / 5
        print('\n')
        print('#' * 50)
        print('Episode: %d' % episode)
        print('Reward: %f' % total_reward)
        print('Steps: %f' % episode_len)
        print('Loss: %f' % )
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
            plt.axis([0, 100*(len(episode_lens)+1), 0, 2000])

            plt.xlabel('Episodes')
            plt.ylabel('Average episode length per 100 episodes')
            plt.savefig('episode_length_a2c.png')

            plt.clf()
            loss_line = plt.plot([100*i for i in range(len(losses))], losses, aa=True, label='Actor loss')
            critic_loss_line = plt.plot([100*i for i in range(len(critic_losses))], critic_losses, aa=True, label='Critic loss')

            plt.legend()
            plt.axis([0, 100*(len(losses)+1), -0.1, 0.1])

            plt.xlabel('Episodes')
            plt.ylabel('Average loss per 100 episodes')
            plt.savefig('losses_a2c.png')

            plt.clf()
            entropy_line = plt.plot([100*i for i in range(len(entropies))], entropies, aa=True)
            plt.axis([0, 100*(len(episode_lens)+1), 0, 2])

            plt.xlabel('Episodes')
            plt.ylabel('Average entropy per 100 episodes')
            plt.savefig('entropy_a2c.png')



if __name__ == '__main__':
    main(sys.argv)
