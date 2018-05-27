# breakout-demo

Super simple, no-frills A2C agent that achieves over 200 reward on Atari Breakout, with [MG2033/A2C](https://github.com/MG2033/A2C) and [openai/baselines](https://github.com/openai/baselines) as reference.

To train a model from scratch, run

`python a2c.py`

Better graphs, Tensorboard visualizations, testing, and saved model files on the way.

### Results

| N     | Max Reward |
|-------|------------|
|1      |            |
|5      |            |
|20     | 376        |
|50     | 300 (so far)|
|100    | 297 (so far)|
|Inf    |             |

### N = 20 graphs

It is interesting to note that after about 9000 training episodes (4M iterations) on one environment the softmax output converges to zeroes and ones and the agent performance plummets --- a case of "overfitting" in reinforcement learning! Despite the learning rate decay, training is still not stable enough to continue slow improvement or even plateau.

![Entropy](./img/20_entropy.png)
![Losses](./img/20_losses.png)
![Episode length](./img/20_episode_length.png)
![Max reward](./img/20_max_reward.png)
![Average reward](./img/20_rewards.png)

