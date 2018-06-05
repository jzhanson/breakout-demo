# breakout-demo

Super simple, no-frills A2C agent that achieves over 200 reward on Atari Breakout, with [MG2033/A2C](https://github.com/MG2033/A2C) and [openai/baselines](https://github.com/openai/baselines) as reference.

Here's the corresponding [blog post](http://blog.jzhanson.com/blog/rl/project/2018/05/28/breakout.html).

To train a model from scratch, run

`python a2c.py`

I recommend a value of N = 50 or 100 for best results, though training does take some time with those values.

`python a2c.py --n 100`

Better graphs, Tensorboard visualizations, testing, and saved model files on the way.

### Results

| N     | Max Reward | Iterations before overfit |
|-------|------------|------------|
|1      |            |            |
|5      |            |            |
|20     | 376        |            |
|50     | 397        | Less than 1020626  |
|100    | 428        | Less than 1031646  |
|Inf    |             |           |


