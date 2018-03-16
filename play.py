import sys
import gym
import getch
import pickle


env = gym.make(sys.argv[1])

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

            if sys.argv[1] == 'MountainCar-v0':
                if key_pressed == 'a':
                    action = 0
                elif key_pressed == 'd':
                    action = 2
                elif key_pressed == 's':
                    action = 1
                elif key_pressed == 'q':
                    break
            elif sys.argv[1] == 'CartPole-v0':
                if key_pressed == 'a':
                    action = 0
                elif key_pressed == 'd':
                    action = 1
                elif key_pressed == 'q':
                    break
            elif sys.argv[1] == 'Breakout-v0':
                if key_pressed == 'd':
                    action = 2
                elif key_pressed == 'a':
                    action = 3
                elif key_pressed == 'e':
                    action = 1
                elif key_pressed == 's':
                    action = 0
                elif key_pressed == 'q':
                    break
        if key_pressed == 'q':
            break
        prev_obs = obs
        obs, reward, done, info = env.step(action)
        total_reward += reward
    if key_pressed == 'q':
        break
    print('Your reward was: %d' % total_reward)
