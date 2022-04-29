import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
import time



def run_episode(env, model=None, print_reward=False, render=False, seed=None, domain=None):
    env.seed(seed)
    obs, rews, done = env.reset(), [], False
    while not done:
        if model == None:
            a = env.action_space.sample()
        else:
            if np.random.random() < 0.0:
                a, _ = model.predict(obs, deterministic=False)
            else:
                a, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = env.step(a)
        rews.append(r)
        if render:
            env.render()
        if print_reward:
            print("reward[", len(rews),"]: ", r)
    return rews




