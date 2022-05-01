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



def load_model(domain, algo, model_nr, step_nr):
    file = "./trained_models/" + domain + "/" + algo + "/" + model_nr + "/model_" + step_nr

    if algo == "ours":
        if domain == "mountain_car":
            model = DQN.load(file)
        elif domain == "acrobot" or domain == "half_cheetah_discrete" or domain == "lunarlander":
            model = PPO.load(file)

    elif algo == "bc":
        pass

    elif algo == "gail":
        pass

    elif algo == "airl":
        pass

    return model



