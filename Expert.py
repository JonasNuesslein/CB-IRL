import gym
from stable_baselines3 import DQN, PPO
import numpy as np
import time

import utils
import CustomEnvs



def train_expert(env, domain):
    if domain == "mountain_car":
        model = DQN("MlpPolicy",
                    env,
                    learning_rate=0.004,
                    batch_size=128,
                    buffer_size=10000,
                    learning_starts=1000,
                    gamma=0.98,
                    target_update_interval=600,
                    train_freq=16,
                    gradient_steps=8,
                    exploration_fraction=0.2,
                    exploration_final_eps=0.07,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    verbose=1)
        model.learn(total_timesteps=200000)
    elif domain == "acrobot":
        model = PPO("MlpPolicy",
                    env,
                    n_steps=256,
                    gamma=0.99,
                    gae_lambda=0.94,
                    n_epochs=4,
                    ent_coef=0.0,
                    verbose=1)
        model.learn(total_timesteps=150000)
    elif domain == "lunarlander":
        model = PPO("MlpPolicy",
                    env,
                    gamma=0.999,
                    ent_coef=0.01,
                    n_steps=1024,
                    batch_size=64,
                    gae_lambda=0.98,
                    verbose=1)
        model.learn(total_timesteps=1000000)
    elif domain == "half_cheetah_discrete":
        model = PPO("MlpPolicy",
                    env,
                    verbose=1)
        model.learn(total_timesteps=1000000)
    file = "./trained_models/" + domain + "/expert"
    model.save(file)



def get_expert_trajectory(domain, deterministic, seed=None):

    if domain == "mountain_car":
        env = gym.make("MountainCar-v0")
    elif domain == "acrobot":
        env = gym.make("Acrobot-v1")
    elif domain == "lunarlander":
        env = gym.make("LunarLander-v2")
    else:
        env = CustomEnvs.DiscreteHalfCheetah(20)

    done = False
    env.seed(seed)
    o = env.reset()
    obs, acts, rews = [o], [], []

    file = "./trained_models/" + domain + "/expert"
    if domain == "mountain_car":
        model = DQN.load(file)
    elif domain == "acrobot" or domain == "lunarlander" or domain == "half_cheetah_discrete":
        model = PPO.load(file)

    c = 0
    while not done:
        a, _ = model.predict(o, deterministic=deterministic)
        o, r, done, _ = env.step(a)
        if c%10 == 0:
            obs.append(o)
        acts.append(a)
        rews.append(r)
        c += 1
        #env.render()

    # ensure expert trajectory is indeed good
    if domain == "lunarlander" and np.sum(rews) < 230:
        return get_expert_trajectory(domain, deterministic, seed)

    return np.array(obs), np.array(acts), np.array(rews, dtype=np.float32)



def test_expert(env, domain):
    file = "./trained_models/" + domain + "/expert"
    if domain == "mountain_car":
        model = DQN.load(file)
    elif domain == "acrobot" or domain == "lunarlander" or domain == "half_cheetah_discrete":
        model = PPO.load(file)
    rews = utils.run_episode(env, model, print_reward=False, render=False, domain=domain)
    return rews



