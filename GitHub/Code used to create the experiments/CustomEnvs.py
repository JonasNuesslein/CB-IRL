import gym
from gym import spaces
import numpy as np
from sklearn_extra.cluster import KMedoids

import Expert



class MountainCar(gym.Env):

    def __init__(self):
        self.env = gym.make("MountainCar-v0")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.finish = False
        self.last_obs = None
        self.steps = 0

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.steps += 1
        if done and self.steps < 200 and not self.finish:
            self.finish = True
            self.last_obs = obs
        elif self.finish:
            obs = self.last_obs
            reward = 0
        done = True if self.steps == 200 else False
        #reward = 0
        return obs, reward, done, {}

    def reset(self):
        obs = self.env.reset()
        self.finish = False
        self.last_obs = None
        self.steps = 0
        return obs

    def render(self):
        self.env.render()

    def seed(self, seed):
        self.env.seed(seed)



class Acrobot(gym.Env):

    def __init__(self):
        self.env = gym.make("Acrobot-v1")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.finish = False
        self.last_obs = None
        self.steps = 0

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.steps += 1
        if done and self.steps < 500 and not self.finish:
            self.finish = True
            self.last_obs = obs
        elif self.finish:
            obs = self.last_obs
            reward = 0
        done = True if self.steps == 500 else False
        #reward = 0
        return obs, reward, done, {}

    def reset(self):
        obs = self.env.reset()
        self.finish = False
        self.last_obs = None
        self.steps = 0
        return obs

    def render(self):
        self.env.render()

    def seed(self, seed):
        self.env.seed(seed)



class LunarLander(gym.Env):

    def __init__(self):
        self.env = gym.make("LunarLander-v2")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.finish = False
        self.last_obs = None
        self.steps = 0

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.steps += 1
        if done and self.steps < 400 and not self.finish:
            self.finish = True
            self.last_obs = obs
        elif self.finish:
            obs = self.last_obs
            reward = 0
        done = True if self.steps == 400 else False
        #reward = 0
        return obs, reward, done, {}

    def reset(self):
        obs = self.env.reset()
        self.finish = False
        self.last_obs = None
        self.steps = 0
        return obs

    def render(self):
        self.env.render()

    def seed(self, seed):
        self.env.seed(seed)



# this Environment is a clone of the gym domain "HalfCheetah",
# but it has as an additional observation variable: the x-coordinate of the cheetah
class DiscreteHalfCheetah(gym.Env):

    def __init__(self, size):
        self.env = gym.make("HalfCheetah-v2")
        self.action_space = spaces.Discrete(size)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(18,))

        #expert_actions = []
        #for _ in range(1):
        #    _, actions, _ = Expert.get_expert_trajectory(self.env, "half_cheetah", deterministic=True, seed=0)
        #    expert_actions.extend(actions)
        #self.env.seed(None)
        #self.kmedoids = KMedoids(n_clusters=size, random_state=0).fit(expert_actions)

        self.random = np.zeros((size, 6))
        self.env.action_space.seed(0)
        for i in range(size):
            self.random[i] = self.env.action_space.sample()
        self.env.action_space.seed(None)

    def step(self, action_id):
        #continuous_action = self.kmedoids.cluster_centers_[action_id]
        continuous_action = self.random[action_id]
        obs, reward, done, _ = self.env.step(continuous_action)
        new_obs = np.concatenate((obs, [self.env.sim.data.qpos.flat[0]]))
        #reward = 0
        return new_obs, reward, done, {}

    def reset(self):
        obs = self.env.reset()
        new_obs = np.concatenate((obs, [self.env.sim.data.qpos.flat[0]]))
        return new_obs

    def render(self):
        self.env.render()

    def seed(self, seed):
        self.env.seed(seed)

