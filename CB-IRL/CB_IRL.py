from stable_baselines3 import DQN, PPO
import gym
import pickle

import config
import EqualityNet
import Logging
import CustomEnvs



class newEnvironment(gym.Env):

    def __init__(self, env, equalityNet, hyps):
        self.env = env
        self.equalityNet = equalityNet
        self.hyps = hyps

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.trajectory, self.rews = [], []

    def step(self, action):
        o, _, done, _ = self.env.step(action)
        self.trajectory.append(o)
        self.rews.append(self.equalityNet.reward(o))
        reward = self.rews[-1] - self.hyps["alpha"] * self.rews[-2]
        if done:
            self.equalityNet.replay_buffer.append(self.trajectory)
            self.equalityNet.train(1)
        return o, reward, done, {}

    def reset(self):
        o = self.env.reset()
        self.trajectory = [o]
        self.rews = [0]
        return o



# newEnvironment is a Gym Environment with the reconstructed Reward function

if __name__ == "__main__":
    env = CustomEnvs.DiscreteHalfCheetah(20)
    # expert_data is a list of trajectories. A trajectory is a list of states.
    expert_data = pickle.load(open("tests/expert_data_half_cheetah_discrete.p", "rb"))
    hyps = config.halfcheetahdiscrete_hyperparameters
    equalityNet = EqualityNet.EqualityNet(env, expert_data, hyps)
    newEnv = newEnvironment(env, equalityNet, hyps)
    callback = Logging.EvalCallback(env, newEnv, check_freq=10000)
    model = PPO("MlpPolicy",
                newEnv,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=0.0001,
                ent_coef=0.001,
                verbose=0)
    model.learn(total_timesteps=1000000, callback=callback)

