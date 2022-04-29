import numpy as np
from stable_baselines3 import DQN, PPO
import gym
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Expert
import Logging
import utils
import CustomEnvs
import config



class Embedding_Net(nn.Module):
    def __init__(self, env, hidden_size):
        super().__init__()
        self.dim = len(env.reset())
        self.fc1 = nn.Linear(2*self.dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.double()
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.sigmoid(self.fc3(x))
        return output


class ReplayBuffer():

    def __init__(self, C, hyps):
        super(ReplayBuffer, self).__init__()
        self.C = C
        self.agent_data = []
        self.hyps = hyps

    def append(self, trajectory):
        self.agent_data.append(trajectory)
        self.agent_data = self.agent_data[-self.hyps["n_trajectories"]:]

    def get_train_data(self):
        train_data = []

        if len(self.agent_data) >= 2:
            # convergence of agent data
            c = 0
            while c < 75:
                t_id = np.random.choice(len(self.agent_data))
                o1 = np.random.choice(len(self.agent_data[t_id])-self.hyps["frame_size"])
                o2 = o1 + np.random.choice(self.hyps["frame_size"])
                x = np.concatenate((self.agent_data[t_id][o1], self.agent_data[t_id][o2]))
                x2 = np.concatenate((self.agent_data[t_id][o2], self.agent_data[t_id][o1]))
                y = np.array([1])
                train_data.append([x,y])
                train_data.append([x2,y])
                c += 1
            # divergence of agent data
            c = 0
            while c < 150:
                t_ids = np.random.choice(len(self.agent_data), 2, replace=False)
                o1 = np.random.choice(len(self.agent_data[t_ids[0]]))
                o2 = np.random.choice(len(self.agent_data[t_ids[1]]))
                x = np.concatenate((self.agent_data[t_ids[0]][o1], self.agent_data[t_ids[1]][o2]))
                y = np.array([0])
                train_data.append([x,y])
                c += 1
            # divergence of agent and expert data
            c = 0
            while c < self.hyps["agent_expert_divergence"]:
                t_id = np.random.choice(len(self.agent_data))
                o1 = np.random.choice(len(self.agent_data[t_id]))
                o2 = np.random.choice(len(self.C))
                x = np.concatenate((self.agent_data[t_id][o1], self.C[o2]))
                y = np.array([0])
                train_data.append([x,y])
                c += 1

        # convergence of expert data
        c = 0
        while c < 150:
            o1 = np.random.choice(len(self.C) - (self.hyps["frame_size"] // 10))
            o2 = o1 + np.random.choice(self.hyps["frame_size"] // 10)
            x = np.concatenate((self.C[o1], self.C[o2]))
            y = np.array([1])
            train_data.append([x,y])
            c += 1
        # divergence of expert data
        c = 0
        while c < 150:
            ids = np.random.choice(len(self.C), 2)
            if np.abs(ids[0] - ids[1]) < (self.hyps["frame_size"] // 10):
                continue
            x = np.concatenate((self.C[ids[0]], self.C[ids[1]]))
            y = np.array([0])
            train_data.append([x,y])
            c += 1

        return train_data


class Embedding():

    def __init__(self, env, domain, expert_data, hyps):
        self.domain = domain
        self.C = expert_data[0][0]
        self.hyps = hyps

        self.replay_buffer = ReplayBuffer(self.C, hyps)
        self.model = Embedding_Net(env, self.hyps["hidden_size"])

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyps["lr"])
        self.train(100)

    def reward(self, obs):
        inputs = []
        for expert_obs in self.C:
            inputs.append(np.concatenate((obs, expert_obs)))
        inputs = np.array(inputs)
        outputs = self.model(torch.tensor(inputs).double()).detach().numpy()

        best_expert_obs_id = np.argmax(outputs)
        if outputs[best_expert_obs_id] < self.hyps["threshold"]:
            best_expert_obs_id = -5

        reward = best_expert_obs_id - len(self.C)
        reward /= len(self.C)
        return reward

    def train(self, epochs):
        train_data = self.replay_buffer.get_train_data()
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=50)

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs.double())
                loss = self.criterion(outputs, labels.double())
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()

            #print(f'[{epoch + 1}] loss: {running_loss / (i+1):.3f}')


class newEnvironment(gym.Env):

    def __init__(self, env, embedding, model_nr, train, alpha=0):
        self.env = env
        self.embedding = embedding
        self.model_nr = model_nr
        self.train = train
        self.alpha = alpha
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.t_obs = []
        self.t_rews = []

    def step(self, action):

        obs, _, done, _ = self.env.step(action)

        self.t_obs.append(obs)
        self.t_rews.append(self.embedding.reward(obs))

        reward = 0
        if len(self.t_rews) > 1:
            reward = self.t_rews[-1] - self.alpha * self.t_rews[-2]

        if done and self.train:
            self.embedding.replay_buffer.append(self.t_obs)
            self.embedding.train(1)
            folder = "./trained_models/" + self.embedding.domain + "/ours/"
            pickle.dump(self.embedding, open(folder + "embedding" + self.model_nr + ".p", "wb"))
            pickle.dump(self.embedding, open(folder + "embedding" + self.model_nr + "_backup.p", "wb"))

        return obs, reward, done, {}

    def reset(self):
        obs = self.env.reset()
        self.t_obs = [obs]
        self.t_rews = []
        return obs

    def render(self):
        self.env.render()




def train_ours(env, domain, model_nr, load=False, train=True, expert_data=None):

    folder = "./trained_models/" + domain + "/ours/"
    if not load:
        if domain == "mountain_car":
            hyps = config.mountaincar_hyperparameters
            embedding = Embedding(env, domain, expert_data, hyps)
        elif domain == "acrobot":
            hyps = config.acrobot_hyperparameters
            embedding = Embedding(env, domain, expert_data, hyps)
        elif domain == "lunarlander":
            hyps = config.lunarlander_hyperparameters
            embedding = Embedding(env, domain, expert_data, hyps)
        elif domain == "half_cheetah_discrete":
            hyps = config.halfcheetahdiscrete_hyperparameters
            embedding = Embedding(env, domain, expert_data, hyps)
        pickle.dump(embedding, open(folder + "embedding" + model_nr + ".p", "wb"))
    embedding = pickle.load(open(folder + "embedding" + model_nr + ".p", "rb"))

    if load:
        model_nr += "_e"

    newEnv = newEnvironment(env, embedding, model_nr, train=train, alpha=embedding.hyps["alpha"])
    callback = Logging.EvalCallback(env, newEnv, check_freq=10000, domain=domain, algo="ours", model_nr=model_nr)

    if domain == "mountain_car":
        model = DQN("MlpPolicy",
                    newEnv,
                    learning_rate=0.001,
                    batch_size=256,
                    buffer_size=30000,
                    learning_starts=10000,
                    gamma=0.99,
                    target_update_interval=600,
                    train_freq=16,
                    gradient_steps=8,
                    exploration_fraction=0.2,
                    exploration_final_eps=0.1,
                    policy_kwargs=dict(net_arch=[128, 128]),
                    verbose=0)
        model.learn(total_timesteps=100000, callback=callback)
    elif domain == "acrobot":
        model = PPO("MlpPolicy",
                    newEnv,
                    n_steps=256,
                    gamma=0.99,
                    gae_lambda=0.94,
                    n_epochs=4,
                    ent_coef=0.0,
                    verbose=0)
        model.learn(total_timesteps=60000, callback=callback)
    elif domain == "lunarlander":
        model = PPO("MlpPolicy",
                    newEnv,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=0.0001,
                    verbose=0)
        model.learn(total_timesteps=1000000, callback=callback)
    elif domain == "half_cheetah_discrete":
        model = PPO("MlpPolicy",
                    newEnv,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=0.0002,
                    verbose=0)
        model.learn(total_timesteps=1000000, callback=callback)



