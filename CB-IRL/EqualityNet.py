import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ReplayBuffer



class EqualityNetModule(nn.Module):

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



class EqualityNet():

    def __init__(self, env, expert_data, hyps):
        self.expert_data = expert_data
        self.hyps = hyps

        self.replay_buffer = ReplayBuffer.ReplayBuffer(self.expert_data, hyps)
        self.model = EqualityNetModule(env, self.hyps["hidden_size"])

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyps["lr"])
        self.train(100)

    def reward(self, o):
        mostSimilar = self.hyps["mu"]
        similarity = self.hyps["tau"]

        for expertTrajectory in self.expert_data:
            inputs = []
            for expert_o in expertTrajectory:
                inputs.append(np.concatenate((o, expert_o)))
            outputs = self.model(torch.tensor(np.array(inputs)).double()).detach().numpy()
            best_expert_o_id = np.argmax(outputs)
            if outputs[best_expert_o_id] > similarity:
                mostSimilar = best_expert_o_id / len(expertTrajectory)
                similarity = outputs[best_expert_o_id]

        return mostSimilar

    def train(self, epochs):
        train_data = self.replay_buffer.get_train_data()
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=50)

        for _ in range(epochs):  # loop over the dataset multiple times

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

            # print(f'[{epoch + 1}] loss: {running_loss / (i+1):.3f}')


