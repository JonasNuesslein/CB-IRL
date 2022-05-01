import numpy as np



class ReplayBuffer():

    def __init__(self, expert_data, hyps):
        super(ReplayBuffer, self).__init__()
        self.expert_data = expert_data
        self.agent_data = []
        self.hyps = hyps

    def append(self, trajectory):
        self.agent_data.append(trajectory)
        self.agent_data = self.agent_data[-self.hyps["n_trajectories"]:]

    def get_train_data(self):
        train_data = []

        if len(self.agent_data) >= 2:

            # convergence of agent data
            for _ in range(75):
                t_id = np.random.choice(len(self.agent_data))
                o1 = np.random.choice(len(self.agent_data[t_id])-self.hyps["agentWindowSize"])
                o2 = o1 + np.random.choice(self.hyps["agentWindowSize"])
                x = np.concatenate((self.agent_data[t_id][o1], self.agent_data[t_id][o2]))
                x2 = np.concatenate((self.agent_data[t_id][o2], self.agent_data[t_id][o1]))
                y = np.array([1])
                train_data.append([x,y])
                train_data.append([x2,y])

            # divergence of agent data
            for _ in range(150):
                t_ids = np.random.choice(len(self.agent_data), 2, replace=False)
                o1 = np.random.choice(len(self.agent_data[t_ids[0]]))
                o2 = np.random.choice(len(self.agent_data[t_ids[1]]))
                x = np.concatenate((self.agent_data[t_ids[0]][o1], self.agent_data[t_ids[1]][o2]))
                y = np.array([0])
                train_data.append([x,y])

            # divergence of agent and expert data
            for _ in range(self.hyps["agent_expert_divergence"]):
                t_id = np.random.choice(len(self.agent_data))
                t_id2 = np.random.choice(len(self.expert_data))
                o1 = np.random.choice(len(self.agent_data[t_id]))
                o2 = np.random.choice(len(self.expert_data))
                x = np.concatenate((self.agent_data[t_id][o1], self.expert_data[t_id2][o2]))
                y = np.array([0])
                train_data.append([x,y])

        # convergence of expert data
        for _ in range(150):
            t_id = np.random.choice(len(self.expert_data))
            o1 = np.random.choice(len(self.expert_data[t_id])-self.hyps["expertWindowSize"])
            o2 = o1 + np.random.choice(self.hyps["expertWindowSize"])
            x = np.concatenate((self.expert_data[t_id][o1], self.expert_data[t_id][o2]))
            y = np.array([1])
            train_data.append([x,y])

        # divergence of expert data
        for _ in range(150):
            t_id = np.random.choice(len(self.expert_data))
            ids = np.random.choice(len(self.expert_data[t_id]), 2)
            if np.abs(ids[0] - ids[1]) < self.hyps["expertWindowSize"]:
                continue
            x = np.concatenate((self.expert_data[t_id][ids[0]], self.expert_data[t_id][ids[1]]))
            y = np.array([0])
            train_data.append([x,y])

        return train_data



