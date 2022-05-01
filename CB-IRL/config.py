


# Mountain Car
mountaincar_hyperparameters = dict(
    tau=0.01,
    hidden_size=256,
    lr=0.003,
    expertWindowSize=2,
    agentWindowSize=20,
    agent_expert_divergence=0,
    mu=-0.1,
    n_trajectories=100,
    alpha=0,
)


# Acrobot
acrobot_hyperparameters = dict(
    tau=0.5,
    hidden_size=256,
    lr=0.0005,
    expertWindowSize=1,
    agentWindowSize=10,
    agent_expert_divergence=5,
    mu=-0.1,
    n_trajectories=100,
    alpha=0,
)


# Lunar Lander
lunarlander_hyperparameters = dict(
    tau=0.01,
    hidden_size=256,
    lr=0.0001,
    expertWindowSize=2,
    agentWindowSize=20,
    agent_expert_divergence=0,
    mu=-0.1,
    n_trajectories=200,
    alpha=1,
)


# Half Cheetah Discrete
halfcheetahdiscrete_hyperparameters = dict(
    tau=0.05,
    hidden_size=256,
    lr=0.0001,
    expertWindowSize=3,
    agentWindowSize=30,
    agent_expert_divergence=0,
    mu=-0.1,
    n_trajectories=100,
    alpha=1,
)





