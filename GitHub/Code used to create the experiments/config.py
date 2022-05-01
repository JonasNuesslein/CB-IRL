

# Hyperparameter of CB-IRL:
#    - threshold  E  [0;1]       +++ Then to consider two states as similar
#    - hidden-size and lr
#    - frame_size                +++ States within a frame are considered similar and unsimilar otherwise
#    - agent_expert_divergence   +++ wether (and if yes how often) to discrimiate between agent and expert data
#    - reward_frame_size         +++ average the reward within the reward_frame_size
#    - n_trajectories            +++ Number of trajectories in the replay buffer
#    - reward_style              +++ reward_style=0 -> max reward so far; reward_style=1 -> reward-difference


# Mountain Car
mountaincar_hyperparameters = dict(
    threshold=0.01,
    hidden_size=256,
    lr=0.003,
    frame_size=20,
    agent_expert_divergence=0,
    reward_frame_size=1,
    n_trajectories=100,
    reward_style=0,
)


# Acrobot
acrobot_hyperparameters = dict(
    threshold=0.5,
    hidden_size=256,
    lr=0.0005,
    frame_size=10,
    agent_expert_divergence=5,
    reward_frame_size=1,
    n_trajectories=100,
    reward_style=0,
)


# Lunar Lander
lunarlander_hyperparameters = dict(
    threshold=0.01,
    hidden_size=256,
    lr=0.0001,
    frame_size=20,
    agent_expert_divergence=0,
    reward_frame_size=1,
    n_trajectories=200,
    reward_style=1,
)


# Half Cheetah Discrete
halfcheetahdiscrete_hyperparameters = dict(
    threshold=0.05,
    hidden_size=256,
    lr=0.0001,
    frame_size=30,
    agent_expert_divergence=20, #20 for complete
    reward_frame_size=1,
    n_trajectories=100,
    reward_style=1,
)





