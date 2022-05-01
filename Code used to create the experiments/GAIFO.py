import gym
import pickle
import numpy as np
from tf2rl.algos.dqn import DQN
from tf2rl.algos.gaifo import GAIfO
from tf2rl.experiments.irl_trainer import IRLTrainer
from tf2rl.experiments.utils import restore_latest_n_traj
import os
import sys

import CustomEnvs


def create_expert_data(domain):
    import Expert
    expert_data = [Expert.get_expert_trajectory(domain, deterministic=True) for _ in range(1)]
    obses = expert_data[0][0]
    next_obses = obses[1:]
    acts = expert_data[0][1]
    data = (obses, next_obses, acts)
    pickle.dump(data, open("data_" + domain + ".p", "wb"))


def train(env, test_env, domain, steps, model_nr):
    parser = IRLTrainer.get_argument()
    parser = GAIfO.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="x")
    parser.add_argument('--max-steps', type=int, default=steps)
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--dir-suffix', type=str, default=domain + str(model_nr))
    args = parser.parse_args()

    units = [256, 256]

    if domain == "acrobot":
        policy = DQN(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.n,
            lr=0.0001,
            units=units,)
        irl = GAIfO(
            state_shape=env.observation_space.shape,
            units=units,
            lr=0.0000001,
            batch_size=64,
            gpu=-1)
    elif domain == "mountain_car":
        policy = DQN(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.n,
            lr=0.001,
            units=units,)
        irl = GAIfO(
            state_shape=env.observation_space.shape,
            units=units,
            lr=0.00001,
            batch_size=64,
            gpu=-1)
    elif domain == "lunarlander":
        policy = DQN(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.n,
            lr=0.001,
            units=units,)
        irl = GAIfO(
            state_shape=env.observation_space.shape,
            units=units,
            lr=0.00001,
            batch_size=64,
            gpu=-1)
    elif domain == "half_cheetah_discrete":
        policy = DQN(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.n,
            lr=0.001,
            units=units,)
        irl = GAIfO(
            state_shape=env.observation_space.shape,
            units=units,
            lr=0.00001,
            batch_size=64,
            gpu=-1)

    obses, next_obses, acts = pickle.load(open("results/data_" + domain + ".p", "rb"))
    obses = obses[:-1]

    trainer = IRLTrainer(policy, env, args, irl, obses, next_obses, acts, test_env)
    trainer()


def load_policy(env, test_env, domain, model_nr, ckpt_nr):

    folder = "results/" + domain + str(model_nr) + "/"
    os.rename(folder + "checkpoint", folder + "checkpoint_old")
    old_file = open(folder + "checkpoint_old", 'r')
    new_file = open(folder + "checkpoint", 'w')
    Lines = old_file.readlines()
    new_file.write("model_checkpoint_path: \"ckpt-" + str(ckpt_nr) + "\"\n")
    for line in Lines[1:]:
        new_file.write(line)
    old_file.close()
    new_file.close()
    os.remove(folder + "checkpoint_old")

    parser = IRLTrainer.get_argument()
    parser = GAIfO.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="x")
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--model-dir', type=str, default=folder)
    args = parser.parse_args()

    units = [256, 256]

    policy = DQN(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        units=units)
    irl = GAIfO(
        state_shape=env.observation_space.shape,
        units=units,
        batch_size=64,
        gpu=-1)

    obses, next_obses, acts = pickle.load(open("results/data_" + domain + ".p", "rb"))
    obses = obses[:-1]

    trainer = IRLTrainer(policy, env, args, irl, obses, next_obses, acts, test_env)
    trainer()
    return policy


def run_episode(env, policy):
    o = env.reset()
    done = False
    rews = []
    while not done:
        a = policy.get_action(o)
        o, r, done, _ = env.step(a)
        rews.append(r)
    return rews


#env = CustomEnvs.Acrobot()
#test_env = gym.make("Acrobot-v1")
#domain = "acrobot"
#steps = 60000

#env = CustomEnvs.MountainCar()
#test_env = gym.make("MountainCar-v0")
#domain = "mountain_car"
#steps = 100000

#env = CustomEnvs.DiscreteHalfCheetah(20)
#test_env = CustomEnvs.DiscreteHalfCheetah(20)
#domain = "half_cheetah_discrete"
#steps = 1000000

#env = CustomEnvs.LunarLander()
#test_env = gym.make("LunarLander-v2")
#domain = "lunarlander"
#steps = 1000000

#model_nr = int(sys.argv[1][-1])
#train(env, test_env, domain, steps, model_nr=3)



