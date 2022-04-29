import numpy as np
import gym
import pickle

import CustomEnvs
import CB_IRL
import Expert



#env = CustomEnvs.MountainCar()
#domain = "mountain_car"

#env = CustomEnvs.Acrobot()
#domain = "acrobot"

#env = CustomEnvs.LunarLander()
#domain = "lunarlander"

env = CustomEnvs.DiscreteHalfCheetah(20)
domain = "half_cheetah_discrete"


#Expert.train_expert(env, domain)
#Expert.test_expert(env, domain)
#exit()


expert_data = [Expert.get_expert_trajectory(domain, deterministic=True) for _ in range(1)]
pr_policy = CB_IRL.train_ours(env, domain, model_nr="0", load=False, train=True, expert_data=expert_data)


