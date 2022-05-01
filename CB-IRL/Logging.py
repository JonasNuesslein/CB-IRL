from stable_baselines3.common.callbacks import BaseCallback
import numpy as np



class EvalCallback(BaseCallback):

    def __init__(self, env, newEnv, check_freq):
        super(EvalCallback, self).__init__(0)
        self.env = env
        self.newEnv = newEnv
        self.check_freq = check_freq

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            rews = run_episode(self.env, self.model)
            print("Env, step ", self.n_calls, ":  ", np.sum(rews), "   - Len: ", len(rews))
            self.env.reset()
            rews = run_episode(self.newEnv, self.model)
            print("     NewEnv, step ", self.n_calls, ":  ", np.sum(rews), "   - Len: ", len(rews))
            self.newEnv.reset()



def run_episode(env, model=None, print_reward=False, render=False, seed=None):
    env.seed(seed)
    obs, rews, done = env.reset(), [], False
    while not done:
        if model == None:
            a = env.action_space.sample()
        else:
            a, _ = model.predict(obs, deterministic=True)
        obs, r, done, _ = env.step(a)
        rews.append(r)
        if render:
            env.render()
        if print_reward:
            print("reward[", len(rews),"]: ", r)
    return rews
