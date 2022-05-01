from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

import utils



class EvalCallback(BaseCallback):

    def __init__(self, env, newEnv, check_freq, domain, algo, model_nr):
        super(EvalCallback, self).__init__(0)
        self.env = env
        self.newEnv = newEnv
        self.check_freq = check_freq
        self.domain = domain
        self.file = "./trained_models/" + domain + "/" + algo + "/" + model_nr + "/model_"

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.model.save(self.file + str(self.n_calls))

            rews = utils.run_episode(self.env, self.model, domain=self.domain)
            print("Env, step ", self.n_calls, ":  ", np.sum(rews), "   - Len: ", len(rews))
            self.env.reset()

            rews = utils.run_episode(self.newEnv, self.model, domain=self.domain)
            print("     NewEnv, step ", self.n_calls, ":  ", np.sum(rews), "   - Len: ", len(rews))
            self.newEnv.reset()



