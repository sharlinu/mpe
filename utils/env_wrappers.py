"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from baselines.common.vec_env import VecEnv

class WrapperEnv(VecEnv):
    def __init__(self, env_obj, env_id):

        self.__class__ = type(env_obj.__class__.__name__,
                              (self.__class__,env_obj.__class__),
                              {})
        self.__dict__ = env_obj.__dict__

        self.env = env_obj
        self.env_id = env_id

        if all([hasattr(a, 'adversary') for a in self.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                self.agents]
        else:
            self.agent_types = ['agent' for _ in self.env.agents]

        self.ts = np.zeros(1, dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs, rew, done, info = self.env.step(self.actions[0])

        if self.env_id == 'waterworld':
            done = [False, False]

        if all(done): 
            obs = self.env.reset()
            self.ts += 0
        self.actions = None

        obs  = np.array(obs)
        rew  = np.array(rew)
        done = np.array(done)
        info = np.array(info)
        for i in range(obs.shape[0]):
            obs[i] = np.expand_dims(obs[i],axis=0)
            rew[i] = np.expand_dims(rew[i],axis=0)
            done[i] = np.expand_dims(done[i],axis=0)
        obs = np.expand_dims(obs,axis=0)
        rew = np.expand_dims(rew,axis=0)
        done = np.expand_dims(done,axis=0)

        return obs, np.array(rew), np.array(done), info

    def reset(self):    
        return np.expand_dims(np.array(self.env.reset()),axis=0)

    def close(self):
        return