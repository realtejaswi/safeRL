import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv

class ResidualActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bc_prior, residual_scale=0.25):
        super().__init__(env)
        self.bc_prior = bc_prior
        self.residual_scale = residual_scale
        self.last_obs = None
        
    def step(self, action):
        if self.last_obs is None:
            bc_action = np.zeros_like(action)
        else:
            bc_action = self.bc_prior.get_action(self.last_obs)
            
        final_action = bc_action + self.residual_scale * action
        final_action = np.clip(final_action, self.action_space.low, self.action_space.high)
        
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        self.last_obs = obs
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

class VecResidualActionWrapper(VecEnvWrapper):
    def __init__(self, venv, bc_prior, residual_scale=0.25):
        super().__init__(venv)
        self.bc_prior = bc_prior
        self.residual_scale = residual_scale
        self.last_obs = None
        
    def reset(self):
        obs = self.venv.reset()
        self.last_obs = obs
        return obs
        
    def step_async(self, actions):
        if self.last_obs is None:
            bc_actions = np.zeros_like(actions)
        else:
            # Batch inference
            bc_actions = np.array([self.bc_prior.get_action(o) for o in self.last_obs])
            
        final_actions = bc_actions + self.residual_scale * actions
        final_actions = np.clip(final_actions, self.action_space.low, self.action_space.high)
        self.venv.step_async(final_actions)
        
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.last_obs = obs
        return obs, rewards, dones, infos
