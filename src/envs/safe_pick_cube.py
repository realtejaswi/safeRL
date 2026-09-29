import gymnasium as gym
import numpy as np
import torch
from .obstacle_generator import generate_obstacles
from .collision import compute_clearance, check_collision
from .reward import compute_reward
from .observation import flatten_observation

class SafePickCubeWrapper(gym.Wrapper):
    def __init__(self, env, config=None):
        super().__init__(env)
        if config is None:
            self.config = {
                "reward": {
                    "success": 100.0,
                    "grasp": 10.0,
                    "progress": 5.0,
                    "collision": 50.0,
                    "clearance": 5.0,
                    "energy": 0.01,
                    "smoothness": 0.01,
                    "time": 0.01,
                    "safe_distance": 0.08
                },
                "obstacles": {
                    "enabled": True,
                    "min_count": 1,
                    "max_count": 5
                },
                "termination": {
                    "terminate_on_collision": True
                }
            }
        else:
            self.config = config
            
        self.obstacles = []
        self.last_action = np.zeros(self.action_space.shape)
        
        # Dummy shape, will be updated on reset
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if self.config.get("obstacles", {}).get("enabled", True):
            num_obstacles = np.random.randint(
                self.config["obstacles"]["min_count"],
                self.config["obstacles"]["max_count"] + 1
            )
            workspace = [-0.3, 0.3, -0.3, 0.3, 0.0, 0.3]
            self.obstacles = generate_obstacles(np.random, num_obstacles, workspace)
        else:
            self.obstacles = []
            
        self.last_action = np.zeros(self.action_space.shape)
        flat_obs = flatten_observation(obs)
        
        if self.observation_space.shape[0] != flat_obs.shape[0]:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=flat_obs.shape, dtype=np.float32)
            
        return flat_obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        try:
            # SAPIEN/ManiSkill 3 positions are usually batched tensors
            ee_pos_t = self.unwrapped.tcp.pose.p
            cube_pos_t = self.unwrapped.obj.pose.p
            
            if isinstance(ee_pos_t, torch.Tensor):
                ee_pos = ee_pos_t[0].detach().cpu().numpy()
                cube_pos = cube_pos_t[0].detach().cpu().numpy()
            else:
                ee_pos = ee_pos_t[0]
                cube_pos = cube_pos_t[0]
        except Exception:
            ee_pos = np.zeros(3)
            cube_pos = np.zeros(3)
            
        clearance = compute_clearance(ee_pos, cube_pos, self.obstacles)
        is_collision = check_collision(ee_pos, cube_pos, self.obstacles)
        
        info["clearance"] = float(clearance)
        info["is_collision"] = bool(is_collision)
        
        action_magnitude = float(np.sum(action ** 2))
        action_diff = float(np.sum((action - self.last_action) ** 2))
        self.last_action = action.copy()
        
        progress = float(info.get("progress", 0.0))
        is_success = bool(info.get("success", False))
        is_grasped = bool(info.get("is_grasped", False))
        
        reward = compute_reward(
            self.config.get("reward", {}),
            progress,
            is_success,
            is_grasped,
            is_collision,
            clearance,
            action_magnitude,
            action_diff
        )
        
        if is_collision and self.config.get("termination", {}).get("terminate_on_collision", True):
            terminated = True
            
        return flatten_observation(obs), reward, terminated, truncated, info
