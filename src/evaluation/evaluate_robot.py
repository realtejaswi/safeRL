import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
from ..utils import get_device, get_artifact_dir
from ..models import BCPrior
import logging
from ..rl.wrappers import VecResidualActionWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger(__name__)

def evaluate_robot(config, mode="safe", num_episodes=50, render=False):
    device = get_device(config.get("device", "auto"))
    
    import mani_skill.envs
    from ..envs.safe_pick_cube import SafePickCubeWrapper
    
    def make_env():
        render_mode = "rgb_array" if render else None
        base_env = gym.make("PickCube-v1", obs_mode="state", render_mode=render_mode)
        env = SafePickCubeWrapper(base_env, config=config)
        return env
        
    env = DummyVecEnv([make_env])
    
    bc_prior = None
    if mode in ["residual", "safe"]:
        bc_checkpoint_path = get_artifact_dir("bc") / "best_model.pt"
        bc_stats_path = get_artifact_dir("bc") / "data_stats.json"
        
        if bc_checkpoint_path.exists() and bc_stats_path.exists():
            bc_prior = BCPrior(bc_checkpoint_path, bc_stats_path, device=device)
            env = VecResidualActionWrapper(env, bc_prior, config.get("rl", {}).get("residual_scale", 0.25))
    
    model_path = get_artifact_dir("rl") / f"ppo_{mode}_model.zip"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return None
        
    model = PPO.load(model_path, env=env, device=device)
    
    results = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        ep_reward = 0
        min_clearance = float('inf')
        collisions = 0
        path_length = 0
        energy = 0
        
        while not done and step_count < config.get("termination", {}).get("max_steps", 200):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info_lst = env.step(action)
            info = info_lst[0]
            
            ep_reward += reward[0]
            min_clearance = min(min_clearance, info.get("clearance", 1.0))
            if info.get("is_collision", False):
                collisions += 1
                
            energy += float(np.sum(action[0]**2))
            step_count += 1
            
        success = bool(info.get("success", False))
        
        results.append({
            "episode": ep,
            "success": success,
            "reward": float(ep_reward),
            "length": step_count,
            "min_clearance": float(min_clearance),
            "collisions": int(collisions),
            "energy_proxy": float(energy)
        })
        
    df = pd.DataFrame(results)
    
    summary = {
        "method": mode,
        "success_rate": float(df["success"].mean()),
        "collision_rate": float((df["collisions"] > 0).mean()),
        "mean_reward": float(df["reward"].mean()),
        "std_reward": float(df["reward"].std()),
        "mean_episode_length": float(df["length"].mean()),
        "mean_min_clearance": float(df["min_clearance"].mean()),
        "mean_energy_proxy": float(df["energy_proxy"].mean())
    }
    
    logger.info(f"Evaluation results for {mode}:\n{summary}")
    
    results_dir = get_artifact_dir("results")
    df.to_csv(results_dir / f"eval_{mode}.csv", index=False)
    
    import json
    with open(results_dir / f"summary_{mode}.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary
