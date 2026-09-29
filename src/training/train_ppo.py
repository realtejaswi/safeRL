import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
from ..utils import get_device, get_artifact_dir
from ..models import BCPrior
from .wrappers import VecResidualActionWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

logger = logging.getLogger(__name__)

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # We can log safe metrics from info dict if present
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            if "clearance" in info:
                self.logger.record("safety/clearance", info["clearance"])
            if "is_collision" in info:
                self.logger.record("safety/is_collision", float(info["is_collision"]))
        return True

def get_callbacks(eval_env, save_dir, mode):
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / f"best_model_{mode}"),
        log_path=str(save_dir / f"eval_logs_{mode}"),
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    return [eval_callback, MetricsCallback()]

def train_ppo(config, mode="scratch", device=None):
    if device is None:
        device = get_device(config.get("device", "auto"))
        
    rl_config = config.get("rl", {})
    algo_config = rl_config.get("algorithm", {})
    total_timesteps = algo_config.get("total_timesteps", 200000)
    residual_scale = rl_config.get("residual_scale", 0.25)
    
    # Needs to be imported inside so mani_skill is loaded
    import mani_skill.envs
    from ..envs.safe_pick_cube import SafePickCubeWrapper
    
    def make_env():
        base_env = gym.make("PickCube-v1", obs_mode="state", render_mode="rgb_array")
        env = SafePickCubeWrapper(base_env, config=config)
        return env
        
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    bc_prior = None
    if mode in ["residual", "safe"]:
        bc_checkpoint_path = get_artifact_dir("bc") / "best_model.pt"
        bc_stats_path = get_artifact_dir("bc") / "data_stats.json"
        
        if bc_checkpoint_path.exists() and bc_stats_path.exists():
            bc_prior = BCPrior(bc_checkpoint_path, bc_stats_path, device=device)
            env = VecResidualActionWrapper(env, bc_prior, residual_scale)
            eval_env = VecResidualActionWrapper(eval_env, bc_prior, residual_scale)
        else:
            logger.warning(f"BC prior not found for mode '{mode}'. Falling back to scratch.")
            mode = "scratch"
            
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=algo_config.get("learning_rate", 3e-4),
        n_steps=algo_config.get("n_steps", 1024),
        batch_size=algo_config.get("batch_size", 256),
        gamma=algo_config.get("gamma", 0.99),
        gae_lambda=algo_config.get("gae_lambda", 0.95),
        clip_range=algo_config.get("clip_range", 0.2),
        ent_coef=algo_config.get("ent_coef", 0.01),
        vf_coef=algo_config.get("vf_coef", 0.5),
        max_grad_norm=algo_config.get("max_grad_norm", 0.5),
        tensorboard_log=str(get_artifact_dir("logs")),
        device=device,
        verbose=1
    )
    
    callbacks = get_callbacks(eval_env, get_artifact_dir("rl"), mode)
    
    logger.info(f"Starting PPO training in {mode} mode for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    
    model_path = get_artifact_dir("rl") / f"ppo_{mode}_model.zip"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model
