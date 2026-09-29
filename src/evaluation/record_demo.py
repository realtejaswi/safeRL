import gymnasium as gym
from stable_baselines3 import PPO
from ..utils import get_device, get_artifact_dir
from ..models import BCPrior
import logging
from ..rl.wrappers import ResidualActionWrapper
import cv2

logger = logging.getLogger(__name__)

def record_demo(config, mode="safe"):
    device = get_device(config.get("device", "auto"))
    
    import mani_skill.envs
    from ..envs.safe_pick_cube import SafePickCubeWrapper
    
    base_env = gym.make("PickCube-v1", obs_mode="state", render_mode="rgb_array")
    env = SafePickCubeWrapper(base_env, config=config)
    
    if mode in ["residual", "safe"]:
        bc_checkpoint_path = get_artifact_dir("bc") / "best_model.pt"
        bc_stats_path = get_artifact_dir("bc") / "data_stats.json"
        
        if bc_checkpoint_path.exists() and bc_stats_path.exists():
            bc_prior = BCPrior(bc_checkpoint_path, bc_stats_path, device=device)
            env = ResidualActionWrapper(env, bc_prior, config.get("rl", {}).get("residual_scale", 0.25))
    
    model_path = get_artifact_dir("rl") / f"ppo_{mode}_model.zip"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
        
    model = PPO.load(model_path, env=env, device=device)
    
    obs, info = env.reset()
    done = False
    step = 0
    frames = []
    
    while not done and step < config.get("termination", {}).get("max_steps", 200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        frame = env.render()
        frames.append(frame)
        step += 1
        
    if not frames:
        return
        
    video_dir = get_artifact_dir("videos")
    video_path = video_dir / f"demo_{mode}.mp4"
    
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
    
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
    video.release()
    logger.info(f"Video saved to {video_path}")
