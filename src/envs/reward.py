import numpy as np

def compute_reward(config, progress, is_success, is_grasped, is_collision, clearance, action_magnitude, action_diff):
    reward = 0.0
    
    # Task progress
    reward += config.get("progress", 5.0) * progress
    
    if is_grasped:
        reward += config.get("grasp", 10.0)
        
    if is_success:
        reward += config.get("success", 100.0)
        
    # Safety
    if is_collision:
        reward -= config.get("collision", 50.0)
        
    safe_dist = config.get("safe_distance", 0.08)
    if clearance < safe_dist:
        penalty = config.get("clearance", 5.0) * (safe_dist - clearance)
        reward -= penalty
        
    # Efficiency
    reward -= config.get("energy", 0.01) * action_magnitude
    reward -= config.get("smoothness", 0.01) * action_diff
    
    # Time penalty
    reward -= config.get("time", 0.01)
    
    return reward
