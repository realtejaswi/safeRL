import numpy as np

def flatten_observation(obs_dict):
    """
    Convert dictionary observation to flattened array.
    obs_dict typically comes from ManiSkill.
    """
    parts = []
    
    if "agent" in obs_dict:
        if "qpos" in obs_dict["agent"]:
            parts.append(obs_dict["agent"]["qpos"])
        if "qvel" in obs_dict["agent"]:
            parts.append(obs_dict["agent"]["qvel"])
            
    if "extra" in obs_dict:
        if "tcp_pose" in obs_dict["extra"]:
            parts.append(obs_dict["extra"]["tcp_pose"])
        if "obj_pose" in obs_dict["extra"]:
            parts.append(obs_dict["extra"]["obj_pose"])
        if "goal_pos" in obs_dict["extra"]:
            parts.append(obs_dict["extra"]["goal_pos"])
            
            if "obj_pose" in obs_dict["extra"]:
                cube_pos = obs_dict["extra"]["obj_pose"][:3]
                goal_pos = obs_dict["extra"]["goal_pos"]
                cube_to_goal = goal_pos - cube_pos
                parts.append(cube_to_goal)
    
    if not parts:
        # Fallback for unexpected format
        parts.append(np.zeros(15))
        
    return np.concatenate([np.ravel(p) for p in parts])
