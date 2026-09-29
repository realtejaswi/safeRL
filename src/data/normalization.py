import json
import numpy as np
from pathlib import Path

def compute_statistics(dataloader, max_batches=1000):
    state_sum, state_sq_sum, action_sum, action_sq_sum = 0, 0, 0, 0
    count = 0
    
    state_min, state_max = float('inf'), float('-inf')
    action_min, action_max = float('inf'), float('-inf')
    
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
            
        states = batch["state"].numpy()
        actions = batch["action"].numpy()
        
        batch_size = states.shape[0]
        count += batch_size
        
        state_sum += states.sum(axis=0)
        state_sq_sum += (states ** 2).sum(axis=0)
        action_sum += actions.sum(axis=0)
        action_sq_sum += (actions ** 2).sum(axis=0)
        
        state_min = np.minimum(state_min, states.min(axis=0))
        state_max = np.maximum(state_max, states.max(axis=0))
        action_min = np.minimum(action_min, actions.min(axis=0))
        action_max = np.maximum(action_max, actions.max(axis=0))

    if count == 0:
        return None
        
    state_mean = state_sum / count
    state_var = (state_sq_sum / count) - (state_mean ** 2)
    state_std = np.sqrt(np.maximum(state_var, 1e-8))
    
    action_mean = action_sum / count
    action_var = (action_sq_sum / count) - (action_mean ** 2)
    action_std = np.sqrt(np.maximum(action_var, 1e-8))
    
    return {
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "state_min": state_min.tolist(),
        "state_max": state_max.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "count": count
    }

def save_statistics(stats, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

def load_statistics(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def normalize(data, mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (data - mean) / std

def unnormalize(data, mean, std):
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return data * std + mean
