import numpy as np

def extract_state(example, config):
    if "observation.state" in example:
        return np.array(example["observation.state"], dtype=np.float32)
    elif "state" in example:
        return np.array(example["state"], dtype=np.float32)
    return np.zeros(15, dtype=np.float32)

def extract_action(example, config):
    action_repr = config.get("action_representation", "joint_delta")
    
    if "action" in example:
        action = np.array(example["action"], dtype=np.float32)
        if action_repr == "joint_delta" and len(action) >= 7:
            state = extract_state(example, config)
            if len(state) >= 7:
                action[:7] = action[:7] - state[:7]
        return action
        
    return np.zeros(8, dtype=np.float32)

def preprocess_example(example, config):
    state = extract_state(example, config)
    action = extract_action(example, config)
    episode_id = example.get("episode_index", 0)
    task = example.get("task", "")
    return {
        "state": state,
        "action": action,
        "episode_id": episode_id,
        "task": task
    }
