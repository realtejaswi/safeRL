import numpy as np

def generate_obstacles(np_random, num_obstacles, workspace_bounds):
    obstacles = []
    x_min, x_max, y_min, y_max, z_min, z_max = workspace_bounds
    
    for _ in range(num_obstacles):
        size = np_random.uniform(0.03, 0.10, size=3)
        # Avoid the center area where the robot and target might be
        while True:
            pos = np.array([
                np_random.uniform(x_min, x_max),
                np_random.uniform(y_min, y_max),
                np_random.uniform(z_min, z_max)
            ])
            # Keep away from origin
            if np.linalg.norm(pos[:2]) > 0.15:
                break
        
        obstacles.append({
            "position": pos,
            "size": size,
            "orientation": [1, 0, 0, 0] # quaternion identity for simplicity
        })
    return obstacles
