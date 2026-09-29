import numpy as np

def compute_clearance(ee_pos, cube_pos, obstacles):
    """Compute the minimum clearance to any obstacle."""
    min_clearance = float('inf')
    if not obstacles:
        return 1.0 # large arbitrary clearance
        
    for obs in obstacles:
        obs_pos = obs['position']
        # simple bounding box or sphere check proxy
        dist_ee = np.linalg.norm(ee_pos - obs_pos)
        dist_cube = np.linalg.norm(cube_pos - obs_pos)
        min_clearance = min(min_clearance, dist_ee, dist_cube)
        
    return min_clearance

def check_collision(ee_pos, cube_pos, obstacles, threshold=0.04):
    """Simple collision proxy using clearance threshold."""
    clearance = compute_clearance(ee_pos, cube_pos, obstacles)
    return clearance < threshold
