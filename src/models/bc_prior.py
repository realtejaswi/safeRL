import torch
import numpy as np
from .behavior_cloning import BehaviorCloningModel
from ..data.normalization import normalize

class BCPrior:
    def __init__(self, checkpoint_path, stats_path, device="cpu"):
        self.device = device
        
        import json
        with open(stats_path, "r") as f:
            self.stats = json.load(f)
            
        self.state_mean = self.stats["state_mean"]
        self.state_std = self.stats["state_std"]
        
        self.model = BehaviorCloningModel(
            input_dim=len(self.state_mean), 
            output_dim=len(self.stats.get("action_mean", [0]*8))
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def get_action(self, obs):
        state = obs[:len(self.state_mean)]
        norm_state = normalize(state, self.state_mean, self.state_std)
        
        with torch.no_grad():
            tensor_state = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.model(tensor_state)
            
        return action.squeeze(0).cpu().numpy()
