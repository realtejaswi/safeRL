import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..data import load_droid_dataset, DroidIterableDataset
from ..models import BCPrior
from ..utils import get_device, get_artifact_dir
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)

def evaluate_bc(config, data_config, device=None):
    device = get_device(device)
    
    bc_checkpoint_path = get_artifact_dir("bc") / "best_model.pt"
    bc_stats_path = get_artifact_dir("bc") / "data_stats.json"
    
    if not (bc_checkpoint_path.exists() and bc_stats_path.exists()):
        logger.error("BC model or stats not found.")
        return
        
    prior = BCPrior(bc_checkpoint_path, bc_stats_path, device=device)
    
    logger.info("Loading test dataset for BC...")
    hf_dataset = load_droid_dataset(data_config)
    
    max_samples = config.get("max_samples", 1000)
    test_loader = DataLoader(DroidIterableDataset(hf_dataset, data_config, max_samples=max_samples, is_train=False), batch_size=256)
    
    mse_loss = 0
    mae_loss = 0
    count = 0
    
    all_preds = []
    all_targets = []
    
    criterion_mse = nn.MSELoss(reduction='sum')
    criterion_mae = nn.L1Loss(reduction='sum')
    
    with torch.no_grad():
        for batch in test_loader:
            states = batch["state"].numpy()
            actions = batch["action"].to(device)
            
            preds = []
            for state in states:
                preds.append(prior.get_action(state))
            preds = torch.tensor(np.array(preds), dtype=torch.float32).to(device)
            
            mse_loss += criterion_mse(preds, actions).item()
            mae_loss += criterion_mae(preds, actions).item()
            count += states.shape[0] * actions.shape[1]
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(actions.cpu().numpy())
            
    if count == 0:
        logger.warning("No samples evaluated.")
        return
        
    mse = mse_loss / count
    mae = mae_loss / count
    
    logger.info(f"Test samples: {count // 8}")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    plot_dir = get_artifact_dir("plots") / "bc"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(all_preds.shape[1]):
        plt.figure(figsize=(6, 6))
        plt.scatter(all_targets[:, i], all_preds[:, i], alpha=0.1)
        plt.plot([-1, 1], [-1, 1], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"Action Dimension {i}")
        plt.tight_layout()
        plt.savefig(plot_dir / f"action_{i}_scatter.png")
        plt.close()
        
    return {"mse": mse, "mae": mae}
