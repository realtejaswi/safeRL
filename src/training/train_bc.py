import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..data import load_droid_dataset, DroidIterableDataset, compute_statistics, save_statistics
from ..models import BehaviorCloningModel, save_checkpoint
from ..utils import get_device, get_artifact_dir
import logging

logger = logging.getLogger(__name__)

def train_bc(config, data_config, device=None):
    if device is None:
        device = get_device()
        
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 256)
    lr = config.get("learning_rate", 1e-3)
    wd = config.get("weight_decay", 1e-4)
    max_samples = config.get("max_samples", None)
    
    logger.info("Loading dataset for BC...")
    hf_dataset = load_droid_dataset(data_config)
    
    train_dataset = DroidIterableDataset(hf_dataset, data_config, max_samples=max_samples, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    
    logger.info("Computing normalization statistics...")
    stats = compute_statistics(train_loader, max_batches=100) # smaller for stat compute
    if not stats:
        logger.warning("Could not compute statistics (empty dataset or zero samples), using dummy stats.")
        stats = {
            "state_mean": [0.0]*15,
            "state_std": [1.0]*15,
            "action_mean": [0.0]*8,
            "action_std": [1.0]*8
        }
        
    stats_dir = get_artifact_dir("bc")
    stats_path = stats_dir / "data_stats.json"
    save_statistics(stats, stats_path)
    
    state_dim = len(stats["state_mean"])
    action_dim = len(stats["action_mean"])
    
    model = BehaviorCloningModel(input_dim=state_dim, output_dim=action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_steps = 0
        
        train_loader = DataLoader(DroidIterableDataset(hf_dataset, data_config, max_samples=max_samples, is_train=True), batch_size=batch_size)
        
        for batch in train_loader:
            states = batch["state"].to(device)
            actions = batch["action"].to(device)
            
            states = (states - torch.tensor(stats["state_mean"]).to(device)) / torch.tensor(stats["state_std"]).to(device)
            
            optimizer.zero_grad()
            preds = model(states)
            loss = criterion(preds, actions)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
        avg_train_loss = train_loss / max(1, train_steps)
        
        model.eval()
        val_loss = 0
        val_steps = 0
        val_loader = DataLoader(DroidIterableDataset(hf_dataset, data_config, max_samples=max_samples // 10 if max_samples else None, is_train=False), batch_size=batch_size)
        with torch.no_grad():
            for batch in val_loader:
                states = batch["state"].to(device)
                actions = batch["action"].to(device)
                states = (states - torch.tensor(stats["state_mean"]).to(device)) / torch.tensor(stats["state_std"]).to(device)
                preds = model(states)
                loss = criterion(preds, actions)
                val_loss += loss.item()
                val_steps += 1
                
        avg_val_loss = val_loss / max(1, val_steps)
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        run_stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "input_dim": state_dim,
            "output_dim": action_dim,
            "dataset_name": data_config.get("name")
        }
        
        save_checkpoint(model, optimizer, run_stats, stats_dir / "last_model.pt")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, run_stats, stats_dir / "best_model.pt")
            
    return model
