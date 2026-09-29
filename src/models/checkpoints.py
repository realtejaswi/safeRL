import torch
import json
from pathlib import Path

def save_checkpoint(model, optimizer, stats, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'stats': stats
    }
    torch.save(checkpoint, path)
    
    meta_path = path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(stats, f, indent=2)

def load_checkpoint(model, optimizer, path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('stats', {})
