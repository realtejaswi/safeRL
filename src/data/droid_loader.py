from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def load_droid_dataset(config):
    dataset_name = config.get("name", "lerobot/droid_1.0.1")
    mode = config.get("mode", "stream")
    
    logger.info(f"Loading dataset: {dataset_name} in mode: {mode}")
    
    if mode == "stream":
        ds = load_dataset(dataset_name, streaming=True, split="train", trust_remote_code=True)
    elif mode == "local":
        ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
    elif mode == "demo":
        ds = load_dataset(dataset_name, streaming=True, split="train", trust_remote_code=True)
        ds = ds.take(100) # take a small sample
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")
        
    return ds
