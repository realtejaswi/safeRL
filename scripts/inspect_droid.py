import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import load_droid_dataset
import logging

logging.basicConfig(level=logging.INFO)

def inspect():
    config = {"name": "lerobot/droid_1.0.1", "mode": "demo"}
    print("Loading dataset (streaming first 100 samples)...")
    ds = load_droid_dataset(config)
    
    print("\nDataset Info:")
    if hasattr(ds, "info"):
        print(f"Dataset name: {ds.info.dataset_name}")
        print(f"Description: {ds.info.description}")
        print(f"Features: {ds.info.features}")
    
    print("\nFirst sample features:")
    try:
        sample = next(iter(ds))
        for k, v in sample.items():
            if isinstance(v, (list, tuple)):
                print(f"{k}: list of length {len(v)}")
            else:
                print(f"{k}: {type(v)}")
    except Exception as e:
        print(f"Could not load sample: {e}")
            
    print("\nThis dataset streams from HF and contains robotic demonstrations from the DROID dataset.")

if __name__ == "__main__":
    inspect()
