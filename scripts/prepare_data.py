import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import load_all_configs, setup_logging
from src.data import load_droid_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=50000)
    args = parser.parse_args()
    
    logger = setup_logging()
    configs = load_all_configs()
    
    config = configs.get("dataset", {})
    config["mode"] = "stream"
    
    logger.info(f"Preparing dataset with max samples: {args.max_samples}")
    ds = load_droid_dataset(config)
    
    count = 0
    for _ in ds:
        count += 1
        if count >= args.max_samples:
            break
            
    logger.info(f"Successfully streamed {count} samples.")
    logger.info("Data preparation completed. The stream works locally.")

if __name__ == "__main__":
    main()
