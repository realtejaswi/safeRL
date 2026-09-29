import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import load_all_configs, setup_logging, set_seed
from src.training import train_bc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    logger = setup_logging()
    configs = load_all_configs()
    set_seed(configs.get("seed", 42))
    
    bc_config = configs.get("bc", {})
    if args.max_samples is not None:
        bc_config["max_samples"] = args.max_samples
    if args.epochs is not None:
        bc_config["epochs"] = args.epochs
        
    logger.info("Starting BC training")
    train_bc(bc_config, configs.get("dataset", {}))

if __name__ == "__main__":
    main()
