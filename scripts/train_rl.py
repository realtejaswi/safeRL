import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import load_all_configs, setup_logging, set_seed
from src.training.train_ppo import train_ppo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["scratch", "residual", "safe"], default="scratch")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()
    
    logger = setup_logging()
    configs = load_all_configs()
    set_seed(configs.get("seed", 42))
    
    if args.timesteps is not None:
        if "rl" not in configs:
            configs["rl"] = {"algorithm": {}}
        configs["rl"]["algorithm"]["total_timesteps"] = args.timesteps
        
    train_ppo(configs, mode=args.mode)

if __name__ == "__main__":
    main()
