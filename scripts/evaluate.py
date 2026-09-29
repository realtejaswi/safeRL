import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import load_all_configs, setup_logging, set_seed
from src.evaluation import evaluate_robot, compare_methods

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["scratch", "residual", "safe"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--all-methods", action="store_true")
    args = parser.parse_args()
    
    logger = setup_logging()
    configs = load_all_configs()
    set_seed(configs.get("seed", 42))
    
    if args.all_methods:
        compare_methods()
    else:
        if not args.method:
            print("Please specify --method or --all-methods")
            return
        evaluate_robot(configs, mode=args.method, num_episodes=args.episodes)

if __name__ == "__main__":
    main()
