import sys
import argparse
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_cmd(cmd):
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    
    python_exe = sys.executable
    
    if args.smoke_test or args.quick:
        run_cmd([python_exe, "scripts/setup_check.py"])
        run_cmd([python_exe, "scripts/train_bc.py", "--max-samples", "1000", "--epochs", "1"])
        run_cmd([python_exe, "scripts/train_rl.py", "--mode", "safe", "--timesteps", "1000"])
        run_cmd([python_exe, "scripts/evaluate.py", "--method", "safe", "--episodes", "2"])
        run_cmd([python_exe, "scripts/evaluate.py", "--all-methods"])
    else:
        run_cmd([python_exe, "scripts/setup_check.py"])
        run_cmd([python_exe, "scripts/train_bc.py"])
        run_cmd([python_exe, "scripts/train_rl.py", "--mode", "scratch"])
        run_cmd([python_exe, "scripts/train_rl.py", "--mode", "safe"])
        run_cmd([python_exe, "scripts/evaluate.py", "--method", "scratch", "--episodes", "50"])
        run_cmd([python_exe, "scripts/evaluate.py", "--method", "safe", "--episodes", "50"])
        run_cmd([python_exe, "scripts/evaluate.py", "--all-methods"])

if __name__ == "__main__":
    main()
