import sys
import platform
import torch

def print_section(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

def check_system():
    print_section("SYSTEM INFO")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version.split()[0]}")
    
    print("\nCUDA & PYTORCH")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA: Available")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA: Not available")
        
    print("\nPACKAGES")
    packages = ["mani_skill", "sapien", "gymnasium", "stable_baselines3", "datasets"]
    for pkg in packages:
        try:
            import importlib
            module = importlib.import_module(pkg)
            version = getattr(module, "__version__", "unknown")
            print(f"{pkg}: {version}")
        except ImportError:
            print(f"{pkg}: Not installed")

if __name__ == "__main__":
    check_system()
