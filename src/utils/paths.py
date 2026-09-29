from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

def get_artifact_dir(subdir: str) -> Path:
    path = ARTIFACTS_DIR / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path
