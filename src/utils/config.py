import yaml
from pathlib import Path

def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_all_configs(configs_dir: str | Path = "configs") -> dict:
    configs_dir = Path(configs_dir)
    cfg = {}
    for yaml_file in configs_dir.glob("*.yaml"):
        cfg.update(load_config(yaml_file))
    return cfg
