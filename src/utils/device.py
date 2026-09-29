import torch
import platform
import logging

logger = logging.getLogger(__name__)

def get_device(device_config="auto"):
    if device_config == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = device_config
    
    logger.info(f"Using device: {device} on {platform.system()}")
    return device
