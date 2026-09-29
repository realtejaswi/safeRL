import torch
from torch.utils.data import IterableDataset
from .preprocessor import preprocess_example
import itertools

class DroidIterableDataset(IterableDataset):
    def __init__(self, hf_dataset, config, max_samples=None, is_train=True):
        super().__init__()
        self.dataset = hf_dataset
        self.config = config
        self.max_samples = max_samples
        self.is_train = is_train
        
    def __iter__(self):
        count = 0
        for example in self.dataset:
            if self.max_samples is not None and count >= self.max_samples:
                break
                
            # Deterministic split by episode_index
            ep_idx = example.get("episode_index", 0)
            is_val_ep = (ep_idx % 10) == 0
            
            if self.is_train and is_val_ep:
                continue
            if not self.is_train and not is_val_ep:
                continue
                
            processed = preprocess_example(example, self.config)
            
            yield {
                "state": torch.tensor(processed["state"], dtype=torch.float32),
                "action": torch.tensor(processed["action"], dtype=torch.float32),
                "episode_id": torch.tensor(processed["episode_id"], dtype=torch.long)
            }
            
            count += 1
