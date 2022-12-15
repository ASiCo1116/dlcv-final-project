import torch
import random
import numpy as np


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)


def get_device(device_id):
    return (
        f"cuda:{device_id}" if device_id > -1 and torch.cuda.is_available() else "cpu"
    )
