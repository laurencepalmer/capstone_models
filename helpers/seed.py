import torch
import numpy as np
import random

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"Seed {seed} has been set")

def seed_dataloader(work_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_generator(seed: int):
    g_seed = torch.Generator()
    g_seed.manual_seed(seed)
    return g_seed

# if you want to use the gpu, then select it in the hardware accelerator
def set_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")