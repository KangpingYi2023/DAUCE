import os

import torch
from torch.nn import Module

from update_utils import path_util


def save_torch_model(model: Module, relative_path: str) -> None:
    """
    save model to given path
    """
    absolute_path = path_util.get_absolute_path(relative_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    torch.save(model.state_dict(), absolute_path)

    print("Saved to:", absolute_path)
