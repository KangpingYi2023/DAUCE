import torch


def get_torch_device(extra=False):
    if not torch.cuda.is_available():
        device = "cpu"
    elif extra:
        device = "cuda:0"
    else:
        # Historically the heavy training/eval workloads ran on the second GPU
        # to leave cuda:0 free; fall back to cuda:0 on single-GPU boxes.
        n_gpus = torch.cuda.device_count()
        device = f"cuda:{n_gpus - 1}" if n_gpus > 1 else "cuda:0"
    print("Torch device:", device)
    return device
