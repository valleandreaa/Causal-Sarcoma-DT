import torch

def is_cuda_available():
    """
    Check if CUDA is available.
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()

def get_device():
    """
    Get the appropriate device (CUDA if available, otherwise CPU).
    Returns:
        torch.device: The device to use.
    """
    return torch.device("cuda" if is_cuda_available() else "cpu")

def move_to_device(tensor):
    """
    Move a tensor to the appropriate device.
    Args:
        tensor (torch.Tensor): The tensor to move.
    Returns:
        torch.Tensor: The tensor moved to the appropriate device.
    """
    return tensor.to(get_device())

def move_model_to_device(model):
    """
    Move a model to the appropriate device.
    Args:
        model (torch.nn.Module): The model to move.
    Returns:
        torch.nn.Module: The model moved to the appropriate device.
    """
    return model.to(get_device())

def move_batch_to_device(batch):
    """
    Move a batch of data to the appropriate device.
    Args:
        batch (dict or list or tuple): The batch of data to move.
    Returns:
        The batch of data moved to the appropriate device.
    """
    device = get_device()
    if isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
    else:
        raise TypeError("Batch must be a dict, list, or tuple")

def get_current_device():
    """
    Get the current device being used.
    Returns:
        torch.device: The current device.
    """
    return torch.cuda.current_device() if is_cuda_available() else torch.device("cpu")

def synchronize():
    """
    Synchronize CUDA operations.
    """
    if is_cuda_available():
        torch.cuda.synchronize()