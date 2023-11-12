from abc import abstractmethod
import numpy as np


class EmbeddorBase:
    def __init__(self) -> None:
        pass

    @abstractmethod
def _get_torch_device() -> torch.device:
    """
    Returns the device to be used for PyTorch operations.

    Returns:
        torch.device: Device to be used for PyTorch operations.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")