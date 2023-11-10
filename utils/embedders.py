from abc import ABC, abstractmethod
import numpy as np

class EmbeddorBase:

    def __init__(self) -> None:
        pass

    @abstractmethod
    def transform(self, reviews : [str]) -> np.ndarray:
        pass