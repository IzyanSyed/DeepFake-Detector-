from dataclasses import dataclass
import numpy as np


@dataclass
class SignalWindow:
    frames: list[np.ndarray]
    timestamp: float
    window_id: str
    audio: np.ndarray | None = None