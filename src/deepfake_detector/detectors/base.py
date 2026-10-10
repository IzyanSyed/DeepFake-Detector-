from __future__ import annotations

import abc
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from deepfake_detector.ingestion.stream_reader import SignalWindow


class SignalFamily(str, Enum):
    PHYSIOLOGICAL = "PHYSIOLOGICAL"
    SPATIAL_ARTIFACTS = "SPATIAL_ARTIFACTS"
    TEMPORAL_CONSISTENCY = "TEMPORAL_CONSISTENCY"
    AV_SYNC = "AV_SYNC"
    PROVENANCE = "PROVENANCE"


class AlertLevel(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DetectorResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Likelihood the window is fake; higher = more likely fake")
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    signal_family: SignalFamily
    raw_signals: dict[str, Any]


class Detector(abc.ABC):
    @abc.abstractmethod
    async def analyze(self, window: "SignalWindow") -> DetectorResult:
        """Analyze a SignalWindow and return a DetectorResult."""
        ...