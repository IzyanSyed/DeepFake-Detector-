"""Tests for SpatialArtifactsDetector (fully mocked: no model downloads or real inference)."""

import asyncio

import numpy as np
import torch
from torch import nn

from deepfake_detector.config import Settings
from deepfake_detector.detectors.base import SignalFamily
from deepfake_detector.detectors.spatial_artifacts import SpatialArtifactsDetector
from deepfake_detector.ingestion.stream_reader import SignalWindow


class _StubPixelBranch(nn.Module):
    """Returns a constant logit for every frame in the batch."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full((inputs.shape[0], 1), self.logit)


class _StubFrequencyBranch(nn.Module):
    """Returns a constant logit for every frame in the batch."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full((inputs.shape[0], 1), self.logit)


def _make_detector(monkeypatch, *, pixel_logit=0.0, frequency_logit=0.0, mediapipe_box=None, haar_box=None):
    detector = SpatialArtifactsDetector(
        settings=Settings(),
        pixel_model=_StubPixelBranch(pixel_logit),
        frequency_model=_StubFrequencyBranch(frequency_logit),
    )
    monkeypatch.setattr(detector, "_detect_face_mediapipe", lambda rgb: mediapipe_box)
    monkeypatch.setattr(detector, "_detect_face_haar", lambda rgb: haar_box)
    return detector


def _make_window(num_frames: int = 3) -> SignalWindow:
    frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(num_frames)]
    return SignalWindow(frames=frames, timestamp=0.0, window_id="test-window")


def test_no_face_detected_returns_zero_confidence(monkeypatch):
    detector = _make_detector(monkeypatch)
    result = asyncio.run(detector.analyze(_make_window()))
    assert result.confidence == 0.0
    assert result.raw_signals["face_detection_method"] == "none"


def test_mediapipe_face_detection_used(monkeypatch):
    detector = _make_detector(
        monkeypatch,
        mediapipe_box=(5, 5, 30, 30),
        haar_box=(0, 0, 10, 10),
    )
    result = asyncio.run(detector.analyze(_make_window()))
    assert result.raw_signals["face_detection_method"] == "mediapipe"
    assert result.confidence > 0.0


def test_haar_fallback_used_when_mediapipe_fails(monkeypatch):
    detector = _make_detector(
        monkeypatch,
        mediapipe_box=None,
        haar_box=(0, 0, 20, 20),
    )
    result = asyncio.run(detector.analyze(_make_window()))
    assert result.raw_signals["face_detection_method"] == "haar"


def test_fusion_output_in_unit_range(monkeypatch):
    for pixel_logit, frequency_logit in [(15.0, 15.0), (-15.0, -15.0), (15.0, -15.0)]:
        detector = _make_detector(
            monkeypatch,
            pixel_logit=pixel_logit,
            frequency_logit=frequency_logit,
            mediapipe_box=(5, 5, 30, 30),
        )
        result = asyncio.run(detector.analyze(_make_window()))
        assert 0.0 <= result.score <= 1.0


def test_detector_result_fields_populated(monkeypatch):
    detector = _make_detector(monkeypatch, mediapipe_box=(5, 5, 30, 30))
    result = asyncio.run(detector.analyze(_make_window()))
    assert isinstance(result.score, float)
    assert isinstance(result.confidence, float)
    assert isinstance(result.latency_ms, float)
    assert result.latency_ms >= 0.0
    assert result.signal_family is SignalFamily.SPATIAL_ARTIFACTS
    assert isinstance(result.raw_signals, dict)
    assert isinstance(result.raw_signals["pixel_branch_score"], float)
    assert isinstance(result.raw_signals["frequency_branch_score"], float)
    assert isinstance(result.raw_signals["high_frequency_energy_ratio"], float)