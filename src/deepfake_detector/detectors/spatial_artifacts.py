"""Spatial artifacts detector.

Two-branch detector operating on a window of frames:

* Pixel branch: EfficientNet-B0 backbone (timm, ImageNet-pretrained) followed by
  adaptive avg pool -> dropout -> linear head, applied to the detected face crop
  of every frame in the window (batched through the encoder together).
* Frequency branch: grayscale face crop -> 2D FFT -> radial energy distribution
  plus high-frequency energy ratio -> small MLP.

Branch scores are fused with configurable weights (see config.py).

Face detection uses MediaPipe first and falls back to an OpenCV Haar cascade.
If neither finds a face the detector reports confidence=0.0 rather than guessing.

The face-crop helpers and the per-window aggregation strategy are ports of the
reference solution (kernel_utils.py / training/zoo/classifiers.py), rewritten
here so this module has no dependency on the reference code.

Frames in a SignalWindow are expected in BGR order (OpenCV convention).
"""

from __future__ import annotations

import os
import time
from collections import Counter

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn

from deepfake_detector.config import Settings
from deepfake_detector.detectors.base import Detector, DetectorResult, SignalFamily
from deepfake_detector.ingestion.stream_reader import SignalWindow

IMAGE_SIZE = 224
FACE_CROP_MARGIN = 0.2
FREQUENCY_BINS = 32
HIGH_FREQUENCY_RADIUS_RATIO = 0.5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    """Resize img so its longer side equals `size`, preserving aspect ratio.

    Ported from the reference kernel_utils.py.
    """
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    return cv2.resize(img, (int(w), int(h)), interpolation=interpolation)


def put_to_center(img, input_size):
    """Crop to input_size x input_size and zero-pad the result to center it.

    Ported from the reference kernel_utils.py.
    """
    img = img[:input_size, :input_size]
    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    start_w = (input_size - img.shape[1]) // 2
    start_h = (input_size - img.shape[0]) // 2
    image[start_h:start_h + img.shape[0], start_w:start_w + img.shape[1], :] = img
    return image


def confident_strategy(pred, t=0.8):
    """Aggregate per-frame probabilities into one window-level score.

    Emphasizes confident predictions when most frames agree on a verdict,
    otherwise falls back to the mean. Ported from the reference kernel_utils.py.
    """
    pred = np.array(pred)
    sz = len(pred)
    fakes = np.count_nonzero(pred > t)
    if fakes > sz // 2.5 and fakes > 11:
        return np.mean(pred[pred > t])
    elif np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    else:
        return np.mean(pred)


def frequency_features(gray_crop: np.ndarray) -> np.ndarray:
    """Radial FFT energy distribution (plus high-frequency ratio) of a grayscale crop.

    Returns a vector of FREQUENCY_BINS radial energy bins (normalized to sum to
    1.0) followed by the high-frequency energy ratio.
    """
    gray = gray_crop.astype(np.float32)
    gray = gray - gray.mean()
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    height, width = spectrum.shape
    center_y, center_x = height // 2, width // 2
    yy, xx = np.mgrid[:height, :width]
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    max_radius = float(np.hypot(center_y, center_x))
    if max_radius <= 0.0:
        return np.zeros(FREQUENCY_BINS + 1, dtype=np.float32)

    bin_edges = np.linspace(0.0, max_radius, FREQUENCY_BINS + 1)
    radial_energy, _ = np.histogram(radius, bins=bin_edges, weights=spectrum)
    total_energy = float(radial_energy.sum())
    distribution = radial_energy / total_energy if total_energy > 0.0 else radial_energy

    high_frequency_mask = radius > HIGH_FREQUENCY_RADIUS_RATIO * max_radius
    high_frequency_energy = float(spectrum[high_frequency_mask].sum())
    high_frequency_ratio = high_frequency_energy / total_energy if total_energy > 0.0 else 0.0

    return np.concatenate([distribution, [high_frequency_ratio]]).astype(np.float32)


class PixelBranch(nn.Module):
    """Encoder -> pool -> dropout -> linear head, following the reference DeepFakeClassifier pattern."""

    def __init__(self, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.encoder = timm.create_model("efficientnet_b0", pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.encoder.num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class FrequencyBranch(nn.Module):
    """Small MLP over the FFT radial energy distribution and high-frequency energy ratio."""

    def __init__(
        self,
        in_features: int = FREQUENCY_BINS + 1,
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SpatialArtifactsDetector(Detector):
    """Two-branch spatial artifacts detector (see module docstring)."""

    def __init__(
        self,
        settings: Settings | None = None,
        device: str | None = None,
        pixel_model: nn.Module | None = None,
        frequency_model: nn.Module | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pixel_model = (pixel_model if pixel_model is not None else PixelBranch()).to(self.device).eval()
        self.frequency_model = (frequency_model if frequency_model is not None else FrequencyBranch()).to(self.device).eval()
        self._mediapipe_detector = None
        self._haar_cascade = None
        if self.settings.spatial_artifacts_checkpoint_path:
            self.load_checkpoint()

    async def analyze(self, window: SignalWindow) -> DetectorResult:
        start_time = time.perf_counter()

        frames = self._sample_frames(window.frames)
        face_crops: list[np.ndarray] = []
        detection_methods: list[str] = []

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected = self._find_face(rgb)
            if detected is None:
                detection_methods.append("none")
                continue
            box, method = detected
            face_crops.append(self._crop_face(rgb, box))
            detection_methods.append(method)

        num_frames = len(frames)
        num_faces = len(face_crops)

        if num_faces == 0:
            return DetectorResult(
                score=0.0,
                confidence=0.0,
                latency_ms=self._elapsed_ms(start_time),
                signal_family=SignalFamily.SPATIAL_ARTIFACTS,
                raw_signals={
                    "face_detection_method": "none",
                    "pixel_branch_score": 0.0,
                    "frequency_branch_score": 0.0,
                    "high_frequency_energy_ratio": 0.0,
                    "frames_analyzed": num_frames,
                    "faces_detected": 0,
                },
            )

        pixel_scores = self._pixel_branch_scores(face_crops)
        frequency_scores, high_frequency_ratio = self._frequency_branch_scores(face_crops)

        pixel_branch_score = float(confident_strategy(pixel_scores))
        frequency_branch_score = float(confident_strategy(frequency_scores))
        score = float(
            np.clip(
                self.settings.fusion_weight_pixel * pixel_branch_score
                + self.settings.fusion_weight_frequency * frequency_branch_score,
                0.0,
                1.0,
            )
        )

        return DetectorResult(
            score=score,
            confidence=float(num_faces / num_frames),
            latency_ms=self._elapsed_ms(start_time),
            signal_family=SignalFamily.SPATIAL_ARTIFACTS,
            raw_signals={
                "face_detection_method": self._dominant_method(detection_methods),
                "pixel_branch_score": pixel_branch_score,
                "frequency_branch_score": frequency_branch_score,
                "high_frequency_energy_ratio": high_frequency_ratio,
                "frames_analyzed": num_frames,
                "faces_detected": num_faces,
            },
        )

    def load_checkpoint(self, path: str | None = None) -> None:
        """Load trained branch weights from a checkpoint.

        Expected format: a dict with "pixel_model" and "frequency_model" state
        dicts, as will be produced by scripts/train_spatial_artifacts.py.
        """
        checkpoint_path = path or self.settings.spatial_artifacts_checkpoint_path
        if not checkpoint_path:
            return
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Spatial artifacts checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.pixel_model.load_state_dict(checkpoint["pixel_model"])
        self.frequency_model.load_state_dict(checkpoint["frequency_model"])
        self.pixel_model.eval()
        self.frequency_model.eval()

    def _sample_frames(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Take every `frame_sample_rate`-th frame from the window."""
        if not frames:
            return []
        stride = max(1, int(self.settings.frame_sample_rate))
        return frames[::stride]

    def _find_face(self, rgb: np.ndarray) -> tuple[tuple[int, int, int, int], str] | None:
        """Detect a face in an RGB frame; MediaPipe first, Haar cascade as fallback."""
        box = self._detect_face_mediapipe(rgb)
        if box is not None:
            return box, "mediapipe"
        box = self._detect_face_haar(rgb)
        if box is not None:
            return box, "haar"
        return None

    def _detect_face_mediapipe(self, rgb: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return an (x, y, w, h) box via MediaPipe, or None if unavailable/no face."""
        try:
            import mediapipe as mp
        except ImportError:
            return None
        if self._mediapipe_detector is None:
            self._mediapipe_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self.settings.face_detection_min_confidence,
            )
        result = self._mediapipe_detector.process(rgb)
        if not result.detections:
            return None
        relative = result.detections[0].location_data.relative_bounding_box
        height, width = rgb.shape[:2]
        x, y = int(relative.xmin * width), int(relative.ymin * height)
        w, h = int(relative.width * width), int(relative.height * height)
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)

    def _detect_face_haar(self, rgb: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return an (x, y, w, h) box via an OpenCV Haar cascade, or None if no face."""
        if self._haar_cascade is None:
            self._haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self._haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = (int(value) for value in faces[0])
        return (x, y, w, h)

    def _crop_face(self, rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        """Crop the face region with a small margin, clipped to the frame bounds."""
        x, y, w, h = box
        margin_x, margin_y = int(FACE_CROP_MARGIN * w), int(FACE_CROP_MARGIN * h)
        x0, y0 = max(0, x - margin_x), max(0, y - margin_y)
        x1, y1 = min(rgb.shape[1], x + w + margin_x), min(rgb.shape[0], y + h + margin_y)
        return rgb[y0:y1, x0:x1]

    def _preprocess_face(self, crop: np.ndarray) -> np.ndarray:
        """Resize/center a face crop to IMAGE_SIZE and normalize for the backbone."""
        resized = isotropically_resize_image(crop, IMAGE_SIZE)
        centered = put_to_center(resized, IMAGE_SIZE)
        normalized = (centered.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return np.transpose(normalized, (2, 0, 1))

    def _pixel_branch_scores(self, face_crops: list[np.ndarray]) -> np.ndarray:
        """Run all face crops through the pixel branch as a single batch."""
        batch = np.stack([self._preprocess_face(crop) for crop in face_crops])
        inputs = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.pixel_model(inputs).squeeze(-1))
        return probabilities.cpu().numpy()

    def _frequency_branch_scores(self, face_crops: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """Run FFT features for all face crops through the frequency branch as a single batch."""
        features = np.stack(
            [frequency_features(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)) for crop in face_crops]
        )
        high_frequency_ratio = float(features[:, -1].mean())
        inputs = torch.from_numpy(features).to(self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.frequency_model(inputs).squeeze(-1))
        return probabilities.cpu().numpy(), high_frequency_ratio

    def _dominant_method(self, detection_methods: list[str]) -> str:
        """Report the face detection method used for the majority of detected faces."""
        detected = [method for method in detection_methods if method != "none"]
        if not detected:
            return "none"
        counts = Counter(detected)
        return max(counts, key=lambda method: (counts[method], method == "mediapipe"))

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000.0