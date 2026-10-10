"""Provenance detector — C2PA Content Credentials verification.

This detector checks whether a video file carries valid C2PA (Coalition for Content
Provenance and Authenticity) Content Credentials. It uses the official c2pa-python
library when available (full cryptographic verification), with a fallback to a basic
presence check if the library fails.

Verification levels:
- No manifest found → "unknown provenance" (score=0.5, confidence=0.0) — not evidence of fakery
- Manifest found but signature invalid/tampered → flagged distinctly (score=0.7, confidence=0.5)
- Manifest found and valid → strong "real" signal (score=0.1, confidence=0.9+)

Note: This detector only ever supports REAL or UNKNOWN — it never asserts FAKE.
A missing C2PA manifest means "no signal," not "fake."
"""

from __future__ import annotations

import os
from pathlib import Path

from deepfake_detector.config import Settings
from deepfake_detector.detectors.base import Detector, DetectorResult, SignalFamily
from deepfake_detector.ingestion.stream_reader import SignalWindow

# Try to import c2pa-python for full verification
try:
    import c2pa
    import c2pa.c2pa as c2pa_errors
    C2PA_AVAILABLE = True
    C2PA_VERSION = getattr(c2pa, '__version__', 'unknown')
except ImportError:
    c2pa = None
    c2pa_errors = None
    C2PA_AVAILABLE = False
    C2PA_VERSION = None


class ProvenanceDetector(Detector):
    """C2PA Content Credentials provenance detector."""

    def __init__(
        self,
        settings: Settings | None = None,
        device: str | None = None,
        video_path: str | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.device = device or "cpu"
        self._video_path = video_path
        self._verification_method = "c2pa-python" if C2PA_AVAILABLE else "basic-scan"

    def set_video_path(self, video_path: str) -> None:
        """Set the video file path for analysis.

        This is needed because SignalWindow doesn't carry file paths.
        """
        self._video_path = video_path

    async def analyze(self, window: SignalWindow) -> DetectorResult:
        """Analyze the video file for C2PA Content Credentials.

        Since this detector operates on the source video file (not frame pixels),
        the video path must be set via set_video_path() before calling analyze(),
        or passed as an attribute on the window object.
        """
        import time
        start_time = time.perf_counter()

        # Try to get video path from window if not set
        video_path = self._video_path
        if video_path is None and hasattr(window, 'video_path'):
            video_path = window.video_path

        if not video_path or not os.path.isfile(video_path):
            return DetectorResult(
                score=0.5,
                confidence=0.0,
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                signal_family=SignalFamily.PROVENANCE,
                raw_signals={
                    "manifest_found": False,
                    "verification_method": self._verification_method,
                    "signature_valid": None,
                    "error": "No video file path provided",
                    "claims": None,
                },
            )

        # Try full c2pa-python verification first
        if C2PA_AVAILABLE:
            return await self._analyze_with_c2pa(video_path, start_time)

        # Fallback to basic scan
        return await self._analyze_basic_scan(video_path, start_time)

    async def _analyze_with_c2pa(self, video_path: str, start_time: float) -> DetectorResult:
        """Full cryptographic verification using c2pa-python."""
        try:
            reader = c2pa.Reader(video_path)

            if not reader.has_manifest():
                return DetectorResult(
                    score=0.5,  # neutral - no signal either way
                    confidence=0.0,
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    signal_family=SignalFamily.PROVENANCE,
                    raw_signals={
                        "manifest_found": False,
                        "verification_method": "c2pa-python",
                        "signature_valid": None,
                        "claims": None,
                        "c2pa_version": C2PA_VERSION,
                    },
                )

            # Manifest exists - check its validity
            manifest = reader.active_manifest()
            trust_status = manifest.trust_status if hasattr(manifest, 'trust_status') else None
            claims = manifest.claims if hasattr(manifest, 'claims') else None
            ingredients = manifest.ingredients if hasattr(manifest, 'ingredients') else None

            # Extract claim data
            claim_data = {}
            if claims:
                try:
                    claim_data = json.loads(claims) if isinstance(claims, str) else claims
                except Exception:
                    claim_data = {"raw": str(claims)}

            # Determine if signature is valid
            is_valid = trust_status == "trusted" if trust_status else False

            if is_valid:
                # Valid manifest - strong real signal
                score = 0.1
                confidence = 0.95
            else:
                # Manifest found but invalid/tampered - suspicious
                score = 0.7
                confidence = 0.5

            return DetectorResult(
                score=score,
                confidence=confidence,
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                signal_family=SignalFamily.PROVENANCE,
                raw_signals={
                    "manifest_found": True,
                    "verification_method": "c2pa-python",
                    "signature_valid": is_valid,
                    "trust_status": trust_status,
                    "claims": claim_data,
                    "ingredients": str(ingredients) if ingredients else None,
                    "c2pa_version": C2PA_VERSION,
                },
            )

        except c2pa_errors._C2paManifestNotFound:
            # No manifest found - this is the expected case for most videos
            return DetectorResult(
                score=0.5,
                confidence=0.0,
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                signal_family=SignalFamily.PROVENANCE,
                raw_signals={
                    "manifest_found": False,
                    "verification_method": "c2pa-python",
                    "signature_valid": None,
                    "claims": None,
                    "c2pa_version": C2PA_VERSION,
                },
            )
        except Exception as e:
            # Other errors (corrupt file, etc.) - fall back to basic scan
            import traceback
            return await self._analyze_basic_scan(video_path, start_time, fallback_error=str(e))

    async def _analyze_basic_scan(self, video_path: str, start_time: float, fallback_error: str = None) -> DetectorResult:
        """Basic fallback: scan for JUMBF box marker in file metadata.

        This is NOT a full verification - it only checks for presence of
        C2PA-like markers. No cryptographic validation is performed.
        Use only when c2pa-python is unavailable.
        """
        try:
            # Try using ffprobe first (most reliable for container metadata)
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format_tags", "-of", "json", video_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                import json
                tags = json.loads(result.stdout).get("format", {}).get("tags", {})
                # Look for C2PA-related tags
                c2pa_tags = {k: v for k, v in tags.items() if "c2pa" in k.lower() or "jumbf" in k.lower()}
                if c2pa_tags:
                    return DetectorResult(
                        score=0.3,  # suggestive but not verified
                        confidence=0.3,
                        latency_ms=(time.perf_counter() - start_time) * 1000.0,
                        signal_family=SignalFamily.PROVENANCE,
                        raw_signals={
                            "manifest_found": True,
                            "verification_method": "basic-scan-ffprobe",
                            "signature_valid": None,
                            "claims": c2pa_tags,
                            "fallback_error": fallback_error,
                        },
                    )
        except Exception:
            pass

        # Last resort: binary scan for JUMBF marker
        try:
            with open(video_path, "rb") as f:
                # Read first and last 1MB for JUMBF signature
                head = f.read(1024 * 1024)
                f.seek(-1024 * 1024, os.SEEK_END)
                tail = f.read(1024 * 1024)

            # JUMBF boxes typically contain "uuid" or "c2pa" markers
            jumbf_markers = [b'c2pa', b'C2PA', b'jumbf', b'JUMBF', b'cai']
            found_in_head = any(marker in head for marker in jumbf_markers)
            found_in_tail = any(marker in tail for marker in jumbf_markers)

            if found_in_head or found_in_tail:
                return DetectorResult(
                    score=0.3,
                    confidence=0.2,
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    signal_family=SignalFamily.PROVENANCE,
                    raw_signals={
                        "manifest_found": True,
                        "verification_method": "basic-scan-binary",
                        "signature_valid": None,
                        "claims": {"marker_found": True, "locations": ["head"] if found_in_head else ["tail"]},
                        "fallback_error": fallback_error,
                    },
                )
        except Exception:
            pass

        # Nothing found
        return DetectorResult(
            score=0.5,
            confidence=0.0,
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
            signal_family=SignalFamily.PROVENANCE,
            raw_signals={
                "manifest_found": False,
                "verification_method": "basic-scan",
                "signature_valid": None,
                "claims": None,
                "fallback_error": fallback_error,
            },
        )


# Need time import for basic scan
import time
import json