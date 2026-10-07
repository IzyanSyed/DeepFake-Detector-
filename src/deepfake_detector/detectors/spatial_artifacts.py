import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelBranch(nn.Module):
    """
    Spatial Domain Branch: extracts spatial features, blending boundary artifacts,
    and visual inconsistencies from RGB face frames.
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_block(x)
        feat = torch.flatten(feat, 1)
        return F.relu(self.fc(feat))


class FrequencyBranch(nn.Module):
    """
    Frequency Domain Branch: computes 2D Fast Fourier Transform (FFT) / magnitude spectra
    to capture high-frequency GAN/diffusion synthesis artifacts and checkerboard noise.
    """
    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, feature_dim)

    def extract_fft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the log magnitude spectrum of input RGB frames via 2D Real FFT.
        """
        # x shape: (B, C, H, W)
        fft = torch.fft.rfft2(x, dim=(-2, -1))
        mag = torch.abs(fft)
        log_mag = torch.log(mag + 1e-8)
        # Shift zero-frequency component to the center for isotropic spatial consistency
        log_mag_shifted = torch.fft.fftshift(log_mag, dim=(-2, -1))
        # Resize frequency spectrum back to spatial size (H, W) if needed
        if log_mag_shifted.shape[-2:] != x.shape[-2:]:
            log_mag_shifted = F.interpolate(log_mag_shifted, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return log_mag_shifted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq_spec = self.extract_fft_magnitude(x)
        feat = self.conv_block(freq_spec)
        feat = torch.flatten(feat, 1)
        return F.relu(self.fc(feat))


class SpatialArtifactsDetector(nn.Module):
    """
    Two-branch deepfake detector combining Spatial (Pixel) and Frequency (FFT) streams.
    """
    def __init__(self, feature_dim: int = 256, dropout_rate: float = 0.5):
        super().__init__()
        self.pixel_branch = PixelBranch(in_channels=3, feature_dim=feature_dim)
        self.freq_branch = FrequencyBranch(in_channels=3, feature_dim=feature_dim)

        self.fusion_head = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W) in range [0, 1] or normalized.
        Returns:
            torch.Tensor: Unnormalized raw logits of shape (B, 1).
        """
        pixel_feat = self.pixel_branch(x)
        freq_feat = self.freq_branch(x)
        fused = torch.cat([pixel_feat, freq_feat], dim=1)
        logits = self.fusion_head(fused)
        return logits

    def predict(self, x: torch.Tensor) -> dict:
        """
        Convenience method to return probability scores and feature representations.
        """
        self.eval()
        with torch.no_grad():
            pixel_feat = self.pixel_branch(x)
            freq_feat = self.freq_branch(x)
            fused = torch.cat([pixel_feat, freq_feat], dim=1)
            logits = self.fusion_head(fused)
            probs = torch.sigmoid(logits).squeeze(-1)

        return {
            "probabilities": probs,
            "logits": logits.squeeze(-1),
            "pixel_features": pixel_feat,
            "frequency_features": freq_feat
        }
