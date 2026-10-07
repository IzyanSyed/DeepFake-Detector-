import unittest
import torch
import sys
import os

# Ensure src is in Python path for test import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from deepfake_detector.detectors.spatial_artifacts import (
    SpatialArtifactsDetector,
    PixelBranch,
    FrequencyBranch
)


class TestSpatialArtifactsDetector(unittest.TestCase):

    def setUp(self):
        self.feature_dim = 128
        self.model = SpatialArtifactsDetector(feature_dim=self.feature_dim, dropout_rate=0.3)
        self.model.eval()

    def test_pixel_branch_shape(self):
        branch = PixelBranch(in_channels=3, feature_dim=self.feature_dim)
        x = torch.randn(2, 3, 128, 128)
        out = branch(x)
        self.assertEqual(out.shape, (2, self.feature_dim))

    def test_frequency_branch_shape(self):
        branch = FrequencyBranch(in_channels=3, feature_dim=self.feature_dim)
        x = torch.randn(2, 3, 128, 128)
        out = branch(x)
        self.assertEqual(out.shape, (2, self.feature_dim))

    def test_detector_forward_shape(self):
        x = torch.randn(4, 3, 224, 224)
        logits = self.model(x)
        self.assertEqual(logits.shape, (4, 1))

    def test_detector_predict_output(self):
        x = torch.rand(2, 3, 128, 128)
        res = self.model.predict(x)
        
        self.assertIn("probabilities", res)
        self.assertIn("logits", res)
        self.assertIn("pixel_features", res)
        self.assertIn("frequency_features", res)

        probs = res["probabilities"]
        self.assertEqual(probs.shape, (2,))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

        pixel_feats = res["pixel_features"]
        self.assertEqual(pixel_feats.shape, (2, self.feature_dim))

        freq_feats = res["frequency_features"]
        self.assertEqual(freq_feats.shape, (2, self.feature_dim))

    def test_single_batch_inference(self):
        x = torch.randn(1, 3, 380, 380)
        logits = self.model(x)
        self.assertEqual(logits.shape, (1, 1))


if __name__ == '__main__':
    unittest.main()
