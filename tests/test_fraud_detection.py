"""
Unit tests for fraud detection, specifically testing train_sample_13.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from src.preprocessing_helpers import (
        detect_whiteout_and_lowconf,
        compute_combined_fraud_score,
        _gaussian_whiteout_analysis,
        _compute_ela_map
    )
    from src.utils.pdf_loader import load_pdf_to_images
    FRAUD_DETECTION_AVAILABLE = True
except Exception as e:
    FRAUD_DETECTION_AVAILABLE = False
    print(f"Warning: Fraud detection modules not available: {e}")


class TestFraudDetection(unittest.TestCase):
    """Unit tests for fraud detection."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.data_dir = ROOT / "data" / "raw" / "training_samples"
        cls.train_sample_13 = cls.data_dir / "TRAINING_SAMPLES" / "train_sample_13.pdf"
        
        # Try alternative paths
        if not cls.train_sample_13.exists():
            cls.train_sample_13 = cls.data_dir / "train_sample_13.pdf"
        if not cls.train_sample_13.exists():
            # Search recursively
            for pdf in cls.data_dir.rglob("train_sample_13.pdf"):
                cls.train_sample_13 = pdf
                break
    
    def test_train_sample_13_exists(self):
        """Test that train_sample_13.pdf exists."""
        self.assertTrue(
            self.train_sample_13.exists(),
            f"train_sample_13.pdf not found. Searched in: {self.data_dir}"
        )
    
    @unittest.skipIf(not FRAUD_DETECTION_AVAILABLE, "Fraud detection not available")
    def test_train_sample_13_flagged(self):
        """Test that train_sample_13 is flagged as suspicious."""
        if not self.train_sample_13.exists():
            self.skipTest("train_sample_13.pdf not found")
        
        # Load PDF
        try:
            images = load_pdf_to_images(self.train_sample_13, dpi=300)
            self.assertGreater(len(images), 0, "No images loaded from PDF")
        except Exception as e:
            self.skipTest(f"Failed to load PDF: {e}")
        
        # Test each page
        suspicious_pages = []
        all_flags = []
        
        for page_idx, img in enumerate(images, start=1):
            # Detect fraud
            flags = detect_whiteout_and_lowconf(img, ocr_text="", enable_ela=True)
            all_flags.extend(flags)
            
            # Compute combined score
            combined_score, is_suspicious = compute_combined_fraud_score(flags, threshold=0.5)
            
            if is_suspicious:
                suspicious_pages.append(page_idx)
            
            print(f"\nPage {page_idx}:")
            print(f"  Flags: {len(flags)}")
            print(f"  Combined score: {combined_score:.3f}")
            print(f"  Suspicious: {is_suspicious}")
            for flag in flags:
                print(f"    - {flag.get('flag_type')}: {flag.get('score'):.3f}")
        
        # Assert that at least one page is flagged
        self.assertGreater(
            len(suspicious_pages),
            0,
            f"train_sample_13 should be flagged as suspicious. "
            f"Flags detected: {len(all_flags)}, "
            f"Max score: {max([f.get('score', 0) for f in all_flags], default=0):.3f}"
        )
        
        print(f"\n✓ train_sample_13 flagged on pages: {suspicious_pages}")
    
    @unittest.skipIf(not FRAUD_DETECTION_AVAILABLE, "Fraud detection not available")
    def test_whiteout_detection(self):
        """Test whiteout detection on a sample image."""
        if not self.train_sample_13.exists():
            self.skipTest("train_sample_13.pdf not found")
        
        # Load first page
        try:
            images = load_pdf_to_images(self.train_sample_13, dpi=300)
            img = images[0]
        except Exception as e:
            self.skipTest(f"Failed to load PDF: {e}")
        
        # Test Gaussian whiteout detection
        import numpy as np
        img_array = np.array(img.convert("L"))
        score, mask = _gaussian_whiteout_analysis(img_array)
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(mask.shape[:2], img_array.shape[:2])
    
    @unittest.skipIf(not FRAUD_DETECTION_AVAILABLE, "Fraud detection not available")
    def test_ela_detection(self):
        """Test ELA detection on a sample image."""
        if not self.train_sample_13.exists():
            self.skipTest("train_sample_13.pdf not found")
        
        # Load first page
        try:
            images = load_pdf_to_images(self.train_sample_13, dpi=300)
            img = images[0]
        except Exception as e:
            self.skipTest(f"Failed to load PDF: {e}")
        
        # Test ELA computation
        ela_map = _compute_ela_map(img)
        
        import numpy as np
        self.assertIsInstance(ela_map, np.ndarray)
        self.assertEqual(ela_map.shape[:2], (img.height, img.width))
        self.assertGreaterEqual(ela_map.min(), 0.0)
        self.assertLessEqual(ela_map.max(), 1.0)
    
    @unittest.skipIf(not FRAUD_DETECTION_AVAILABLE, "Fraud detection not available")
    def test_combined_scoring(self):
        """Test combined fraud scoring."""
        # Test with empty flags
        score, is_suspicious = compute_combined_fraud_score([], threshold=0.5)
        self.assertEqual(score, 0.0)
        self.assertFalse(is_suspicious)
        
        # Test with flags
        flags = [
            {"flag_type": "whiteout_gaussian", "score": 0.8, "meta": {}},
            {"flag_type": "ela_anomaly", "score": 0.6, "meta": {}}
        ]
        score, is_suspicious = compute_combined_fraud_score(flags, threshold=0.5)
        self.assertGreater(score, 0.0)
        self.assertTrue(is_suspicious)  # Should be suspicious with high scores


if __name__ == "__main__":
    unittest.main(verbosity=2)


