#!/usr/bin/env python3
"""
Test script for EdgeTAM ComfyUI nodes
Verifies that EdgeTAM is properly installed and can be imported.
"""

import sys
import os
import traceback

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not found")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not found")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow")
    except ImportError:
        print("✗ Pillow not found")
        return False
    
    return True

def test_edgetam():
    """Test EdgeTAM imports."""
    print("\nTesting EdgeTAM...")
    
    try:
        import sam2
        print("✓ SAM2 package found")
    except ImportError:
        print("✗ SAM2 package not found")
        print("  Please install EdgeTAM:")
        print("  git clone https://github.com/facebookresearch/EdgeTAM.git")
        print("  cd EdgeTAM && pip install -e .")
        return False
    
    try:
        from sam2.build_sam import build_sam2_video_predictor, build_sam2
        print("✓ EdgeTAM build functions")
    except ImportError as e:
        print(f"✗ EdgeTAM build functions: {e}")
        return False
    
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✓ EdgeTAM image predictor")
    except ImportError as e:
        print(f"✗ EdgeTAM image predictor: {e}")
        return False
    
    return True

def test_node_imports():
    """Test ComfyUI node imports."""
    print("\nTesting node imports...")
    
    try:
        from edgetam_utils import get_model_path, get_config_path
        print("✓ EdgeTAM utilities")
    except ImportError as e:
        print(f"✗ EdgeTAM utilities: {e}")
        return False
    
    try:
        from edgetam_nodes import EdgeTAMVideoTracker, EdgeTAMImageSegmentor
        print("✓ EdgeTAM nodes")
    except ImportError as e:
        print(f"✗ EdgeTAM nodes: {e}")
        return False
    
    return True

def test_model_download():
    """Test model download functionality."""
    print("\nTesting model download...")
    
    try:
        from edgetam_utils import get_model_path
        model_path = get_model_path()
        
        if os.path.exists(model_path):
            print(f"✓ Model found at: {model_path}")
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"  Size: {file_size:.1f} MB")
        else:
            print(f"✗ Model not found at: {model_path}")
            print("  Model will be downloaded on first use")
        
        return True
        
    except Exception as e:
        print(f"✗ Model download test failed: {e}")
        return False

def test_config():
    """Test configuration file."""
    print("\nTesting configuration...")
    
    try:
        from edgetam_utils import get_config_path
        config_path = get_config_path()
        
        if os.path.exists(config_path):
            print(f"✓ Config found at: {config_path}")
        else:
            print(f"✗ Config not found at: {config_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_device():
    """Test device detection."""
    print("\nTesting device detection...")
    
    try:
        from edgetam_utils import get_device
        device = get_device()
        print(f"✓ Detected device: {device}")
        
        import torch
        if device == "cuda" and torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name()}")
        elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  Apple Silicon MPS available")
        elif device == "cpu":
            print("  Using CPU")
        
        return True
        
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("EdgeTAM ComfyUI Node Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_edgetam,
        test_node_imports,
        test_model_download,
        test_config,
        test_device,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! EdgeTAM is ready to use.")
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
