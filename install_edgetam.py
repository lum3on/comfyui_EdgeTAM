#!/usr/bin/env python3
"""
EdgeTAM Installation Script for ComfyUI
Automatically installs EdgeTAM dependencies and downloads model weights.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("Error: Python 3.10 or higher is required for EdgeTAM")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"Python version OK: {version.major}.{version.minor}.{version.micro}")

def check_torch():
    """Check if PyTorch is installed with correct version."""
    try:
        import torch
        version = torch.__version__
        print(f"PyTorch version: {version}")
        
        # Check version
        major, minor = map(int, version.split('.')[:2])
        if major < 2 or (major == 2 and minor < 3):
            print("Warning: PyTorch 2.3.1 or higher is recommended for EdgeTAM")
            return False
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("MPS (Apple Silicon) available")
        else:
            print("Running on CPU only")
        
        return True
        
    except ImportError:
        print("PyTorch not found. Please install PyTorch first:")
        print("https://pytorch.org/get-started/locally/")
        return False

def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    
    # Install from requirements.txt
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        run_command(f"pip install -r {requirements_file}")
    else:
        # Install manually if requirements.txt is missing
        packages = [
            "torch>=2.3.1",
            "torchvision>=0.18.1", 
            "numpy>=1.21.0",
            "opencv-python>=4.5.0",
            "Pillow>=8.3.0",
            "hydra-core>=1.1.0",
            "omegaconf>=2.1.0",
            "huggingface-hub>=0.16.0",
            "moviepy>=1.0.3",
            "tqdm>=4.62.0"
        ]
        
        for package in packages:
            run_command(f"pip install {package}")

def clone_edgetam():
    """Clone EdgeTAM repository into the custom node directory."""
    print("Cloning EdgeTAM repository...")
    
    parent_dir = Path(__file__).parent
    edgetam_path = parent_dir / "EdgeTAM"
    
    if edgetam_path.exists():
        print(f"EdgeTAM directory already exists at {edgetam_path}. Skipping clone.")
    else:
        # Clone repository
        run_command(f"git clone https://github.com/facebookresearch/EdgeTAM.git {edgetam_path}")
    
    # Install EdgeTAM
    print("Installing EdgeTAM...")
    original_dir = os.getcwd()
    try:
        # Change to the EdgeTAM directory to run installation
        os.chdir(edgetam_path)
        run_command("pip install -e .")
    finally:
        # Change back to the original directory
        os.chdir(original_dir)

def download_model():
    """Download EdgeTAM model checkpoint."""
    print("Downloading EdgeTAM model checkpoint...")
    
    # Create models directory
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "edgetam.pt"
    
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return
    
    # Download model
    import urllib.request
    model_url = "https://github.com/facebookresearch/EdgeTAM/releases/download/v1.0/edgetam.pt"
    
    print(f"Downloading from {model_url}...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Model downloaded to {model_path}")
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("You can manually download the model from:")
        print("https://github.com/facebookresearch/EdgeTAM/tree/main/checkpoints")

def test_installation():
    """Test if EdgeTAM is properly installed."""
    print("Testing EdgeTAM installation...")
    
    try:
        import sam2
        from sam2.build_sam import build_sam2_video_predictor
        print("EdgeTAM import successful!")
        
        # Try to load model (without actually running inference)
        config_path = Path(__file__).parent / "configs" / "edgetam.yaml"
        model_path = Path(__file__).parent / "models" / "edgetam.pt"
        
        if config_path.exists() and model_path.exists():
            print("Configuration and model files found!")
            print("EdgeTAM installation appears to be successful!")
        else:
            print("Warning: Some files may be missing")
            
    except ImportError as e:
        print(f"EdgeTAM import failed: {e}")
        print("Installation may have failed. Please check the error messages above.")
        return False
    
    return True

def main():
    """Main installation function."""
    print("=" * 60)
    print("EdgeTAM ComfyUI Installation Script")
    print("=" * 60)
    
    # Check prerequisites
    check_python_version()
    
    if not check_torch():
        print("Please install PyTorch first, then run this script again.")
        sys.exit(1)
    
    # Install components
    try:
        install_requirements()
        clone_edgetam()
        download_model()
        
        # Test installation
        if test_installation():
            print("\n" + "=" * 60)
            print("Installation completed successfully!")
            print("You can now use EdgeTAM nodes in ComfyUI.")
            print("Please restart ComfyUI to load the new nodes.")
            print("=" * 60)
        else:
            print("\nInstallation completed with warnings.")
            print("Please check the messages above.")
            
    except KeyboardInterrupt:
        print("\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nInstallation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
