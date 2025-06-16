"""
EdgeTAM Utilities for ComfyUI
Utility functions for EdgeTAM model loading, video processing, and tensor conversions.
"""

import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np
import torch
import cv2
from PIL import Image

# Try to import ComfyUI folder_paths, fallback if not available
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False

# Model download URLs and configurations
EDGETAM_MODEL_URL = "https://huggingface.co/spaces/facebook/EdgeTAM/resolve/main/checkpoints/edgetam.pt"
EDGETAM_CONFIG = "edgetam.yaml"

def get_model_path(model_name: str = "edgetam.pt") -> str:
    """Get the path to the EdgeTAM model, downloading if necessary."""
    # Try to find model in ComfyUI models directory if available
    if COMFYUI_AVAILABLE:
        models_dir = folder_paths.models_dir
        checkpoints_dir = os.path.join(models_dir, "checkpoints")
        model_path = os.path.join(checkpoints_dir, model_name)

        if os.path.exists(model_path):
            return model_path

    # Fallback to this node's directory
    node_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(node_dir, "models", model_name)
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Download if model doesn't exist
    if not os.path.exists(model_path):
        print(f"Downloading EdgeTAM model to {model_path}...")
        urllib.request.urlretrieve(EDGETAM_MODEL_URL, model_path)
        print("Download complete!")
    
    return model_path

def get_config_path() -> str:
    """Get the path to the EdgeTAM config file."""
    node_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(node_dir, "configs", EDGETAM_CONFIG)
    
    # Create a minimal config if it doesn't exist
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        create_default_config(config_path)
    
    return config_path

def create_default_config(config_path: str):
    """Create a default EdgeTAM configuration file."""
    config_content = """# EdgeTAM Configuration
model:
  _target_: sam2.modeling.sam2_base.SAM2Base
  image_size: 1024
  use_original_imgsize: false
  backbone:
    _target_: sam2.modeling.backbones.hieradet.Hiera
    embed_dim: 96
    num_heads: [1, 2, 4, 8]
    stages: [2, 3, 16, 3]
    global_att_blocks: [12, 16, 20]
    window_pos_embed_bkg_spatial_size: [14, 14]
  sam_mask_decoder_extra_args:
    dynamic_multimask_via_stability: true
    dynamic_multimask_stability_delta: 0.05
    dynamic_multimask_stability_thresh: 0.98
  memory_attention: CrossAttentionBlock
  memory_encoder:
    _target_: sam2.modeling.memory_encoder.MemoryEncoder
    out_dim: 64
  num_maskmem: 7
  image_encoder:
    _target_: sam2.modeling.backbones.image_encoder.ImageEncoder
    scalp: 1
    trunk:
      _target_: sam2.modeling.backbones.hieradet.Hiera
      embed_dim: 96
      num_heads: [1, 2, 4, 8]
      stages: [2, 3, 16, 3]
      global_att_blocks: [12, 16, 20]
      window_pos_embed_bkg_spatial_size: [14, 14]
    neck:
      _target_: sam2.modeling.backbones.image_encoder.FpnNeck
      position_encoding:
        _target_: sam2.modeling.position_encoding.PositionEmbeddingSine
        num_pos_feats: 256
        normalize: true
        scale: null
        temperature: 10000
      d_model: 256
      backbone_channel_list: [768, 384, 192, 96]
      fpn_top_down_levels: [2, 3]
      fpn_interp_model: nearest
  memory_bank:
    _target_: sam2.modeling.memory_bank.MemoryBank
    num_maskmem: 7
  sam_prompt_encoder:
    _target_: sam2.modeling.sam.prompt_encoder.PromptEncoder
    embed_dim: 256
    image_embedding_size: [64, 64]
    input_image_size: [1024, 1024]
    mask_in_chans: 16
  sam_mask_decoder:
    _target_: sam2.modeling.sam.mask_decoder.MaskDecoder
    num_multimask_outputs: 3
    transformer:
      _target_: sam2.modeling.sam.transformer.TwoWayTransformer
      depth: 2
      embedding_dim: 256
      mlp_dim: 2048
      num_heads: 8
    transformer_dim: 256
    iou_head_depth: 3
    iou_head_hidden_dim: 256
"""
    with open(config_path, 'w') as f:
        f.write(config_content)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor to PIL Image."""
    # ComfyUI tensors are typically (B, H, W, C) in range [0, 1]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Convert to numpy and scale to [0, 255]
    np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
    
    # Convert to PIL
    if np_image.shape[-1] == 3:  # RGB
        return Image.fromarray(np_image, 'RGB')
    elif np_image.shape[-1] == 1:  # Grayscale
        return Image.fromarray(np_image.squeeze(-1), 'L')
    else:
        raise ValueError(f"Unsupported number of channels: {np_image.shape[-1]}")

def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor format."""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    np_image = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor with ComfyUI format (B, H, W, C)
    tensor = torch.from_numpy(np_image).unsqueeze(0)
    
    return tensor

def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Convert mask array to ComfyUI tensor format."""
    # Ensure mask is in [0, 1] range
    if mask.dtype == bool:
        mask = mask.astype(np.float32)
    elif mask.max() > 1.0:
        mask = mask.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions if needed
    if mask.ndim == 2:
        mask = mask[None, :, :, None]  # (1, H, W, 1)
    elif mask.ndim == 3 and mask.shape[0] != 1:
        mask = mask[None, :, :, :]  # (1, H, W, C)
    
    return torch.from_numpy(mask)

def video_frames_to_tensor_batch(frames: List[np.ndarray]) -> torch.Tensor:
    """Convert list of video frames to ComfyUI tensor batch."""
    # Convert frames to tensors and stack
    tensor_frames = []
    for frame in frames:
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32) / 255.0
        tensor_frames.append(torch.from_numpy(frame))
    
    # Stack into batch (B, H, W, C)
    return torch.stack(tensor_frames, dim=0)

def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Load video frames from file path."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    return frames

def save_video_from_frames(frames: List[np.ndarray], output_path: str, fps: float = 30.0):
    """Save video frames to file."""
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    out.release()

def apply_mask_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.6) -> np.ndarray:
    """Apply colored mask overlay to image."""
    overlay = image.copy()
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0.5] = color
    
    # Blend with original image
    result = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
    
    return result

def normalize_points(points: List[Tuple[int, int]], image_size: Tuple[int, int], target_size: int = 1024) -> np.ndarray:
    """Normalize point coordinates for EdgeTAM input."""
    height, width = image_size
    normalized_points = []
    
    for x, y in points:
        # Normalize to [0, 1]
        norm_x = x / width
        norm_y = y / height
        
        # Scale to target size
        scaled_x = norm_x * target_size
        scaled_y = norm_y * target_size
        
        normalized_points.append([scaled_x, scaled_y])
    
    return np.array(normalized_points, dtype=np.float32)

def get_device() -> str:
    """Get the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def patch_edgetam_video_loading():
    """Patch EdgeTAM's video loading to use OpenCV instead of decord."""
    try:
        import sam2.utils.misc as misc

        def load_video_frames_from_video_file_opencv(
            video_path,
            image_size,
            offload_video_to_cpu,
            img_mean=(0.485, 0.456, 0.406),
            img_std=(0.229, 0.224, 0.225),
            compute_device=torch.device("cuda"),
        ):
            """Load video frames using OpenCV instead of decord."""
            import cv2

            img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
            img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            # Get original video dimensions
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            images = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize frame
                frame = cv2.resize(frame, (image_size, image_size))

                # Convert to tensor and normalize to [0, 1]
                frame_tensor = torch.from_numpy(frame).float() / 255.0

                # Permute to (C, H, W) format
                frame_tensor = frame_tensor.permute(2, 0, 1)

                images.append(frame_tensor)

            cap.release()

            if not images:
                raise ValueError(f"No frames found in video: {video_path}")

            # Stack all frames
            images = torch.stack(images, dim=0)  # (T, C, H, W)

            # Move to appropriate device
            if not offload_video_to_cpu:
                images = images.to(compute_device)

            # Apply normalization
            if not offload_video_to_cpu:
                img_mean = img_mean.to(compute_device)
                img_std = img_std.to(compute_device)

            images = images - img_mean
            images = images / img_std

            return images, video_height, video_width

        # Replace the original function
        misc.load_video_frames_from_video_file = load_video_frames_from_video_file_opencv
        print("✓ EdgeTAM video loading patched to use OpenCV")

        # Patch BFloat16 issue on older macOS by monkey-patching torch.bfloat16 conversion
        try:
            import platform

            # Check if we're on macOS with MPS
            if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and
                platform.system() == 'Darwin'):

                # Store original tensor.to method
                original_tensor_to = torch.Tensor.to

                def patched_tensor_to(self, *args, **kwargs):
                    """Patched tensor.to that avoids BFloat16 on MPS when not supported."""
                    # Check if we're trying to convert to bfloat16 on MPS
                    if len(args) > 0 and args[0] == torch.bfloat16 and self.device.type == 'mps':
                        try:
                            return original_tensor_to(self, *args, **kwargs)
                        except TypeError as e:
                            if "MPS BFloat16 is only supported on MacOS 14 or newer" in str(e):
                                print("⚠ BFloat16 not supported on this macOS version, using Float32")
                                # Use float32 instead
                                new_args = (torch.float32,) + args[1:]
                                return original_tensor_to(self, *new_args, **kwargs)
                            else:
                                raise
                    else:
                        return original_tensor_to(self, *args, **kwargs)

                # Apply the patch
                torch.Tensor.to = patched_tensor_to
                print("✓ EdgeTAM BFloat16 compatibility patched for older macOS")

        except Exception as e:
            print(f"⚠ Failed to patch BFloat16 compatibility: {e}")

    except ImportError:
        print("⚠ EdgeTAM not available, skipping video loading patch")
    except Exception as e:
        print(f"⚠ Failed to patch EdgeTAM video loading: {e}")
