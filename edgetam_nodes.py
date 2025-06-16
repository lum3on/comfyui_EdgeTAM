"""
EdgeTAM ComfyUI Nodes
Main node implementations for EdgeTAM video tracking and image segmentation.
"""

import numpy as np
import torch
from PIL import Image

# Import EdgeTAM components
try:
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    EDGETAM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    EDGETAM_AVAILABLE = False

try:
    from .edgetam_utils import (
        get_model_path, get_device,
        tensor_to_pil, pil_to_tensor, mask_to_tensor,
        apply_mask_overlay, patch_edgetam_video_loading
    )
except ImportError:
    # Fallback for direct execution
    from edgetam_utils import (
        get_model_path, get_device,
        tensor_to_pil, pil_to_tensor, mask_to_tensor,
        apply_mask_overlay, patch_edgetam_video_loading
    )


# Global holder for a lightweight image predictor for previews
IMAGE_PREDICTOR = None

def get_image_predictor():
    """
    Initializes and returns a singleton SAM2ImagePredictor instance.
    """
    global IMAGE_PREDICTOR
    if IMAGE_PREDICTOR is None:
        print("Initializing EdgeTAM Image Predictor for previews...")
        try:
            model_path = get_model_path()
            device = get_device()
            sam2_model = build_sam2(
                config_file="edgetam.yaml",
                ckpt_path=model_path,
                device=device
            )
            IMAGE_PREDICTOR = SAM2ImagePredictor(sam2_model)
            print("EdgeTAM Image Predictor loaded successfully.")
        except Exception as e:
            print(f"Error initializing EdgeTAM Image Predictor: {e}")
            return None
    return IMAGE_PREDICTOR

class EdgeTAMVideoTracker:
    """
    EdgeTAM Video Object Tracking Node
    
    Tracks objects across video frames using point or box prompts.
    Optimized for real-time performance on consumer hardware.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),  # Batch of video frames
            },
            "optional": {
                "mask_data": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": "Mask data from the editor (hidden)",
                    "input": "hidden"
                }),
                "model_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to EdgeTAM checkpoint (auto-download if empty)"
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto"
                }),
                "max_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000,
                    "tooltip": "Maximum frames to process (0 = all)"
                }),
                "overlay_masks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply colored overlay to show segmentation"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("tracked_frames", "masks", "overlay_frames")
    FUNCTION = "track_video"
    CATEGORY = "EdgeTAM"
    
    def __init__(self):
        self.predictor = None
        self.current_model_path = None
        self.current_device = None
    
    def load_model(self, model_path: str, device: str):
        """Load EdgeTAM model if not already loaded."""
        if device == "auto":
            device = get_device()
            
        # Check if we need to reload the model
        if (self.predictor is None or
            self.current_model_path != model_path or
            self.current_device != device):

            print(f"Loading EdgeTAM model on {device}...")

            # Patch EdgeTAM video loading to use OpenCV instead of decord
            patch_edgetam_video_loading()

            # Get model path
            if not model_path:
                model_path = get_model_path()

            # Use EdgeTAM config name (Hydra will find it in sam2 package)
            config_name = "edgetam.yaml"

            # Build video predictor
            self.predictor = build_sam2_video_predictor(
                config_file=config_name,
                ckpt_path=model_path,
                device=device
            )
            
            self.current_model_path = model_path
            self.current_device = device
            print("EdgeTAM model loaded successfully!")
    
    def track_video(self, video_frames, mask_data="{}", model_path="", device="auto", max_frames=0, overlay_masks=True):
        """Track objects in video frames."""

        # Check if EdgeTAM is available
        if not EDGETAM_AVAILABLE:
            raise RuntimeError("EdgeTAM is not installed. Please run: python install_edgetam.py")

        # Load model
        self.load_model(model_path, device)

        # Parse mask data from the editor
        try:
            import json
            mask_json = json.loads(mask_data)
            points = mask_json.get("points", [])
            labels = mask_json.get("labels", [])
        except (json.JSONDecodeError, AttributeError):
            # Fallback to default if mask data is invalid
            points = [[100, 100]]
            labels = [1]

        if not points or not labels or len(points) != len(labels):
            # If no valid points, we can't proceed with tracking.
            # Return empty/original frames.
            print("Warning: No valid points received from mask editor. Returning original frames.")
            return (video_frames, torch.zeros(1, video_frames.shape[1], video_frames.shape[2], 1), video_frames)
        
        # Convert video frames to list of PIL images
        batch_size = video_frames.shape[0]
        if max_frames > 0:
            batch_size = min(batch_size, max_frames)

        frames_list = []
        for i in range(batch_size):
            frame_tensor = video_frames[i]
            pil_frame = tensor_to_pil(frame_tensor)
            frames_list.append(pil_frame)

        import tempfile
        import cv2

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            temp_video_path = tmp_file.name

        try:
            # Save frames as video using OpenCV
            if frames_list:
                # Get dimensions from first frame
                first_frame = frames_list[0]
                width, height = first_frame.size  # PIL size is (width, height)

                print(f"Creating video with dimensions: {width}x{height}, frames: {len(frames_list)}")

                # Use a more compatible codec
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))

                if not out.isOpened():
                    raise RuntimeError("Failed to create video writer")

                for i, frame in enumerate(frames_list):
                    # Ensure all frames have the same size
                    if frame.size != (width, height):
                        print(f"Resizing frame {i} from {frame.size} to {(width, height)}")
                        frame = frame.resize((width, height))

                    # Convert PIL to OpenCV format (BGR)
                    frame_array = np.array(frame)
                    bgr_frame = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)

                out.release()
                print(f"Video created successfully with {len(frames_list)} frames")

            # Initialize inference state with video file
            inference_state = self.predictor.init_state(video_path=temp_video_path)
            
            # Add prompts on first frame
            points_array = np.array(points, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int32)

            # Get video dimensions for coordinate normalization
            first_frame = frames_list[0]
            video_width, video_height = first_frame.size

            print(f"Video dimensions: {video_width}x{video_height}")
            print(f"Input points: {points_array}")
            print(f"Input labels: {labels_array}")

            # Normalize coordinates to [0, 1] range if they appear to be pixel coordinates
            normalized_points = points_array.copy()
            if np.any(points_array > 1.0):
                print("Normalizing pixel coordinates to [0, 1] range")
                normalized_points[:, 0] = points_array[:, 0] / video_width
                normalized_points[:, 1] = points_array[:, 1] / video_height
                print(f"Normalized points: {normalized_points}")

            # Add points to first frame
            _, _, _ = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=0,  # Single object tracking
                points=normalized_points,
                labels=labels_array,
                normalize_coords=False  # We already normalized
            )
            
            # Propagate through video
            output_masks = []
            output_frames = []
            overlay_frames = []

            print(f"Starting propagation through {len(frames_list)} frames...")

            for frame_idx, _, video_res_masks in self.predictor.propagate_in_video(inference_state):
                if frame_idx >= len(frames_list):
                    break

                # Get original frame
                original_frame = frames_list[frame_idx]
                frame_array = np.array(original_frame)

                # Get mask for first object (handle potential dimension issues)
                if len(video_res_masks) > 0:
                    mask = video_res_masks[0].cpu().numpy()  # Shape could be (H, W) or (1, H, W)

                    # Handle different mask shapes
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        # Remove batch dimension: (1, H, W) -> (H, W)
                        mask = mask.squeeze(0)
                    elif mask.ndim != 2:
                        print(f"Unexpected mask shape: {mask.shape}, using first 2D slice")
                        mask = mask[0] if mask.ndim > 2 else mask

                    # Ensure mask dimensions match frame dimensions
                    frame_h, frame_w = frame_array.shape[:2]
                    mask_h, mask_w = mask.shape

                    if (mask_h, mask_w) != (frame_h, frame_w):
                        print(f"Resizing mask from {mask.shape} to {(frame_h, frame_w)}")
                        # Resize mask to match frame dimensions
                        import cv2
                        # OpenCV resize expects (width, height) not (height, width)
                        mask = cv2.resize(mask.astype(np.float32), (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                        mask = mask.astype(bool)
                else:
                    # Create empty mask if no results
                    frame_h, frame_w = frame_array.shape[:2]
                    mask = np.zeros((frame_h, frame_w), dtype=bool)

                # Convert mask to tensor format
                mask_tensor = mask_to_tensor(mask.astype(np.float32))
                print(f"DEBUG: Appending mask tensor with shape: {mask_tensor.shape}, dtype: {mask_tensor.dtype}")
                output_masks.append(mask_tensor)

                # Create overlay if requested
                if overlay_masks:
                    overlay_frame = apply_mask_overlay(
                        frame_array, mask,
                        color=(0, 255, 0), alpha=0.6
                    )
                    overlay_tensor = pil_to_tensor(Image.fromarray(overlay_frame))
                    overlay_frames.append(overlay_tensor)
                else:
                    overlay_frames.append(pil_to_tensor(original_frame))

                # Keep original frame
                output_frames.append(pil_to_tensor(original_frame))

                print(f"Processed frame {frame_idx + 1}/{len(frames_list)}")
            
            # Stack results
            if output_frames:
                tracked_frames = torch.cat(output_frames, dim=0)
                print(f"DEBUG: Collected {len(output_masks)} masks.")
                masks_batch = torch.cat(output_masks, dim=0)
                print(f"DEBUG: Final masks_batch shape: {masks_batch.shape}")
                overlay_batch = torch.cat(overlay_frames, dim=0)
            else:
                # Return empty tensors if no results
                tracked_frames = torch.zeros_like(video_frames[:1])
                masks_batch = torch.zeros(1, video_frames.shape[1], video_frames.shape[2], 1)
                overlay_batch = torch.zeros_like(video_frames[:1])
            
            return (tracked_frames, masks_batch, overlay_batch)

        except Exception as e:
            print(f"Error during video tracking: {e}")
            import traceback
            traceback.print_exc()

            # Return empty tensors if tracking fails
            tracked_frames = torch.zeros_like(video_frames[:1])
            masks_batch = torch.zeros(1, video_frames.shape[1], video_frames.shape[2], 1)
            overlay_batch = torch.zeros_like(video_frames[:1])
            return (tracked_frames, masks_batch, overlay_batch)

        finally:
            # Clean up temporary video file
            try:
                import os
                os.unlink(temp_video_path)
            except:
                pass
    



import asyncio
import server
import uuid
from aiohttp import web
import json
import io
import base64
import threading

# A dictionary to hold threading events for each interactive session
INTERACTIVE_SESSIONS = {}

class InteractiveMaskEditor:
    """
    An interactive node to create or edit a mask for video tracking.
    Pauses the workflow and opens a UI for the user to draw points.
    Can also accept a pre-defined mask as a JSON string for automation.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "optional_mask_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional. A JSON string to bypass the editor for automation. Format: {\"points\": [[x1, y1], ...], \"labels\": [1, 0, ...]}"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mask_data",)
    FUNCTION = "execute"
    CATEGORY = "EdgeTAM"

    def execute(self, image, optional_mask_data=None):
        # Automation Path: If mask data is provided, use it directly.
        if optional_mask_data and optional_mask_data.strip() and optional_mask_data.strip() != "{}":
            print("EdgeTAM Mask Editor: Using provided mask data (automation mode).")
            return (optional_mask_data,)

        # Interactive Path: No mask data, so open the editor.
        print("EdgeTAM Mask Editor: Launching interactive editor.")
        
        # Prepare the image for the frontend
        first_frame_tensor = image[0]
        pil_image = tensor_to_pil(first_frame_tensor)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Create a unique ID for this editing session
        session_id = str(uuid.uuid4())
        
        # Use a threading.Event to pause the worker thread
        event = threading.Event()
        result_holder = {}  # Use a dict to hold the result
        INTERACTIVE_SESSIONS[session_id] = (event, result_holder)

        # Schedule the websocket send on the main asyncio loop
        main_loop = server.PromptServer.instance.loop
        asyncio.run_coroutine_threadsafe(
            server.PromptServer.instance.send("edgetam-open-mask-editor", {
                "sessionId": session_id,
                "image": f"data:image/png;base64,{img_str}",
                "width": pil_image.width,
                "height": pil_image.height
            }),
            main_loop
        )

        # Wait for the event to be set by the web request
        try:
            event.wait()  # This is a blocking call
            result = result_holder.get("result")
            
            if result == "cancel":
                raise Exception("Workflow cancelled by user from Mask Editor.")

            if result is None:
                print("Warning: Mask editor closed without providing data. Returning empty mask.")
                return (json.dumps({"points": [], "labels": []}),)

            return (json.dumps(result),)
        finally:
            # Clean up the session from the dictionary
            if session_id in INTERACTIVE_SESSIONS:
                del INTERACTIVE_SESSIONS[session_id]

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "EdgeTAMVideoTracker": EdgeTAMVideoTracker,
    "InteractiveMaskEditor": InteractiveMaskEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EdgeTAMVideoTracker": "EdgeTAM Video Tracker",
    "InteractiveMaskEditor": "Interactive Mask Editor",
}
