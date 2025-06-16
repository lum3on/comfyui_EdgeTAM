"""
ComfyUI EdgeTAM Custom Nodes
On-Device Track Anything Model for efficient video object segmentation and tracking.

This package provides ComfyUI nodes for:
- Video object tracking with EdgeTAM
- Single image segmentation
- High-performance inference optimized for consumer hardware

Based on EdgeTAM by Meta Reality Labs (CVPR 2025)
Repository: https://github.com/facebookresearch/EdgeTAM
"""

import os
import sys
import subprocess

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add the cloned EdgeTAM repo to the path if it exists, for editable installs
edge_tam_repo_path = os.path.join(current_dir, "EdgeTAM")
if os.path.isdir(edge_tam_repo_path):
    if edge_tam_repo_path not in sys.path:
        sys.path.insert(0, edge_tam_repo_path)

def install_edgetam():
    """Attempt to automatically install EdgeTAM."""
    print("=" * 60)
    print("WARNING: EdgeTAM not found!")
    print("Attempting to automatically install EdgeTAM...")
    
    install_script = os.path.join(current_dir, "install_edgetam.py")
    
    try:
        # Ensure the script is executable
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", os.path.join(current_dir, "requirements.txt")], check=True)
        
        # Run the installation script
        result = subprocess.run([sys.executable, install_script], check=True, capture_output=True, text=True)
        print(result.stdout)
        
        print("EdgeTAM installation script finished.")
        print("Please restart ComfyUI to apply the changes.")
        print("=" * 60)
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print("ERROR: EdgeTAM automatic installation failed.")
        print(f"Please check the error output below and install manually.")
        print(f"Stderr: {e.stderr}")
        print("=" * 60)
        return False
    except FileNotFoundError:
        print("=" * 60)
        print("ERROR: install_edgetam.py not found!")
        print("Please ensure the installation script is in the correct directory.")
        print("=" * 60)
        return False

# Check for EdgeTAM installation
try:
    import sam2
    EDGETAM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    EDGETAM_AVAILABLE = False
    if not install_edgetam():
        # If installation fails, print manual instructions as a fallback
        print("=" * 60)
        print("Manual Installation Instructions:")
        print("1. Clone EdgeTAM repository:")
        print("   git clone https://github.com/facebookresearch/EdgeTAM.git")
        print("2. Install EdgeTAM:")
        print("   cd EdgeTAM")
        print("   pip install -e .")
        print("3. Restart ComfyUI")
        print("=" * 60)

# Import nodes only if EdgeTAM is available
if EDGETAM_AVAILABLE:
    try:
        from .edgetam_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        # Export for ComfyUI
        __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
        
        print("EdgeTAM ComfyUI nodes loaded successfully!")
        print("Available nodes:")
        for node_name in NODE_DISPLAY_NAME_MAPPINGS.values():
            print(f"  - {node_name}")
            
    except Exception as e:
        print(f"Error loading EdgeTAM nodes: {e}")
        print("Please check your EdgeTAM installation.")
        
        # Provide empty mappings to prevent ComfyUI errors
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}
        
else:
    # Provide empty mappings when EdgeTAM is not available
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Metadata for ComfyUI
WEB_DIRECTORY = "./web"
__version__ = "1.0.0"

# Add API endpoints for the mask editor
try:
    from aiohttp import web
    import server
    import numpy as np
    import io
    import base64
    import folder_paths
    from PIL import Image
    import cv2
    from .edgetam_nodes import INTERACTIVE_SESSIONS, get_image_predictor
    import torch

    @server.PromptServer.instance.routes.post("/edgetam/preview_mask")
    async def preview_mask_handler(request):
        data = await request.json()
        image_b64 = data.get("image").split(',')[1]
        points = data.get("points")
        labels = data.get("labels")

        try:
            # Decode image
            img_data = base64.b64decode(image_b64)
            pil_image = Image.open(io.BytesIO(img_data))
            image_np = np.array(pil_image)

            # Get predictor
            predictor = get_image_predictor()
            if predictor is None:
                return web.Response(status=500, text="Image predictor not available.")

            predictor.set_image(image_np)

            # Predict mask
            with torch.inference_mode():
                masks, _, _ = predictor.predict(
                    point_coords=np.array(points, dtype=np.float32),
                    point_labels=np.array(labels, dtype=np.int32),
                    multimask_output=False
                )
            
            # Get the first mask and convert to an image
            mask = masks[0]
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

            # Convert to base64
            buffer = io.BytesIO()
            mask_pil.save(buffer, format="PNG")
            mask_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return web.json_response({"mask": f"data:image/png;base64,{mask_b64}"})

        except Exception as e:
            import traceback
            print(f"Error generating mask preview: {e}")
            traceback.print_exc()
            return web.Response(status=500, text=f"Error generating preview: {e}")

    @server.PromptServer.instance.routes.get("/edgetam/frame/{node_id}")
    async def get_first_frame(request):
        node_id = request.match_info["node_id"]
        
        try:
            prompt = server.PromptServer.instance.last_prompt
            if not prompt or 'prompt' not in prompt:
                return web.Response(status=400, text="No workflow prompt available. Please queue the workflow once.")

            graph = prompt['prompt']
            
            target_node_info = next((n for n in graph if str(n[0]) == node_id), None)
            if not target_node_info:
                return web.Response(status=404, text=f"Node with ID {node_id} not found in the current workflow.")

            if target_node_info[1] != "InteractiveMaskEditor":
                 return web.Response(status=400, text="The specified node is not an InteractiveMaskEditor.")

            inputs = target_node_info[2].get("inputs", {})
            video_input_link_id = inputs.get("image")
            if not video_input_link_id:
                return web.Response(status=400, text="No image node is connected to the 'image' input.")

            source_node_id = video_input_link_id[0]
            source_node_info = next((n for n in graph if n[0] == source_node_id), None)
            
            if not source_node_info:
                return web.Response(status=404, text="Source video node not found in the graph.")

            widgets_values = source_node_info[2].get('widgets_values')
            if not widgets_values or not isinstance(widgets_values, list) or len(widgets_values) == 0:
                return web.Response(status=400, text="Could not find widget values in the source video node.")
            
            video_filename = widgets_values[0]
            
            input_dir = folder_paths.get_input_directory()
            video_path = os.path.join(input_dir, video_filename)

            if not os.path.exists(video_path):
                return web.Response(status=404, text=f"Video file does not exist: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return web.Response(status=500, text="Failed to open video file with OpenCV.")
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return web.Response(status=500, text="Failed to read the first frame from the video.")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return web.json_response({
                "image": f"data:image/png;base64,{img_str}",
                "width": pil_image.width,
                "height": pil_image.height
            })

        except Exception as e:
            import traceback
            print(f"Error getting first frame for EdgeTAM editor: {e}")
            traceback.print_exc()
            return web.Response(status=500, text=f"An unexpected error occurred: {e}")

    @server.PromptServer.instance.routes.post("/edgetam/save_mask")
    async def save_mask_handler(request):
        data = await request.json()
        session_id = data.get("sessionId")
        mask_data = data.get("maskData")

        if session_id in INTERACTIVE_SESSIONS:
            event, result_holder = INTERACTIVE_SESSIONS[session_id]
            result_holder["result"] = mask_data
            event.set()
            return web.Response(status=200, text="Mask saved successfully.")
        else:
            return web.Response(status=404, text="Session not found.")

    @server.PromptServer.instance.routes.post("/edgetam/cancel_mask")
    async def cancel_mask_handler(request):
        data = await request.json()
        session_id = data.get("sessionId")
        if session_id in INTERACTIVE_SESSIONS:
            event, result_holder = INTERACTIVE_SESSIONS[session_id]
            result_holder["result"] = "cancel"
            event.set()
            return web.Response(status=200, text="Workflow cancelled.")
        else:
            return web.Response(status=404, text="Session not found.")

except ImportError:
    print("Could not import aiohttp. API endpoints for EdgeTAM editor will not be available.")
