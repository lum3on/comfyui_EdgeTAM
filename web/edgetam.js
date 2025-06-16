// EdgeTAM ComfyUI Web Extensions
// Provides enhanced UI for EdgeTAM nodes with visual point editor

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Add CSS for the Mask Editor
const style = document.createElement('style');
style.textContent = `
.edgetam-mask-editor-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.8);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
}

.edgetam-mask-editor-content {
    background-color: #222;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.edgetam-mask-editor-canvas-container {
    position: relative;
    border: 1px solid #444;
    background-color: #000;
}

.edgetam-mask-editor-canvas, .edgetam-mask-preview-canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
}

.edgetam-mask-preview-canvas {
    pointer-events: none; /* Allow clicks to pass through to the main canvas */
    opacity: 0.7;
}

.edgetam-mask-editor-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
`;
document.head.appendChild(style);

// Function to create and manage the mask editor UI
function createInteractiveMaskEditor(data) {
    let points = [];
    let labels = [];

    // Create the editor overlay
    const overlay = document.createElement('div');
    overlay.className = 'edgetam-mask-editor-overlay';

    // Create the editor content
    const content = document.createElement('div');
    content.className = 'edgetam-mask-editor-content';
    content.innerHTML = `
        <h2>Interactive Mask Editor</h2>
        <p>Click on the image to add points. Right-click to add an 'exclude' point.</p>
        <div class="edgetam-mask-editor-canvas-container">
            <canvas class="edgetam-mask-editor-canvas" style="z-index: 1;"></canvas>
            <canvas class="edgetam-mask-preview-canvas" style="z-index: 2;"></canvas>
        </div>
        <div class="edgetam-mask-editor-controls">
            <div>
                <button class="edgetam-editor-btn preview-btn">Preview Mask</button>
                <button class="edgetam-editor-btn clear-btn">Clear Points</button>
            </div>
            <div>
                <button class="edgetam-editor-btn cancel-btn" style="background-color: #c53939;">Cancel and Stop Workflow</button>
                <button class="edgetam-editor-btn save-btn">Save and Continue</button>
            </div>
        </div>
    `;

    overlay.appendChild(content);
    document.body.appendChild(overlay);

    const canvas = content.querySelector('.edgetam-mask-editor-canvas');
    const previewCanvas = content.querySelector('.edgetam-mask-preview-canvas');
    const container = content.querySelector('.edgetam-mask-editor-canvas-container');
    const ctx = canvas.getContext('2d');
    const previewCtx = previewCanvas.getContext('2d');
    const image = new Image();

    const draw = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        for (let i = 0; i < points.length; i++) {
            ctx.fillStyle = labels[i] === 1 ? 'rgba(0, 255, 0, 0.7)' : 'rgba(255, 0, 0, 0.7)';
            ctx.beginPath();
            ctx.arc(points[i][0], points[i][1], 5, 0, 2 * Math.PI);
            ctx.fill();
        }
    };

    image.onload = () => {
        // Set container size to match image aspect ratio
        const aspectRatio = data.width / data.height;
        const maxHeight = window.innerHeight * 0.7;
        const maxWidth = window.innerWidth * 0.8;
        let height = maxHeight;
        let width = height * aspectRatio;
        if (width > maxWidth) {
            width = maxWidth;
            height = width / aspectRatio;
        }
        container.style.width = `${width}px`;
        container.style.height = `${height}px`;

        // Set canvas sizes
        canvas.width = data.width;
        canvas.height = data.height;
        previewCanvas.width = data.width;
        previewCanvas.height = data.height;
        
        draw();
    };
    image.src = data.image;

    const addPoint = (e) => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        points.push([Math.round(x), Math.round(y)]);
        labels.push(e.button === 2 ? 0 : 1); // Right-click for exclude
        draw();
    };

    canvas.addEventListener('click', addPoint);
    canvas.addEventListener('contextmenu', addPoint);

    content.querySelector('.clear-btn').addEventListener('click', () => {
        points = [];
        labels = [];
        previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
        draw();
    });

    content.querySelector('.preview-btn').addEventListener('click', async () => {
        if (points.length === 0) {
            alert("Please add at least one point to generate a preview.");
            return;
        }
        try {
            const response = await fetch('/edgetam/preview_mask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: image.src,
                    points: points,
                    labels: labels
                })
            });
            if (response.ok) {
                const result = await response.json();
                const maskImage = new Image();
                maskImage.onload = () => {
                    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
                    previewCtx.drawImage(maskImage, 0, 0, previewCanvas.width, previewCanvas.height);
                };
                maskImage.src = result.mask;
            } else {
                alert("Failed to generate preview: " + await response.text());
            }
        } catch (error) {
            console.error("Error generating preview:", error);
            alert("An error occurred while generating the preview.");
        }
    });

    content.querySelector('.save-btn').addEventListener('click', async () => {
        const maskData = {
            points: points,
            labels: labels,
            width: data.width,
            height: data.height
        };

        try {
            const response = await fetch('/edgetam/save_mask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sessionId: data.sessionId, maskData: maskData })
            });
            if (response.ok) {
                document.body.removeChild(overlay);
            } else {
                alert("Failed to save mask: " + await response.text());
            }
        } catch (error) {
            console.error("Error saving mask:", error);
            alert("An error occurred while saving the mask.");
        }
    });

    content.querySelector('.cancel-btn').addEventListener('click', async () => {
        try {
            await fetch('/edgetam/cancel_mask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sessionId: data.sessionId })
            });
        } catch (error) {
            console.error("Error cancelling workflow:", error);
        } finally {
            document.body.removeChild(overlay);
        }
    });
}

// Register the extension to listen for the backend signal
app.registerExtension({
    name: "EdgeTAM.InteractiveMaskEditor",
    init(app) {
        console.log("[EdgeTAM] Initializing interactive editor listener.");
        // Listen for the custom event sent by the backend
        api.addEventListener("edgetam-open-mask-editor", (e) => {
            console.log("[EdgeTAM] Received open editor event:", e.detail);
            // When the event is received, create the editor UI
            createInteractiveMaskEditor(e.detail);
        });
    }
});

console.log("EdgeTAM ComfyUI extensions loaded");
