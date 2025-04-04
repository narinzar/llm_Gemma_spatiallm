# -*- coding: utf-8 -*-
"""
Step 2: 3D Visualization Script

This script creates a 3D visualization from the object detection results.
It reads the detection data from "detection_results.json" (created by Step 1)
and generates an interactive 3D scene showing the objects with depth.

Execute this script after Step 1.
"""

# Install required packages
# !pip install -q plotly pillow matplotlib numpy

import json
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import matplotlib.pyplot as plt
import os

# Check if detection results exist
DETECTION_FILE = "detection_results.json"
if not os.path.exists(DETECTION_FILE):
    raise FileNotFoundError(
        f"Detection file '{DETECTION_FILE}' not found. Please run Step 1 first."
    )

# Check if image exists
IMAGE_PATH = "1.jpg"
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image file '{IMAGE_PATH}' not found. Please place it in the current directory."
    )

# Load detection results
print(f"Loading detection results from {DETECTION_FILE}...")
with open(DETECTION_FILE, "r") as f:
    detection_data = json.load(f)

detections = detection_data["detections"]
image_dimensions = detection_data["image_dimensions"]

# Load the original image
image = Image.open(IMAGE_PATH)

# Show a summary of detected objects
print(f"Found {len(detections)} objects:")
for i, detection in enumerate(detections):
    print(f"  {i+1}. {detection['class']} (confidence: {detection['confidence']:.2f})")

# Function to estimate depth based on position in the image
def estimate_depth(box, image_height):
    """
    Estimate object depth based on position and size.
    Objects lower in the frame and larger are typically closer to the camera.
    """
    y1, y2 = box[1], box[3]
    height = y2 - y1
    width = box[2] - box[0]
    
    # Get vertical position (objects at bottom are closer)
    y_center = (y1 + y2) / 2
    position_factor = y_center / image_height  # 0 (top) to 1 (bottom)
    
    # Get size (larger objects are typically closer)
    size = width * height
    size_factor = size / (image_dimensions["width"] * image_dimensions["height"])
    
    # Combine factors (position has more weight than size)
    depth = 10 * (1 - (0.7 * position_factor + 0.3 * (size_factor ** 0.5)))
    
    return depth

# Create 3D visualization
print("Creating 3D visualization...")
fig = go.Figure()

# Add a floor plane for reference
floor_x = [0, image_dimensions["width"], image_dimensions["width"], 0]
floor_y = [0, 0, 10, 10]
floor_z = [0, 0, 0, 0]

fig.add_trace(go.Mesh3d(
    x=floor_x,
    y=floor_y,
    z=floor_z,
    i=[0],
    j=[1],
    k=[2],
    color='lightgray',
    opacity=0.5,
    name='floor'
))

# Color map for different objects
colors = {
    'person': 'red',
    'chair': 'blue',
    'dining table': 'brown',
    'sofa': 'purple',
    'potted plant': 'green',
    'tv': 'black',
    'bottle': 'cyan',
    'cup': 'orange',
    'bed': 'pink',
    'dining chair': 'darkblue',
    'door': 'tan',
    'curtain': 'magenta',
    'wall': 'lightgray',
}

# Add each object as a 3D box
for i, detection in enumerate(detections):
    box = detection["box"]
    class_name = detection["class"]
    
    # Extract coordinates
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    # Estimate depth based on position and size
    depth = estimate_depth(box, image_dimensions["height"])
    
    # Determine 3D height based on object type
    obj_height = height
    if class_name.lower() in ['chair', 'dining chair']:
        obj_height = height * 0.8
    elif class_name.lower() == 'dining table':
        obj_height = height * 0.7
    elif class_name.lower() == 'sofa':
        obj_height = height * 0.6
    elif class_name.lower() == 'person':
        obj_height = height * 1.2
    
    # Get color based on object class
    color = colors.get(class_name.lower(), 'gray')
    
    # Create 3D box vertices
    # Front face
    vertices_x = [x1, x2, x2, x1, x1, x2, x2, x1]
    vertices_y = [depth, depth, depth, depth, depth+width/3, depth+width/3, depth+width/3, depth+width/3]
    vertices_z = [0, 0, obj_height, obj_height, 0, 0, obj_height, obj_height]
    
    # Create 3D box faces
    i_indices = [0, 0, 0, 1, 4, 4, 4, 5]
    j_indices = [1, 2, 4, 2, 5, 6, 0, 1]
    k_indices = [2, 3, 7, 3, 6, 7, 3, 2]
    
    # Add the 3D box
    fig.add_trace(go.Mesh3d(
        x=vertices_x,
        y=vertices_y,
        z=vertices_z,
        i=i_indices,
        j=j_indices,
        k=k_indices,
        color=color,
        opacity=0.7,
        name=f"{class_name} ({i})"
    ))
    
    # Add text label
    fig.add_trace(go.Scatter3d(
        x=[x1 + width/2],
        y=[depth + width/6],
        z=[obj_height + 5],
        mode='text',
        text=[class_name],
        textposition='top center',
        name=f"label-{class_name}-{i}"
    ))

# Configure the layout
fig.update_layout(
    title="3D Visualization of Detected Objects",
    scene=dict(
        xaxis_title='X (Width)',
        yaxis_title='Y (Depth)',
        zaxis_title='Z (Height)',
        aspectmode='data',
        camera=dict(
            eye=dict(x=1.5, y=-1.5, z=1)
        )
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)

# Save the figure as HTML for interactive viewing
html_path = "3d_visualization.html"
fig.write_html(html_path)
print(f"Saved interactive 3D visualization to {html_path}")

# Show the figure
fig.show()

# Save alternative visualization angles
camera_positions = [
    {"eye": dict(x=0, y=-2, z=0.5)},  # Side view
    {"eye": dict(x=0, y=0, z=2)},      # Top view
    {"eye": dict(x=2, y=-0.5, z=0.5)}  # Angled view
]

for i, camera in enumerate(camera_positions):
    fig.update_layout(scene_camera=camera)
    angle_html = f"3d_angle_{i+1}.html"
    fig.write_html(angle_html)
    print(f"Saved alternative view {i+1} to {angle_html}")

# Create and save a simple image showing the 3D layout
# This is a simplified top-down view
plt.figure(figsize=(10, 10))
plt.xlim(0, image_dimensions["width"])
plt.ylim(0, 10)  # Depth range

for detection in detections:
    box = detection["box"]
    class_name = detection["class"]
    depth = estimate_depth(box, image_dimensions["height"])
    
    # Get color
    color = colors.get(class_name.lower(), 'gray')
    
    # Draw rectangle from top view
    x1, y1, x2, y2 = box
    plt.fill([x1, x2, x2, x1], [depth, depth, depth+((x2-x1)/3), depth+((x2-x1)/3)], color=color, alpha=0.5)
    plt.text(x1 + (x2-x1)/2, depth + (x2-x1)/6, class_name, ha='center')

plt.title("Top-down View of 3D Scene")
plt.xlabel("X (Width)")
plt.ylabel("Y (Depth)")
plt.gca().set_aspect('equal')
plt.savefig("3d_top_view.jpg")
plt.show()

print("\nStep 2 complete! Now you can run Step 3: Gemma Analysis.")
