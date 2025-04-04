# -*- coding: utf-8 -*-
"""
Step 1: Object Detection Script

This script performs object detection on an image file named "1.jpg" in your directory.
It saves the detected objects with bounding boxes as "annotated_image.jpg" and
exports the detection data as "detection_results.json".

Execute this script first.
"""

# Install required packages
!pip install -q git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
!pip install -q torch torchvision
!pip install -q supervision pillow matplotlib numpy

import torch
from transformers import YolosForObjectDetection, YolosImageProcessor
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os

# Define image path (assumes 1.jpg exists in the current directory)
IMAGE_PATH = "1.jpg"

# Check if the image exists
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image file '{IMAGE_PATH}' not found. Please place it in the current directory.")

# Load image
print(f"Loading image from {IMAGE_PATH}...")
image = Image.open(IMAGE_PATH)

# Display original image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.title("Original Image")
plt.show()

# Load YOLO model for object detection
print("Loading object detection model...")
yolo_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
yolo_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
yolo_model.to(device)

# Perform object detection
print("Performing object detection...")
inputs = yolo_processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = yolo_model(**inputs)

# Process YOLO results
target_sizes = torch.tensor([image.size[::-1]])
results = yolo_processor.post_process_object_detection(
    outputs, threshold=0.5, target_sizes=target_sizes)[0]

# Get bounding boxes, labels, and scores
boxes = results["boxes"].cpu().detach().numpy()
labels = results["labels"].cpu().detach().numpy()
scores = results["scores"].cpu().detach().numpy()
id2label = yolo_model.config.id2label

# Create a list of detections
detections = []
for box, label, score in zip(boxes, labels, scores):
    class_name = id2label[label.item()]
    detections.append({
        "box": box.tolist(),
        "class": class_name,
        "confidence": float(score)
    })

print(f"Detected {len(detections)} objects:")
for i, detection in enumerate(detections):
    print(f"  {i+1}. {detection['class']} (confidence: {detection['confidence']:.2f})")

# Display image with bounding boxes
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)

# Draw bounding boxes and labels
for detection in detections:
    box = detection["box"]
    class_name = detection["class"]
    score = detection["confidence"]
    
    # Create rectangle patch
    rect = patches.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    
    # Add label
    ax.text(
        box[0], box[1] - 5, f"{class_name}: {score:.2f}",
        color='white', bbox=dict(facecolor='red', alpha=0.5)
    )

ax.axis('off')
plt.title("Detected Objects")
plt.tight_layout()
plt.savefig("matplotlib_annotated.jpg")  # Save using matplotlib
plt.show()

# Create a better annotated image using PIL
annotated_image = image.copy()
draw = ImageDraw.Draw(annotated_image)

# Try to load a font, fallback to default if not available
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

# Draw bounding boxes and labels
for detection in detections:
    box = detection["box"]
    class_name = detection["class"]
    score = detection["confidence"]
    
    # Convert to integers for PIL
    box = [int(coord) for coord in box]
    
    # Draw rectangle
    draw.rectangle(box, outline="red", width=3)
    
    # Draw label background
    text = f"{class_name}: {score:.2f}"
    text_bbox = draw.textbbox((box[0], box[1] - 25), text, font=font)
    draw.rectangle(text_bbox, fill="red")
    
    # Draw label text
    draw.text((box[0], box[1] - 25), text, fill="white", font=font)

# Save the annotated image
annotated_image_path = "annotated_image.jpg"
annotated_image.save(annotated_image_path)
print(f"Saved annotated image to {annotated_image_path}")

# Display the annotated image
plt.figure(figsize=(10, 10))
plt.imshow(annotated_image)
plt.axis('off')
plt.title("Annotated Image")
plt.show()

# Save detection results to JSON
detection_data = {
    "detections": detections,
    "image_dimensions": {
        "width": image.width,
        "height": image.height
    }
}

json_path = "detection_results.json"
with open(json_path, "w") as f:
    json.dump(detection_data, f, indent=2)

print(f"Saved detection results to {json_path}")
print("\nStep 1 complete! Now you can run Step 2: 3D Visualization or Step 3: Gemma Analysis.")
