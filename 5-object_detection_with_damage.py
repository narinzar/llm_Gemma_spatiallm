# -*- coding: utf-8 -*-
"""
YOLOv5 Object Detection and Damage Analysis (Fixed)

This script:
1. Detects objects in an image using YOLOv5
2. Visualizes them with different colors per class
3. Uses Gemma 3 to analyze potential damage to the objects

This version fixes the image format issue.
"""

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Configuration
IMAGE_PATH = "1.jpg"  # Your image file
DETECTION_THRESHOLD = 0.4
DAMAGE_ANALYSIS = True  # Set to True to analyze damage with Gemma 3

# Colors for different object types (RGB format)
COLORS = {
    "tv": (255, 0, 0),          # Red
    "refrigerator": (0, 255, 0), # Green
    "oven": (0, 0, 255),         # Blue
    "microwave": (255, 255, 0),  # Yellow
    "laptop": (255, 0, 255),     # Magenta
    "cell phone": (0, 255, 255), # Cyan
    "keyboard": (255, 165, 0),   # Orange
    "mouse": (128, 0, 128),      # Purple
    "remote": (0, 128, 0),       # Dark Green
    "default": (128, 128, 128)   # Gray (for unspecified objects)
}

# Check if image exists
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image file '{IMAGE_PATH}' not found. Please place it in the current directory.")

# Load and display the image
print(f"Loading image from {IMAGE_PATH}...")
image = Image.open(IMAGE_PATH)
image_np = np.array(image)

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.title("Original Image")
plt.show()

# Load YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = DETECTION_THRESHOLD  # Set confidence threshold

# Perform inference
print("Performing object detection...")
results = model(image)

# Process results
detections = []
for *box, conf, cls in results.xyxy[0].cpu().numpy():
    x1, y1, x2, y2 = box
    class_name = results.names[int(cls)]
    detections.append({
        "box": [float(x1), float(y1), float(x2), float(y2)],
        "class": class_name,
        "confidence": float(conf)
    })

print(f"Detected {len(detections)} objects:")
for i, detection in enumerate(detections):
    print(f"  {i+1}. {detection['class']} (confidence: {detection['confidence']:.2f})")

# Create visualization image for object detection
# Convert to RGB to ensure compatibility with JPEG format
visualization = image.convert('RGB').copy()
draw = ImageDraw.Draw(visualization)

# Try to load a font, fallback to default if not available
try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

# Draw bounding boxes with different colors for each class
for detection in detections:
    box = detection["box"]
    class_name = detection["class"]
    score = detection["confidence"]
    
    # Get color for this class
    color_rgb = COLORS.get(class_name.lower(), COLORS["default"])
    
    # Convert RGB to hex for PIL
    color_hex = "#{:02x}{:02x}{:02x}".format(*color_rgb)
    
    # Draw rectangle (make it thick)
    box_coords = [int(coord) for coord in box]
    draw.rectangle(box_coords, outline=color_hex, width=3)
    
    # Draw label background
    text = f"{class_name}: {score:.2f}"
    text_bbox = draw.textbbox((box_coords[0], box_coords[1] - 25), text, font=font)
    draw.rectangle(text_bbox, fill=color_hex)
    
    # Draw label text
    draw.text((box_coords[0], box_coords[1] - 25), text, fill="white", font=font)

# Save the visualization
visualization_path = "object_detection.png"  # Changed from .jpg to .png
visualization.save(visualization_path)
print(f"Saved object detection visualization to {visualization_path}")

# Display the visualization
plt.figure(figsize=(10, 10))
plt.imshow(visualization)
plt.axis('off')
plt.title("Object Detection")
plt.show()

# Save detection results to JSON
detection_data = {
    "detections": detections,
    "image_dimensions": {
        "width": image.size[0],
        "height": image.size[1]
    }
}

json_path = "detection_results.json"
with open(json_path, "w") as f:
    json.dump(detection_data, f, indent=2)

print(f"Saved detection results to {json_path}")

# Damage Analysis with Gemma 3 (if enabled)
if DAMAGE_ANALYSIS:
    # Authenticate with Hugging Face
    print("\nTo analyze damage with Gemma 3, you need to authenticate with Hugging Face.")
    
    # Try to get token from environment variable first
    hf_token = os.environ.get("HF_TOKEN")
    
    # If not found, ask for it
    if not hf_token:
        hf_token = input("Enter your Hugging Face token (or set HF_TOKEN env variable): ")
    
    if hf_token:
        login(hf_token)
        print("Successfully logged in to Hugging Face!")
    else:
        print("Warning: No token provided. You may not be able to access Gemma 3 models.")
    
    # Load Gemma 3 model
    print("\nLoading Gemma 3 model...")
    model_checkpoint = "google/gemma-3-4b-it"  # Use 2b if memory is an issue
    print(f"Using model: {model_checkpoint}")
    
    try:
        # Load model with reduced precision to save memory
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            model_checkpoint, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
        )
        gemma_processor = AutoProcessor.from_pretrained(model_checkpoint)
        print("Successfully loaded Gemma 3 model!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    # Create a damage analysis prompt
    detected_objects_str = ", ".join([d["class"] for d in detections])
    damage_prompt = f"""In this image, the following objects have been detected: {detected_objects_str}.

Please analyze the image and tell me:
1. If any of these objects appear to be damaged
2. For any damaged objects, describe the specific type and extent of damage
3. Estimate what percentage of each object is damaged (if applicable)
4. Note any visible signs of wear, corrosion, breakage, etc.

Format your analysis like this:
DAMAGE ANALYSIS:
[Object 1]: [Damage description] [Damage percentage]
[Object 2]: [Damage description] [Damage percentage]
...

If there's no visible damage to an object, simply state "No visible damage"."""

    print("\nPrepared prompt for Gemma 3:")
    print("=" * 80)
    print(damage_prompt)
    print("=" * 80)
    
    # Create messages with the image
    print("\nAnalyzing damage with Gemma 3...")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a damage assessment assistant focusing on detailed analysis of object conditions."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": damage_prompt}
            ]
        }
    ]
    
    # Process the messages
    inputs = gemma_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(gemma_model.device)
    
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate response
    print("Generating response from Gemma 3...")
    with torch.inference_mode():
        generation = gemma_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False
        )
        generation = generation[0][input_len:]
    
    # Decode the response
    damage_analysis = gemma_processor.decode(generation, skip_special_tokens=True)
    
    print("\nGemma 3's damage analysis:")
    print("=" * 80)
    print(damage_analysis)
    print("=" * 80)
    
    # Save the damage analysis
    with open("damage_analysis.txt", "w") as f:
        f.write(damage_analysis)
    
    print(f"Saved damage analysis to damage_analysis.txt")
    
    # Create final visualization with both detection and damage analysis
    final_viz = visualization.copy()
    
    # Add damage analysis as text at the bottom
    width, height = final_viz.size
    analysis_img = Image.new('RGB', (width, height + 300), (0, 0, 0))
    analysis_img.paste(final_viz, (0, 0))
    draw_final = ImageDraw.Draw(analysis_img)
    
    # Extract key parts of the damage analysis
    analysis_lines = []
    if "DAMAGE ANALYSIS:" in damage_analysis:
        # Extract the structured part
        analysis_part = damage_analysis.split("DAMAGE ANALYSIS:")[1].strip()
        analysis_lines = analysis_part.split("\n")[:6]  # Limit to first 6 lines
    else:
        # Just take the first few lines
        analysis_lines = damage_analysis.split("\n")[:6]
    
    # Add title
    draw_final.text((20, height + 10), "DAMAGE ANALYSIS:", fill="white", font=font)
    
    # Add analysis lines
    y_offset = height + 40
    for line in analysis_lines:
        draw_final.text((20, y_offset), line, fill="white", font=font)
        y_offset += 30
    
    # Save and display
    final_path = "object_detection_with_damage.png"  # Changed from .jpg to .png
    analysis_img.save(final_path)
    
    print(f"Saved final visualization to {final_path}")
    
    plt.figure(figsize=(10, 12))
    plt.imshow(analysis_img)
    plt.axis('off')
    plt.title("Object Detection with Damage Analysis")
    plt.show()

print("\nAnalysis complete!")