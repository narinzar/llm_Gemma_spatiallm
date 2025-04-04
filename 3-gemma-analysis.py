# -*- coding: utf-8 -*-
"""
Step 3: Gemma 3 Spatial Analysis Script

This script uses the Gemma 3 multimodal model to analyze the spatial arrangement
of objects in the annotated image. It reads the detection data, asks Gemma 3 to
interpret the spatial relationships, and saves the analysis results.

Execute this script after Step 1 (Step 2 is optional).
"""

# Install required packages
!pip install -q git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
!pip install -q torch pillow matplotlib numpy

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
import re

# File paths
DETECTION_FILE = "detection_results.json"
ANNOTATED_IMAGE = "annotated_image.jpg"
ORIGINAL_IMAGE = "1.jpg"

# Check if required files exist
if not os.path.exists(DETECTION_FILE):
    raise FileNotFoundError(
        f"Detection file '{DETECTION_FILE}' not found. Please run Step 1 first."
    )

if not os.path.exists(ANNOTATED_IMAGE):
    raise FileNotFoundError(
        f"Annotated image '{ANNOTATED_IMAGE}' not found. Please run Step 1 first."
    )

# Load detection results
print(f"Loading detection results from {DETECTION_FILE}...")
with open(DETECTION_FILE, "r") as f:
    detection_data = json.load(f)

detections = detection_data["detections"]
image_dimensions = detection_data["image_dimensions"]

# Print detected objects
print(f"Found {len(detections)} objects:")
for i, detection in enumerate(detections):
    print(f"  {i+1}. {detection['class']} (confidence: {detection['confidence']:.2f})")

# Load the annotated image
annotated_image = Image.open(ANNOTATED_IMAGE)

# Display the annotated image
plt.figure(figsize=(10, 10))
plt.imshow(annotated_image)
plt.axis('off')
plt.title("Annotated Image for Gemma 3 Analysis")
plt.show()

# Authenticate with Hugging Face (you'll need to provide your token)
print("\nTo access Gemma 3 models, you need to authenticate with Hugging Face.")
print("You can get your token from https://huggingface.co/settings/tokens")

# Try to get token from environment variable first
import os
hf_token = os.environ.get("HF_TOKEN")

# If not found, ask for it
if not hf_token:
    hf_token = input("Enter your Hugging Face token (or set HF_TOKEN env variable): ")

if hf_token:
    login(hf_token)
    print("Successfully logged in to Hugging Face!")
else:
    print("Warning: No token provided. You may not be able to access Gemma 3 models.")

# Function to prepare spatial prompt
def prepare_spatial_prompt(detections, image_dimensions):
    """
    Create a detailed prompt about spatial relationships for Gemma 3
    """
    # Extract object classes with their positions
    objects_with_positions = []
    for detection in detections:
        box = detection["box"]
        class_name = detection["class"]
        
        # Calculate center point
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        # Determine rough position in image
        position_x = "left" if center_x < image_dimensions["width"] / 3 else "right" if center_x > 2 * image_dimensions["width"] / 3 else "center"
        position_y = "top" if center_y < image_dimensions["height"] / 3 else "bottom" if center_y > 2 * image_dimensions["height"] / 3 else "middle"
        
        position = f"{position_y} {position_x}"
        objects_with_positions.append(f"{class_name} ({position})")
    
    # Create prompt
    prompt = f"""This image contains the following detected objects with their approximate positions:
{', '.join(objects_with_positions)}

Please analyze the spatial relationships between these objects and describe:
1. The overall layout of the scene (what kind of room/environment is this?)
2. The relative positions of objects to each other
3. How these objects might be arranged in 3D space
4. Any functional relationships between objects (e.g., a chair positioned at a table)

After your analysis, please provide a structured JSON representation of the spatial relationships
with the following format:
```json
{{
  "scene_type": "room type or environment",
  "spatial_relationships": [
    {{
      "object1": "object name",
      "relation": "relationship (e.g., next to, above, on, under)",
      "object2": "object name"
    }}
  ],
  "functional_groups": [
    {{
      "group_name": "name of functional area (e.g., dining area)",
      "objects": ["object1", "object2"]
    }}
  ]
}}
```

Focus on spatial understanding rather than visual details."""

    return prompt

# Load Gemma 3 model
print("\nLoading Gemma 3 model...")
print("This may take a few minutes depending on your internet connection and hardware.")

model_checkpoint = "google/gemma-3-4b-it"  # You can try 2b if memory is an issue
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
    print("\nIf you're getting authentication errors, make sure your Hugging Face token")
    print("has access to the Gemma 3 models. You may need to accept the license at:")
    print("https://huggingface.co/google/gemma-3-4b-it")
    raise

# Prepare prompt for Gemma 3
prompt = prepare_spatial_prompt(detections, image_dimensions)
print("\nPrepared prompt for Gemma 3:")
print("=" * 80)
print(prompt)
print("=" * 80)

# Create messages with the annotated image
print("\nProcessing image with Gemma 3...")
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant with spatial reasoning abilities."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": annotated_image},
            {"type": "text", "text": prompt}
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
response = gemma_processor.decode(generation, skip_special_tokens=True)

print("\nGemma 3's spatial analysis:")
print("=" * 80)
print(response)
print("=" * 80)

# Extract JSON from the response if present
def extract_json_from_response(response):
    """Extract JSON structure from Gemma's response if present"""
    # Look for JSON block in markdown format
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Found JSON block but couldn't parse it.")
    
    # Alternative: look for { } structure
    try:
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        pass
    
    return None

# Extract JSON
spatial_json = extract_json_from_response(response)
if spatial_json:
    print("\nExtracted spatial relationships JSON:")
    print(json.dumps(spatial_json, indent=2))
else:
    print("\nCouldn't extract JSON from the response.")

# Save complete analysis results
analysis_results = {
    "detections": detections,
    "image_dimensions": image_dimensions,
    "gemma_prompt": prompt,
    "gemma_response": response,
    "spatial_json": spatial_json
}

analysis_path = "spatial_analysis_results.json"
with open(analysis_path, "w") as f:
    json.dump(analysis_results, f, indent=2)

print(f"\nSaved complete analysis results to {analysis_path}")

# Create a visualization of the spatial relationships
if spatial_json and "spatial_relationships" in spatial_json:
    try:
        import networkx as nx
        
        print("\nCreating spatial relationship visualization...")
        
        # Create a graph
        G = nx.Graph()
        
        # Add nodes (objects)
        object_set = set()
        for rel in spatial_json["spatial_relationships"]:
            object_set.add(rel["object1"])
            object_set.add(rel["object2"])
        
        for obj in object_set:
            G.add_node(obj)
        
        # Add edges (relationships)
        for rel in spatial_json["spatial_relationships"]:
            G.add_edge(rel["object1"], rel["object2"], label=rel["relation"])
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", 
                node_size=3000, font_size=10, font_weight="bold")
        
        # Add edge labels
        edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title("Spatial Relationships Between Objects")
        plt.tight_layout()
        plt.savefig("spatial_relationships_graph.jpg")
        plt.show()
        
        print("Saved spatial relationships visualization to spatial_relationships_graph.jpg")
    except ImportError:
        print("Networkx library not found. Skipping relationship visualization.")
        print("To install: pip install networkx")

print("\nStep 3 complete! You now have a full spatial analysis of your image.")
print("\nSummary of files created:")
print("  - detection_results.json: Object detection results")
print("  - annotated_image.jpg: Image with object detection bounding boxes")
print("  - spatial_analysis_results.json: Complete Gemma 3 spatial analysis")
if os.path.exists("3d_visualization.html"):
    print("  - 3d_visualization.html: Interactive 3D visualization")
if os.path.exists("spatial_relationships_graph.jpg"):
    print("  - spatial_relationships_graph.jpg: Visualization of spatial relationships")
