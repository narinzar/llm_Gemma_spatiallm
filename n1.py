# -*- coding: utf-8 -*-
"""
Simple Electronic Waste Analysis with Gemma 3

This script:
1. Uses Gemma 3 to directly analyze electronic waste in an image
2. Creates a visualization with the analysis results
3. No complex dependencies for object detection or segmentation

Requirements:
- torch
- transformers
- huggingface-hub
- PIL
- matplotlib
"""

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os

# Configuration
IMAGE_PATH = "1.jpg"  # Your image file

# Check if image exists
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image file '{IMAGE_PATH}' not found. Please place it in the current directory.")

# Load and display the image
print(f"Loading image from {IMAGE_PATH}...")
image = Image.open(IMAGE_PATH)

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.title("Electronic Waste Image")
plt.show()

# Authentication with Hugging Face
print("\nTo analyze with Gemma 3, you need to authenticate with Hugging Face.")

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

# Create a detailed analysis prompt
detailed_prompt = """In this image of electronic waste or discarded electronic items:

1. First, identify each visible electronic device or component
2. For each identified item:
   - Describe its physical condition in detail
   - Estimate the percentage of visible damage
   - Classify it as functional, potentially repairable, or non-functional
   - Note specific damage types (cracks, broken parts, rust, etc.)

3. Create a detailed damage assessment with this format:
ELECTRONIC WASTE ANALYSIS:
-----------------------------
ITEM 1: [Item type]
- Condition: [Detailed description]
- Damage: [Percentage] damaged
- Status: [Functional/Repairable/Non-functional]
- Specific issues: [List of damage/issues]

ITEM 2: [Item type]
...and so on

4. Also include an overall assessment of the waste pile and any environmental concerns.

Be very specific about the conditions you can observe.
"""

print("\nPrepared prompt for Gemma 3:")
print("=" * 80)
print(detailed_prompt)
print("=" * 80)

# Create messages with the image
print("\nAnalyzing with Gemma 3...")
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert in electronic waste assessment and condition analysis."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": detailed_prompt}
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
print("Generating detailed analysis from Gemma 3...")
with torch.inference_mode():
    generation = gemma_model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False
    )
    generation = generation[0][input_len:]

# Decode the response
analysis = gemma_processor.decode(generation, skip_special_tokens=True)

print("\nGemma 3's analysis:")
print("=" * 80)
print(analysis)
print("=" * 80)

# Save the analysis
with open("ewaste_analysis.txt", "w") as f:
    f.write(analysis)

print(f"Saved analysis to ewaste_analysis.txt")

# Create visualization with the analysis
width, height = image.size
analysis_img = Image.new('RGB', (width, height + 400), (0, 0, 0))
analysis_img.paste(image, (0, 0))
draw = ImageDraw.Draw(analysis_img)

# Try to load a font, fallback to default if not available
try:
    font = ImageFont.truetype("arial.ttf", 18)
except IOError:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except IOError:
        font = ImageFont.load_default()

# Extract key parts of the analysis
if "ELECTRONIC WASTE ANALYSIS:" in analysis:
    # Extract the structured part
    analysis_part = analysis.split("ELECTRONIC WASTE ANALYSIS:")[1].strip()
    # Take first part of the analysis (limited to fit in the image)
    analysis_lines = analysis_part.split("\n")[:20]  
else:
    # Just take the first few lines
    analysis_lines = analysis.split("\n")[:20]

# Add title
draw.text((20, height + 10), "ELECTRONIC WASTE ANALYSIS:", fill="white", font=font)

# Add analysis lines
y_offset = height + 40
for line in analysis_lines:
    # Check if it's a header line
    if line.startswith("ITEM") or line.strip() == "" or "----" in line:
        draw.text((20, y_offset), line, fill=(255, 255, 0), font=font)  # Yellow for headers
    else:
        draw.text((20, y_offset), line, fill="white", font=font)
    y_offset += 22
    if y_offset > height + 380:  # Ensure we don't go beyond the image
        break

# Save and display
visual_path = "ewaste_with_analysis.png"
analysis_img.save(visual_path)

print(f"Saved visualization to {visual_path}")

plt.figure(figsize=(10, 14))
plt.imshow(analysis_img)
plt.axis('off')
plt.title("Electronic Waste with Analysis")
plt.show()

print("\nAnalysis complete!")