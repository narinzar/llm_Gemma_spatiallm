# -*- coding: utf-8 -*-
"""
Gemma 3 Spatial Image Analysis

This script uses Gemma 3 to directly analyze an image and provide spatial descriptions
of objects and their relationships. It focuses on generating detailed spatial descriptions
similar to "The lower 30% of the wall is wet. Water is pooling around the couch..."

Usage: python gemma_spatial_description.py
"""

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from huggingface_hub import login
from PIL import Image
import matplotlib.pyplot as plt
import os

# File configuration
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
plt.title("Input Image")
plt.show()

# Authenticate with Hugging Face
print("\nTo access Gemma 3 models, you need to authenticate with Hugging Face.")
print("You can get your token from https://huggingface.co/settings/tokens")

# Try to get token from environment variable first
hf_token = os.environ.get("HF_TOKEN")
# Authenticate with Hugging Face using token from .env
 
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

# Create a spatial analysis prompt
spatial_prompt = """Please analyze this image focusing on spatial relationships and object conditions.

1. First, identify the main objects in the scene.
2. Then describe their spatial relationships to each other.
3. Mention any notable conditions of the objects or environment.
4. Focus on precise spatial descriptions (above, below, left, right, inside, next to, etc.)
5. Include quantitative spatial estimates when possible (e.g., "The lower 30% of the wall")

Format your response like this:
Detected objects: [list objects]
[Detailed spatial description with multiple sentences about relationships and conditions]

Remember to be precise and detailed in your spatial descriptions.
"""

print("\nPrepared prompt for Gemma 3:")
print("=" * 80)
print(spatial_prompt)
print("=" * 80)

# Create messages with the image
print("\nProcessing image with Gemma 3...")
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a spatial analysis assistant focusing on detailed object relationships and conditions."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": spatial_prompt}
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

# Save the spatial analysis result
with open("spatial_description.txt", "w") as f:
    f.write(response)

print(f"\nSaved spatial description to spatial_description.txt")

# Optional: Create a visualization of the description
try:
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a copy of the image for annotation
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Add the text description to the image
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Create a semi-transparent overlay at the bottom
    width, height = annotated_image.size
    overlay_height = min(300, height // 3)
    overlay = Image.new('RGBA', (width, overlay_height), (0, 0, 0, 180))
    
    # Extract and format the first part of the response (detected objects and first paragraph)
    response_lines = response.split('\n')
    if len(response_lines) >= 2:
        desc_text = '\n'.join(response_lines[:3])  # Take the first few lines
    else:
        desc_text = response[:200]  # Just take the beginning if not enough lines
        
    # Draw text on the overlay
    draw_overlay = ImageDraw.Draw(overlay)
    draw_overlay.text((20, 20), desc_text, fill="white", font=font)
    
    # Paste the overlay at the bottom of the image
    annotated_image.paste(overlay, (0, height - overlay_height), overlay)
    
    # Save and display the annotated image
    annotated_image.save("spatial_description_image.jpg")
    
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.title("Image with Spatial Description")
    plt.show()
    
    print("Created visualization with spatial description: spatial_description_image.jpg")
except Exception as e:
    print(f"Couldn't create visualization: {str(e)}")

print("\nAnalysis complete!")