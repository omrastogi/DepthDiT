import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Load the DINOv2 model
logging.info("Loading the DINOv2 model...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg').to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(518),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Function to process images and save embeddings
def process_images(csv_path, output_file):
    # Read the CSV file and extract image paths
    df = pd.read_csv(csv_path)
    image_paths = df['rgb_filepath'].tolist()

    embeddings = {}
    logging.info(f"Found {len(image_paths)} images in the CSV.")

    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load and preprocess the image
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

            with torch.no_grad():
                # Compute the embedding
                embedding = model(img_tensor)
                embeddings[image_path] = embedding.cpu().numpy().tolist()

        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            continue

    # Save embeddings to a file
    logging.info(f"Saving embeddings to {output_file}...")
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as f:
        json.dump(embeddings, f)

    logging.info("Processing complete.")

# Input and output paths
csv_path = "csv/filtered_data.csv"  # Path to the CSV file with image paths
output_file = "checkpoint/dinov2_embeddings.json"  # Path to save embeddings

# Process and save embeddings
process_images(csv_path, output_file)
