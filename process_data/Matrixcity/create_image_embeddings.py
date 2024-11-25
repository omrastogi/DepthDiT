# import os
# import torch
# import pandas as pd
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from torchvision import transforms
# import logging
# import json

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Check for CUDA availability
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# logging.info(f"Using device: {device}")

# # Load the DINOv2 model
# logging.info("Loading the DINOv2 model...")
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
# model.eval()

# # Define image transformations
# transform = transforms.Compose([
#     transforms.Resize(518),
#     transforms.CenterCrop(518),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])

# # Function to process images and save embeddings
# def process_images(csv_path, output_file):
#     # Read the CSV file and extract image paths
#     df = pd.read_csv(csv_path)
#     image_paths = df['rgb_filepath'].tolist()

#     embeddings = {}
#     logging.info(f"Found {len(image_paths)} images in the CSV.")

    
#     for image_path in tqdm(image_paths, desc="Processing images"):
#         try:
#             # Load and preprocess the image
#             img = Image.open(image_path).convert('RGB')
#             img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

#             with torch.no_grad():
#                 # Compute the embedding
#                 embedding = model(img_tensor)
#                 embeddings[image_path] = embedding.cpu().numpy().tolist()

#         except Exception as e:
#             logging.error(f"Error processing {image_path}: {e}")
#             continue

#     # Save embeddings to a file
#     logging.info(f"Saving embeddings to {output_file}...")
#     output_dir = os.path.dirname(output_file)
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     with open(output_file, 'w') as f:
#         json.dump(embeddings, f)

#     logging.info("Processing complete.")

# # Input and output paths
# csv_path = "csv/filtered_data.csv"  # Path to the CSV file with image paths
# output_file = "checkpoint/dinov2_embeddings.json"  # Path to save embeddings

# # Process and save embeddings
# process_images(csv_path, output_file)
# -------------------------------------------------------------------------------------------------
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from torch.cuda.amp import autocast


BATCH_SIZE=128
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

# Function to load and preprocess a single image
def load_and_transform_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img)
        return image_path, img_tensor
    except Exception as e:
        logging.error(f"Error loading {image_path}: {e}")
        return image_path, None

# Function to infer a batch of images
def infer_batch(batch_images, model, device):
    
    with torch.no_grad():
        with autocast():
            embeddings = model(batch_images).cpu().numpy()
    return embeddings

# Function to process images and save embeddings
def process_images(csv_path, output_file, batch_size=32):
    # Read the CSV file and extract image paths
    df = pd.read_csv(csv_path)
    image_paths = df['rgb_filepath'].tolist()

    logging.info(f"Found {len(image_paths)} images in the CSV.")
    logging.info(f"Processing images in batches of {batch_size}...")

    # Prepare output directory
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    # Process images incrementally in batches
    with open(output_file, 'w') as f:
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]

            # Load and preprocess images in the current batch
            batch_images = []
            import time
            start_time = time.time()
            # for image_path in batch_paths:
            #     load_and_transform_image(image_path)            
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                batch_images = list(executor.map(load_and_transform_image, batch_paths))
                end_time = time.time()

            logging.info(f"Time taken to load and transform batch: {end_time - start_time} seconds")

            # Filter out any invalid entries
            batch_images = [img[1] for img in batch_images if img is not None]

            if not batch_images:
                logging.warning(f"No valid images in batch {i // batch_size}. Skipping...")
                continue

            # Stack images into a batch tensor
            try:
                batch_tensors = torch.stack(batch_images).to(device)
            except TypeError as e:
                logging.error(f"Error stacking tensors in batch {i // batch_size}: {e}")
                continue
            
            # Infer embeddings
            start_time = time.time()
            batch_embeddings = infer_batch(batch_tensors, model, device)
            end_time = time.time()
            logging.info(f"Batch inference time: {end_time - start_time:.2f} seconds")

            # Save embeddings incrementally
            for path, embedding in zip(batch_paths, batch_embeddings):
                entry = {path: embedding.tolist()}
                f.write(json.dumps(entry) + "\n")

    logging.info("Processing complete.")


# Input and output paths
csv_path = "csv/filtered_data.csv"  # Path to the CSV file with image paths
output_file = "checkpoint/dinov2_embeddings.jsonl"  # Path to save embeddings

# Process and save embeddings
process_images(csv_path, output_file, BATCH_SIZE)
# ----------------------------------------------------------------------------------------------------------------

# import os
# import torch
# import pandas as pd
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from torchvision import transforms
# import logging
# import json
# from concurrent.futures import ThreadPoolExecutor
# from torch.cuda.amp import autocast
# import multiprocessing

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Define image transformations
# transform = transforms.Compose([
#     transforms.Resize(518),
#     transforms.CenterCrop(518),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])

# # Function to load and preprocess a single image
# def load_and_transform_image(image_path):
#     try:
#         img = Image.open(image_path).convert('RGB')
#         img_tensor = transform(img)
#         return image_path, img_tensor
#     except Exception as e:
#         logging.error(f"Error loading {image_path}: {e}")
#         return image_path, None

# # Function to infer a batch of images
# def infer_batch(batch_images, model, device):
#     batch_tensor = torch.stack(batch_images).to(device)
#     with torch.no_grad():
#         with autocast():
#             embeddings = model(batch_tensor).cpu().numpy()
#     return embeddings

# # Function to process images on a single GPU
# def process_images_on_gpu(image_paths, gpu_index, output_file, batch_size):
#     # Set the device
#     device = torch.device(f'cuda:{gpu_index}')
#     logging.info(f"Process {gpu_index}: Using device {device}")

#     # Load the model onto the specific GPU
#     model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg').to(device)
#     model.eval()

#     # Use ThreadPoolExecutor to load and preprocess images in parallel
#     with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#         results = list(tqdm(executor.map(load_and_transform_image, image_paths), total=len(image_paths), desc=f"GPU {gpu_index}: Loading images"))
    
#     # Filter out failed image loads
#     valid_results = [(path, img_tensor) for path, img_tensor in results if img_tensor is not None]
#     if not valid_results:
#         logging.error(f"Process {gpu_index}: No valid images to process.")
#         return
#     all_paths, all_tensors = zip(*valid_results)

#     # Process images in batches and save embeddings incrementally
#     logging.info(f"Process {gpu_index}: Processing images in batches of {batch_size}...")
#     with open(output_file, 'w') as f:
#         for i in tqdm(range(0, len(all_tensors), batch_size), desc=f"GPU {gpu_index}: Processing batches"):
#             batch_paths = all_paths[i:i + batch_size]
#             batch_tensors = all_tensors[i:i + batch_size]
#             print(len(batch_tensors))

#             # Infer embeddings
#             batch_embeddings = infer_batch(batch_tensors, model, device)

#             # Save embeddings incrementally
#             for path, embedding in zip(batch_paths, batch_embeddings):
#                 entry = {path: embedding.tolist()}
#                 f.write(json.dumps(entry) + "\n")

#     logging.info(f"Process {gpu_index}: Processing complete.")

# # Function to process images using multiple GPUs
# def process_images_multigpu(csv_path, output_dir, batch_size=32):
#     # Read the CSV file and extract image paths
#     df = pd.read_csv(csv_path)
#     image_paths = df['rgb_filepath'].tolist()

#     logging.info(f"Found {len(image_paths)} images in the CSV.")

#     num_gpus = torch.cuda.device_count()
#     if num_gpus == 0:
#         logging.error("No GPUs available.")
#         return

#     logging.info(f"Using {num_gpus} GPUs.")

#     # Split image paths into chunks
#     chunks = [image_paths[i::num_gpus] for i in range(num_gpus)]
    
#     # Prepare output directory
#     if not os.path.exists(output_dir) and output_dir != '':
#         os.makedirs(output_dir)

#     # Start a process for each GPU
#     processes = []
#     for i in range(num_gpus):
#         gpu_output_file = os.path.join(output_dir, f'dinov2_embeddings_gpu{i}.jsonl')
#         p = multiprocessing.Process(target=process_images_on_gpu, args=(chunks[i], i, gpu_output_file, batch_size))
#         processes.append(p)
#         p.start()

#     # Wait for all processes to finish
#     for p in processes:
#         p.join()

#     # Combine the output files
#     combined_output_file = os.path.join(output_dir, 'dinov2_embeddings.jsonl')
#     with open(combined_output_file, 'w') as outfile:
#         for i in range(num_gpus):
#             gpu_output_file = os.path.join(output_dir, f'dinov2_embeddings_gpu{i}.jsonl')
#             with open(gpu_output_file, 'r') as infile:
#                 for line in infile:
#                     outfile.write(line)
#             # Optionally delete the individual GPU files
#             os.remove(gpu_output_file)

#     logging.info("All processes complete. Combined embeddings saved.")

# # Input and output paths
# csv_path = "csv/filtered_data.csv"  # Path to the CSV file with image paths
# output_dir = "checkpoint"  # Directory to save embeddings

# # Process and save embeddings
# if __name__ == '__main__':
#     process_images_multigpu(csv_path, output_dir, 256)

