import os
import shutil
import pandas as pd

# List of CSV files to process
csv_files = [
    "csv/final_data_aerial_test_0.5k.csv",  # Replace with the path to your first CSV
    "csv/final_data_aerial_train_5k.csv",  # Replace with the path to your second CSV
    "csv/final_data_street_test_1k.csv",  # Replace with the path to your third CSV
    "csv/final_data_street_train_20k.csv",  # Replace with the path to your fourth CSV
]

# Base directory and new base directory
base_dir = "/mnt/c1e1833e-4df6-4c4c-88aa-8cd3d7d3932b/om/MatrixCity/MatrixCity"  # Replace with the original base directory path
new_base_dir = "/mnt/51eb0667-f71d-4fe0-a83e-beaff24c04fb/om/depth_estimation_experiments/depth_datasets/Matrixcity"

# Function to copy files
def copy_files(row, base_dir, new_base_dir):
    rgb_src = os.path.join(base_dir, row["rgb_file"])
    depth_src = os.path.join(base_dir, row["depth_file"])
    
    rgb_dest = os.path.join(new_base_dir, row["rgb_file"])
    depth_dest = os.path.join(new_base_dir, row["depth_file"])
    
    # Ensure the destination directories exist
    os.makedirs(os.path.dirname(rgb_dest), exist_ok=True)
    os.makedirs(os.path.dirname(depth_dest), exist_ok=True)
    
    # Copy the files
    shutil.copy2(rgb_src, rgb_dest)
    shutil.copy2(depth_src, depth_dest)

# Process each CSV
from tqdm import tqdm

for csv_file in csv_files:
    print(f"Processing {csv_file}...")
    # Read the current CSV file
    df = pd.read_csv(csv_file)
    
    # Iterate through each row in the dataframe and copy files with progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying files from {csv_file}"):
        copy_files(row, base_dir, new_base_dir)

print(f"Files from all CSVs copied successfully to {new_base_dir}")
