import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse

from utils import download_file, check_links

# Full list of RGB file links
files = [
    # Aerial Test
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/test/big_high_block_1_test.tar",
        "rel_path": "big_city/aerial/test"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/test/big_high_block_2_test.tar",
        "rel_path": "big_city/aerial/test"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/test/big_high_block_3_test.tar",
        "rel_path": "big_city/aerial/test"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/test/big_high_block_4_test.tar",
        "rel_path": "big_city/aerial/test"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/test/big_high_block_5_test.tar",
        "rel_path": "big_city/aerial/test"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/test/big_high_block_6_test.tar",
        "rel_path": "big_city/aerial/test"
    },

    # Aerial Train
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/train/big_high_block_1.tar00",
        "rel_path": "big_city/aerial/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/train/big_high_block_1.tar01",
        "rel_path": "big_city/aerial/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/train/big_high_block_1.tar02",
        "rel_path": "big_city/aerial/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/train/big_high_block_1.tar03",
        "rel_path": "big_city/aerial/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/train/big_high_block_1.tar04",
        "rel_path": "big_city/aerial/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/train/big_high_block_2.tar",
        "rel_path": "big_city/aerial/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/train/big_high_block_3.tar",
        "rel_path": "big_city/aerial/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/train/big_high_block_4.tar",
        "rel_path": "big_city/aerial/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/train/big_high_block_5.tar",
        "rel_path": "big_city/aerial/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/aerial/train/big_high_block_6.tar",
        "rel_path": "big_city/aerial/train"
    },

    # Street Test
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/test/top_area_test.tar",
        "rel_path": "big_city/street/test"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/test/bottom_area_test.tar",
        "rel_path": "big_city/street/test"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/test/right_area_test.tar",
        "rel_path": "big_city/street/test"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/test/left_area_test.tar",
        "rel_path": "big_city/street/test"
    },

    # Street Train
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/bottom_area.tar00",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/bottom_area.tar01",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/bottom_area.tar02",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/left_area.tar00",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/left_area.tar01",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/left_area.tar03",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/left_area.tar04",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/left_area.tar05",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/left_area.tar06",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/right_area.tar00",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/right_area.tar01",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/right_area.tar02",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/right_area.tar04",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/top_area.tar00",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/top_area.tar01",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/top_area.tar02",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/top_area.tar03",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/top_area.tar04",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/top_area.tar05",
        "rel_path": "big_city/street/train"
    },
    {
        "url": "https://huggingface.co/datasets/BoDai/MatrixCity/resolve/main/big_city/street/train/top_area.tar06",
        "rel_path": "big_city/street/train"
    }
]

# Directory to save the downloaded files

parser = argparse.ArgumentParser(description="Download RGB files for MatrixCity dataset")
parser.add_argument('--dataset_dir', type=str, required=True, help='Directory to save the downloaded files')
args = parser.parse_args()

output_dir = args.dataset_dir
os.makedirs(output_dir, exist_ok=True)

# Check if all the links are working
print("Testing links")
check_links(files)

# Process all files
print("Starting Downloads...")
for file in files:
    url = file["url"]
    rel_path = file["rel_path"]
    
    # Create the full output directory path
    full_dir = os.path.join(output_dir, rel_path)
    os.makedirs(full_dir, exist_ok=True)
    
    # Determine the output file path
    file_name = os.path.basename(url)
    output_path = os.path.join(full_dir, file_name)
    
    # Skip download if file already exists
    if os.path.exists(output_path):
        print(f"File {output_path} already exists, skipping download.")
        continue
    
    # Download the file
    download_file(url, output_path)

print("All RGB files downloaded successfully!")
