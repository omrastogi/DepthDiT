import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import download_file, check_links
from urls import depth_files, rgb_files
import argparse

def process_files(files, output_dir):
    """
    Check if all the links are working and download the files.
    
    Args:
        files (list): List of dictionaries containing 'url' and 'rel_path'.
        output_dir (str): Path to the directory where files will be downloaded.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Check if all the links are working
    print("Testing links...")
    check_links(files)

    # Process all files for downloading
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files for MatrixCity dataset")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory to save the downloaded files')
    parser.add_argument('--rgb', action='store_true', help='Download RGB files')
    parser.add_argument('--depth', action='store_true', help='Download Depth files')
    args = parser.parse_args()

    output_dir = args.dataset_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.rgb:
        process_files(rgb_files, output_dir)
    elif args.depth:
        process_files(depth_files, output_dir)
    else:
        print("Nothing selected")


