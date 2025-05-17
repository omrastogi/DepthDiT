import os
import tarfile
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def process_tar_member(member, tar_path):
    """
    Process a single tar member to check if it is a PNG file.
    """
    if member.isfile() and member.name.endswith(".png"):
        return {
            "Tar File": tar_path,
            "File Path Inside Tar": member.name
        }
    return None


def process_tar_file(tar_path):
    """
    Process a single tar file to extract PNG file paths using multiprocessing.
    """
    file_list = []
    try:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()  # Get all tar members

            # Use multiprocessing to process members
            with Pool(cpu_count()) as pool:
                results = pool.starmap(process_tar_member, [(member, tar_path) for member in members])

            # Filter out None results
            file_list = [result for result in results if result is not None]

    except Exception as e:
        print(f"Error processing {tar_path}: {e}")

    return file_list

def extract_tar_and_list_files(base_dir):
    """
    Extract PNG files from all .tar files in the directory sequentially.
    """
    tar_files = []

    # Collect all .tar files
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".tar") or file.startswith(".tar"):
                tar_files.append(os.path.join(root, file))

    file_list = []
    
    # Process tar files sequentially
    for tar_file in tqdm(tar_files, desc="Processing TAR files"):
        start_time = time.time()
        result = process_tar_file(tar_file)
        end_time = time.time()
        relative_path = os.path.relpath(tar_file, base_dir)
        print(f"FILE: {relative_path}  FILES: {len(result)}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        file_list.extend(result)

    return file_list


import time

if __name__ == "__main__":
    assert 'BASE_DIR' in os.environ, "Error: 'BASE_DIR' is not set in the environment variables."
    base_dir = os.getenv("BASE_DIR")
    assert os.path.exists(base_dir), f"Error: The directory {base_dir} does not exist."
    
    print("Scanning and processing tar files...")
    file_list = extract_tar_and_list_files(base_dir)
    
    # # Save the list to a CSV file
    output_csv_path = "csv/tar_file_contents.csv"
    print(f"Saving results to {output_csv_path}...")
    df = pd.DataFrame(file_list)
    df.to_csv(output_csv_path, index=False)
    print("CSV file saved successfully.")
