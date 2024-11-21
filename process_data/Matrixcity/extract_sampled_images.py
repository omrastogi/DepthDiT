import os
import tarfile
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count


def process_tar_file(args):
    """
    Process a single TAR file to extract files listed in the CSV.

    Args:
        tar_file (str): Path to the TAR file.
        group (pd.DataFrame): DataFrame group for this TAR file.
        base_dir (str): Base directory for the source.
        destination_root (str): Destination root directory.
        log_file (str): Path to the log file for recording missing and successful files.
        start_index (int): Starting index for log entries.

    Returns:
        int: The number of files processed in this TAR file.
    """
    tar_file, group, base_dir, destination_root, log_file, start_index = args
    # print(f"Processing TAR file: {tar_file}")
    current_index = start_index

    with open(log_file, "a") as log:
        try:
            with tarfile.open(tar_file, 'r') as tar:
                tar_members = {member.name for member in tar.getmembers()}  # Collect all member names for checking
                for _, row in group.iterrows():
                    relative_path = row["File Path Inside Tar"]
                    # Construct the hierarchical destination path
                    dest_path = os.path.join(
                        destination_root,
                        os.path.relpath(os.path.dirname(tar_file), base_dir),
                        relative_path.lstrip("./")
                    )

                    # Ensure the destination directory exists
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

                    # Extract the file from the TAR archive
                    try:
                        tar_member = tar.getmember(relative_path)
                        with tar.extractfile(tar_member) as file:
                            # Write to destination
                            with open(dest_path, "wb") as out_file:
                                out_file.write(file.read())
                        log.write(f"{current_index}: SUCCESS: Extracted {relative_path} -> {dest_path}\n")
                        print(f"{current_index}: Extracted: {relative_path} -> {dest_path}")
                    except KeyError:
                        log.write(f"{current_index}: ERROR: File not found: {relative_path} in TAR: {tar_file}\n")
                        print(f"{current_index}: ERROR: File not found: {relative_path} in TAR: {tar_file}\n")
                        
                        # Check if the file name exists in any member
                        file_name = os.path.basename(relative_path)
                        similar_members = [m for m in tar_members if file_name in m]
                        if similar_members:
                            log.write(f"  Potential matches for '{file_name}' in TAR:\n")
                            for match in similar_members:
                                log.write(f"    - {match}\n")
                        else:
                            log.write(f"  No matches found for '{file_name}' in TAR.\n")
                    except Exception as e:
                        log.write(f"{current_index}: ERROR: Failed to extract {relative_path} from {tar_file}: {e}\n")
                    current_index += 1
        except FileNotFoundError:
            log.write(f"{current_index}: ERROR: TAR file not found: {tar_file}\n")
        except Exception as e:
            log.write(f"{current_index}: ERROR: Failed to process TAR file {tar_file}: {e}\n")

    return current_index - start_index


def extract_files_from_tar_with_logging_multiprocessing(csv_path, root_dir, base_dir, destination_root, log_file):
    """
    Extract files listed in a CSV from TAR archives to the destination directory using multiprocessing.
    Logs missing files, successful extractions, and checks for existence in the TAR members.

    Args:
        csv_path (str): Path to the CSV file.
        base_dir (str): Base directory for the TAR files.
        destination_root (str): Root directory where files will be copied.
        log_file (str): Path to the log file for recording missing and successful files.
    """
    try:
        # Read the CSV
        df = pd.read_csv(csv_path)

        # Ensure required columns are in the CSV
        if "Tar File" not in df.columns or "File Path Inside Tar" not in df.columns:
            print("CSV must contain 'Tar File' and 'File Path Inside Tar' columns.")
            return

        # Group files by TAR file
        grouped = df.groupby("Tar File")

        # Prepare arguments for multiprocessing
        args = []
        current_index = 1  # Start index for logging
        for tar_file, group in grouped:
            # Check if tar_file is an absolute path, if not, prepend it with root_dir
            if not os.path.isabs(tar_file):
                tar_file = os.path.join(root_dir, tar_file)
            args.append((tar_file, group, base_dir, destination_root, log_file, current_index))
            current_index += len(group)

        # Use multiprocessing to process TAR files
        with Pool(cpu_count()) as pool:
            pool.map(process_tar_file, args)

        print("Extraction process completed.")
        print(f"Total files processed: {current_index - 1}")

    except Exception as e:
        print(f"Error reading CSV file: {e}")

from PIL import Image

import os
import pandas as pd
from PIL import Image
import tarfile
from multiprocessing import Pool, cpu_count


def check_and_reextract_corrupted_images(csv_path, root_dir, base_dir, destination_root, log_file):
    """
    Check for corrupted images in the destination directory.
    If an image is corrupted or missing, re-extract it from the TAR file.
    Log every check and re-extraction.

    Args:
        csv_path (str): Path to the CSV file.
        root_dir (str): Root directory for TAR files.
        base_dir (str): Base directory for the TAR files.
        destination_root (str): Root directory where files are extracted.
        log_file (str): Path to the log file for recording checks and actions.
    """
    try:
        # Read the CSV
        df = pd.read_csv(csv_path)

        # Ensure required columns are in the CSV
        if "Tar File" not in df.columns or "File Path Inside Tar" not in df.columns:
            print("CSV must contain 'Tar File' and 'File Path Inside Tar' columns.")
            return

        # Open the log file
        with open(log_file, "a") as log:
            for _, row in df.iterrows():
                relative_path = row["File Path Inside Tar"]
                tar_file = row["Tar File"]
                # Check if tar_file is an absolute path, if not, prepend it with root_dir
                if not os.path.isabs(tar_file):
                    tar_file = os.path.join(root_dir, tar_file)
                # Construct the hierarchical destination path
                dest_path = os.path.join(
                    destination_root,
                    os.path.relpath(os.path.dirname(tar_file), base_dir),
                    relative_path.lstrip("./")
                )

                # Log the check
                log.write(f"Checking file: {dest_path}\n")

                # Check if the file exists
                if not os.path.exists(dest_path):
                    log.write(f"  ERROR: File does not exist: {dest_path}\n")
                    # Attempt to re-extract the file
                    try:
                        with tarfile.open(tar_file, 'r') as tar:
                            tar_member = tar.getmember(relative_path)
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            with tar.extractfile(tar_member) as file:
                                with open(dest_path, "wb") as out_file:
                                    out_file.write(file.read())
                        log.write(f"  Re-extracted missing file: {dest_path}\n")
                    except Exception as e:
                        log.write(f"  ERROR: Failed to re-extract file {relative_path} from {tar_file}: {e}\n")
                    continue

                # Try to open the image
                try:
                    with Image.open(dest_path) as img:
                        img.verify()  # Verify that it's a valid image
                    log.write(f"  OK: Image is valid.\n")
                except Exception as e:
                    log.write(f"  ERROR: Corrupted image detected: {dest_path}: {e}\n")
                    # Attempt to re-extract the file
                    try:
                        with tarfile.open(tar_file, 'r') as tar:
                            tar_member = tar.getmember(relative_path)
                            with tar.extractfile(tar_member) as file:
                                with open(dest_path, "wb") as out_file:
                                    out_file.write(file.read())
                        log.write(f"  Re-extracted corrupted file: {dest_path}\n")
                        # Re-check the image
                        try:
                            with Image.open(dest_path) as img:
                                img.verify()
                            log.write(f"  After re-extraction, image is valid.\n")
                        except Exception as e2:
                            log.write(f"  ERROR: Still corrupted after re-extraction: {dest_path}: {e2}\n")
                    except Exception as e1:
                        log.write(f"  ERROR: Failed to re-extract file {relative_path} from {tar_file}: {e1}\n")

        print("Image corruption check completed.")
    except Exception as e:
        print(f"Error during corruption check: {e}")


# Example usage
csv_path = "csv/sampled_data.csv"  # Path to the CSV file
assert 'BASE_DIR' in os.environ, "Error: 'BASE_DIR' is not set in the environment variables."
root_dir = os.getenv("BASE_DIR")
assert os.path.exists(root_dir), f"Error: The directory {root_dir} does not exist."
base_dir = f"{root_dir}/big_city"  # Base directory for TAR files
destination_root = f"{root_dir}/sampled_big_city"  # Destination directory
log_file = "csv/extraction.log"  # Path to the log file
# Check if log file exists and delete it
if os.path.exists(log_file):
    os.remove(log_file)

extract_files_from_tar_with_logging_multiprocessing(csv_path, root_dir, base_dir, destination_root, log_file)
check_and_reextract_corrupted_images(csv_path, root_dir, base_dir, destination_root, log_file)
