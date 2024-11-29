import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def generate_links(file_path, data_type):
    # Base URL
    base_url = "https://huggingface.co/datasets/omrastogi/Hypersim-Processed/resolve/main"
    
    # List to store the generated links
    links = []

    # Read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into RGB and Depth paths
            rgb_path, depth_path = line.strip().split()
            
            # Generate the full URLs
            rgb_url = f"{base_url}/{data_type}/{rgb_path}"
            depth_url = f"{base_url}/{data_type}/{depth_path}"
            
            # Add the URLs to the list
            links.append(rgb_url)
            links.append(depth_url)
    
    return links

def download_file(base_path, url):
    """
    Downloads a single file from the given URL and saves it in the specified base path.
    
    Args:
    - base_path (str): The base directory where the file will be saved.
    - url (str): URL of the file to download.
    
    Returns:
    - str: Status message indicating success or failure.
    """
    # Extract the relative path from the URL
    relative_path = "/".join(url.split("main/")[-1].split("/"))
    
    # Create the full path for saving the file
    file_path = os.path.join(base_path, relative_path)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Download the file
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        return f"Downloaded: {file_path}"
    except Exception as e:
        return f"Failed to download: {url} (Error: {e})"

def download_and_save_files_parallel(base_path, urls, max_workers=8):
    """
    Downloads files from the provided URLs in parallel and saves them in the specified base path.
    
    Args:
    - base_path (str): The base directory where files will be saved.
    - urls (list): List of URLs to download.
    - max_workers (int): Maximum number of threads to use for parallel downloads.
    """
    # Use ThreadPoolExecutor to download files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Initialize tqdm progress bar
        with tqdm(total=len(urls), desc="Downloading files", unit="file", ascii=True) as progress:
            # Submit download tasks to the executor
            futures = {executor.submit(download_file, base_path, url): url for url in urls}
            for future in futures:
                result = future.result()  # Wait for each thread to complete
                print(result)  # Log the result
                progress.update(1)  # Update the progress bar

def download_and_save_files(base_path, urls):
    """
    Downloads files from the provided URLs and saves them in the specified base path, with a progress bar.
    
    Args:
    - base_path (str): The base directory where files will be saved.
    - urls (list): List of URLs to download.
    """
    # Use tqdm for the entire list of URLs
    for url in tqdm(urls, desc="Downloading files", unit="file", ascii=True):
        # Extract the relative path from the URL
        relative_path = "/".join(url.split("main/")[-1].split("/"))
        
        # Create the full path for saving the file
        file_path = os.path.join(base_path, relative_path)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Download the file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print(f"Downloaded: {file_path}")
        else:
            print(f"Failed to download: {url} (Status code: {response.status_code})")

# Specify the file path and data type
base_path = "/home/omrastogi/om/data/Hypersim"
file_path = "data_split/filename_list_train_filtered_revised.txt"  # Replace with your file's path
data_type = "train"     # Specify the data type (e.g., "train")

# Generate the links

# print(urls)
# Example usage
urls = generate_links(file_path, data_type)
download_and_save_files_parallel(base_path, urls)
# download_and_save_files(base_path, urls)

