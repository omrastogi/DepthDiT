import os
import requests
from tqdm import tqdm

def check_links(files):
    """
    Iterates over the files dictionary to check if all links are working.
    Alerts on any broken links.
    """
    broken_links = []

    for file in files:
        url = file["url"]
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            if response.status_code != 200:
                broken_links.append((url, response.status_code))
                print(f"Broken link detected: {url} (Status Code: {response.status_code})")
            else:
                print(f"Link working: {url}")
        except requests.RequestException as e:
            broken_links.append((url, str(e)))
            print(f"Error checking link: {url} (Error: {e})")

    if broken_links:
        print("\nALERT: The following links are not working:")
        for url, error in broken_links:
            print(f"URL: {url}, Error: {error}")
    else:
        print("\nAll links are working fine!")

def download_file(url, output_path):
    """
    Download a file from a URL to a local path with a progress bar.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(output_path, "wb") as f, tqdm(
        desc=f"Downloading {os.path.basename(output_path)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))
