import gradio as gr
import pandas as pd
from PIL import Image
from multiprocessing import Pool, Manager
import tqdm

# Read the CSV file
csv = 'final_data1.csv'
selected_images = pd.read_csv(csv)[:10000]

# Ensure paths are valid for demonstration (adjust this part as necessary)
selected_images['rgb_filepath'] = selected_images['rgb_filepath'].apply(lambda x: x.strip())

def create_thumbnail_in_memory(image_info):
    """Create a thumbnail in memory for a given image."""
    image_path, idx = image_info
    try:
        image = Image.open(image_path)
        image.thumbnail((500, 500))  # Resize for thumbnails
        return image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def generate_thumbnails_in_memory():
    """Generate thumbnails in memory with progress tracking."""
    image_info = [(row['rgb_filepath'], idx) for idx, row in selected_images.iterrows()]
    images = []
    with Manager() as manager:
        with Pool() as pool:
            for result in tqdm.tqdm(pool.imap(create_thumbnail_in_memory, image_info), total=len(image_info), desc="Generating Thumbnails"):
                if result:
                    images.append(result)
    return images

# Generate thumbnails in memory
print("Generating thumbnails in memory...")
thumbnails = generate_thumbnails_in_memory()

def display_gallery():
    """Return the in-memory thumbnails."""
    return thumbnails

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(f"# Dataset Gallery {csv}")

    gallery = gr.Gallery(label="Gallery of Thumbnails", columns=8, height=700)  # Grid with 6 columns per row

    # Use in-memory thumbnails
    demo.load(fn=display_gallery, inputs=[], outputs=gallery)

demo.launch(server_port=7860)
