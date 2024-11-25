
import gradio as gr
import pandas as pd
from PIL import Image
from multiprocessing import Pool, Manager
import tqdm
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Gradio Grid View with Flagging")
parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing image data')
parser.add_argument('--first_k', type=int, default=500, help='Number of images to display from the CSV file')
parser.add_argument('--port', type=int, default=7860, help='Port to run the Gradio app on')

args = parser.parse_args()

# Get base dir
assert 'BASE_DIR' in os.environ, "Error: 'BASE_DIR' is not set in the environment variables."
base_dir = os.getenv("BASE_DIR")
assert os.path.exists(base_dir), f"Error: The directory {base_dir} does not exist."

# Read the CSV file
csv = args.csv_path
port = 7860
print(f"Loading Gallery for {csv} on {port}")
selected_images = pd.read_csv(csv)[:args.first_k]

# Ensure paths are valid for demonstration (adjust this part as necessary)
selected_images['rgb_file'] = selected_images['rgb_file'].apply(lambda x: x.strip())
selected_images['rgb_file'] = selected_images['rgb_file'].apply(lambda x: os.path.join(base_dir, x))

# In-memory dictionary to store flagged images
flagged_images = {}

def create_thumbnail_in_memory(image_info):
    """Create a thumbnail in memory for a given image."""
    image_path, idx = image_info
    try:
        image = Image.open(image_path)
        image.thumbnail((500, 500))  # Resize for larger thumbnails
        return idx, image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return idx, None

def generate_thumbnails_in_memory():
    """Generate thumbnails in memory with progress tracking."""
    image_info = [(row['rgb_file'], idx) for idx, row in selected_images.iterrows()]
    images = []
    with Manager() as manager:
        with Pool() as pool:
            for idx, result in tqdm.tqdm(pool.imap(create_thumbnail_in_memory, image_info), total=len(image_info), desc="Generating Thumbnails"):
                if result:
                    images.append((idx, result))
    return images

def toggle_flag(image_id, flag):
    """Toggle the flag status for a given image."""
    image_id = int(image_id)
    rgb_filepath = selected_images.iloc[image_id]['rgb_filepath']
    if flag:
        flagged_images[image_id] = rgb_filepath
    else:
        flagged_images.pop(image_id, None)

def get_flagged_filepaths():
    """Retrieve the file paths of all flagged images."""
    return list(flagged_images.values())

# Generate thumbnails in memory
print("Generating thumbnails in memory...")
thumbnails = [img for _, img in generate_thumbnails_in_memory() if img is not None]

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(f"# Dataset Gallery {csv}")

    with gr.Row():
        for idx, thumbnail in enumerate(thumbnails):
            with gr.Column(scale=1, min_width=200):  # Adjust width for better layout
                gr.Image(value=thumbnail, label=f"Image ID: {idx}", interactive=False)
                hidden_image_id = gr.Textbox(value=str(idx), visible=False)  # Hidden image ID
                checkbox_flag = gr.Checkbox(
                    label="Flag this Image",
                    value=False,
                    interactive=True,
                )
                # Correctly pass the checkbox state and the hidden image ID
                checkbox_flag.change(
                    fn=toggle_flag,
                    inputs=[hidden_image_id, checkbox_flag],
                    outputs=[],
                )
    
    with gr.Row():
        flagged_button = gr.Button("Show Flagged Filepaths")
        flagged_output = gr.Textbox(label="Flagged Filepaths", lines=10, interactive=False)

        flagged_button.click(
            fn=get_flagged_filepaths,
            inputs=[],
            outputs=flagged_output,
        )
print("Launching App...")
demo.launch(server_port=port, share=True)
