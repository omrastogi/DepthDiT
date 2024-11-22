
import gradio as gr
import pandas as pd
from PIL import Image
from multiprocessing import Pool, Manager
import tqdm

# Read the CSV file
csv = 'final_data1.csv'
port = 7860
print(f"Loading Gallery for {csv} on {port}")
selected_images = pd.read_csv(csv)[:5000]

# Ensure paths are valid for demonstration (adjust this part as necessary)
selected_images['rgb_filepath'] = selected_images['rgb_filepath'].apply(lambda x: x.strip())

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
    image_info = [(row['rgb_filepath'], idx) for idx, row in selected_images.iterrows()]
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
