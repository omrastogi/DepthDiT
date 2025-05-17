import fiftyone as fo
import pandas as pd

# Path to your CSV file
csv_path = "csv/final_data.csv"

df = pd.read_csv(csv_path)

# Assuming your CSV has a column named 'filepath' with the image paths
# If your column has a different name, replace 'filepath' with the correct column name
image_paths = df[df["Location Type"]=="aerial"]['rgb_filepath'].tolist()

# Create a FiftyOne dataset from the image paths
dataset = fo.Dataset.from_images(image_paths, name="Matrix City")

# Launch the FiftyOne app to view the dataset
session = fo.launch_app(dataset, remote=True)
session.wait()
