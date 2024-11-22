# %%
import os
import numpy as np
import pandas as pd
from PIL import Image
from aesthetic_scorer import AestheticScorer
from utils import load_depth, convert_image_path, convert_tar_path, convert_tar_path_to_sampled


df = pd.read_csv('csv/sampled_data.csv')
assert 'BASE_DIR' in os.environ, "Error: 'BASE_DIR' is not set in the environment variables."
base_path = os.getenv("BASE_DIR")
assert os.path.exists(base_path), f"Error: The directory {base_path} does not exist."

#----------------------------------------------------------------------------

#%%
from joblib import Parallel, delayed
from tqdm import tqdm
from PIL import Image
from numba import njit

@njit
def fast_histogram(data, bin_edges):
    histogram = np.zeros(len(bin_edges) - 1, dtype=np.int32)
    for value in data.ravel():
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= value < bin_edges[i + 1]:
                histogram[i] += 1
                break
    return histogram

# Initialize the scorer
aesthetic_scorer = AestheticScorer()

# Precompute paths
df['tar_file_path'] = df['Tar File'].apply(lambda x: os.path.join(base_path, x) if not os.path.isabs(x) else x)
df['file_path'] = df.apply(lambda row: os.path.join(convert_tar_path(row['tar_file_path']), convert_image_path(row['File Path Inside Tar'])), axis=1)
df['rgb_file_path'] = df.apply(lambda row: os.path.join(convert_tar_path_to_sampled(row['tar_file_path']), row['File Path Inside Tar'].lstrip("./")), axis=1)

# User-defined batch size
BATCH_SIZE = 16  # Specify the number of images per batch
NUM_IMAGES = len(df)

# Function to load files for a batch
def load_files_for_batch(batch_rows):
    return Parallel(n_jobs=-1)(
        delayed(load_files)(row) for row in batch_rows
    )

# Function to load files (I/O for a single row)
def load_files(row):
    rgb_image = Image.open(row.rgb_file_path).convert("RGB")
    depth_image, invalid_mask = load_depth(row.file_path)
    return rgb_image, depth_image

# Initialize results
results = []
bin_edges = np.linspace(0, 1, 11)

try:
    for start in tqdm(range(0, NUM_IMAGES, BATCH_SIZE), desc="Processing batches"):
        end = min(start + BATCH_SIZE, NUM_IMAGES)
        batch_rows = df.iloc[start:end].itertuples(index=False)

        # Load batch of files in parallel
        loaded_data = Parallel(n_jobs=-1, prefer="threads")(
            delayed(load_files)(row) for row in batch_rows
        )
        rgb_images, depth_images = zip(*loaded_data)

        # Perform batch inference for aesthetic scores
        aesthetic_scores = aesthetic_scorer.score_images(rgb_images)

        # Sequentially process the batch for depth-related calculations
        for depth_image, aesthetic_score in zip(depth_images, aesthetic_scores):
            depth_image_real = depth_image.copy()
            mean_depth = np.mean(depth_image)
            median_depth = np.median(depth_image)
            variance_depth = np.var(depth_image)
            depth_image_norm = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
            depth_image_clipped = np.clip(depth_image_norm, 0, 1)
            # histogram, bin_edges = np.histogram(depth_image_clipped, bins=10, range=(0, 1))
            histogram = fast_histogram(depth_image_norm, bin_edges)
            count_above_4000 = (histogram > 4000).sum()
            count_above_1000 = (histogram > 1000).sum()
            variance_histogram = np.var(histogram)

            result = {
                "Mean Depth": mean_depth,
                "Median Depth": median_depth,
                "Variance Depth": variance_depth,
                "Aesthetic Score": aesthetic_score,
                "Max Depth": depth_image_real.max(),
                "Min Depth": depth_image_real.min(),
                "Above 1000": count_above_1000,
                "Above 4000": count_above_4000,
                "Variance in Histogram": variance_histogram
            }
            results.append(result)

except KeyboardInterrupt:
    print("Process interrupted by user. Saving progress...")
    results_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    df.to_csv('csv/featured_sampled_data_partial.csv', index=False)
    exit()

# Save results to a DataFrame and CSV
results_df = pd.DataFrame(results)
df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
df.to_csv('csv/featured_sampled_data.csv', index=False)

#----------------------------------------------------------------------------
# # %% # Simpler Version, without optimizations
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.cm as cm
# from tqdm import tqdm

# aesthetic_scorer = AestheticScorer()
# results = []
# df['tar_file_path'] = df['Tar File'].apply(lambda x: os.path.join(base_path, x) if not os.path.isabs(x) else x)
# df['file_path'] = df.apply(lambda row: os.path.join(convert_tar_path(row['tar_file_path']), convert_image_path(row['File Path Inside Tar'])), axis=1)
# df['rgb_file_path'] = df.apply(lambda row: os.path.join(convert_tar_path_to_sampled(row['tar_file_path']), row['File Path Inside Tar'].lstrip("./")), axis=1)

# for row in tqdm(df.itertuples(index=False), total=len(df)):
#     rgb_image = Image.open(row.rgb_file_path).convert("RGB")
#     depth_image, invalid_mask = load_depth(row.file_path)
#     depth_image_real = depth_image.copy()
#     aesthetic_score = aesthetic_scorer.score_image(rgb_image)

#     mean_depth = np.mean(depth_image)
#     median_depth = np.median(depth_image)
#     variance_depth = np.var(depth_image)
#     depth_image_norm = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
#     depth_image_clipped = np.clip(depth_image_norm, 0, 1)
#     histogram, bin_edges = np.histogram(depth_image_clipped, bins=10, range=(0, 1))
#     count_above_4000 = (histogram > 4000).sum()
#     count_above_1000 = (histogram > 1000).sum()
#     variance_histogram = np.var(histogram)

#     result = {
#         "Mean Depth": mean_depth,
#         "Median Depth": median_depth,
#         "Variance Depth": variance_depth,
#         "Aesthetic Score": aesthetic_score,
#         "Max Depth": depth_image_real.max(),
#         "Min Depth": depth_image_real.min(),
#         "Above 1000": count_above_1000,
#         "Above 4000": count_above_4000,
#         "Variance in Histogram": variance_histogram
#     }
#     results.append(result)
        
#     depth_image_norm = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())

# results_df = pd.DataFrame(results)
# df = pd.concat([df, results_df], axis=1)
# df.to_csv('csv/featured_sampled_data.csv', index=False)
#----------------------------------------------------------------------------
# %% Applying pre-analysed filters 
df = pd.read_csv("csv/featured_sampled_data.csv")
assert 'BASE_DIR' in os.environ, "Error: 'BASE_DIR' is not set in the environment variables."
base_path = os.getenv("BASE_DIR")
assert os.path.exists(base_path), f"Error: The directory {base_path} does not exist."

# Define the "Don't allow" conditions
conditions_to_exclude = (
    (df["Mean Depth"] < 0.05) |
    ((df["Mean Depth"] > 5) & (df["Location Type"] == "street")) |
    ((df["Mean Depth"] < 1) & (df["Location Type"] == "aerial")) |
    ((df["Mean Depth"] > 6) & (df["Location Type"] == "aerial")) |
    (df["Aesthetic Score"] < 3.7)
)
# Define the "Allow" conditions
conditions_to_include = (
    (df["Above 1000"] > 3) & (df["Above 4000"] > 2)
)
# Apply the filtering logic
filtered_df = df[~conditions_to_exclude & conditions_to_include]
rgb_files = []
for _, row in filtered_df.iterrows():
    tar_file = os.path.join(base_path, row["Tar File"]) if not os.path.isabs(row["Tar File"]) else row["Tar File"]
    file_path = os.path.join(convert_tar_path(tar_file), convert_image_path(row["File Path Inside Tar"]))
    rgb_file_path = os.path.join(convert_tar_path_to_sampled(tar_file), row["File Path Inside Tar"].lstrip("./"))
    rgb_files.append(rgb_file_path)

filtered_df['rgb_filepath'] = rgb_files
filtered_df.to_csv("csv/filtered_data.csv", index=False)
#----------------------------------------------------------------------------

# %%
